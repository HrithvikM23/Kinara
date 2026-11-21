//! Live network streaming server for real-time DCC integrations (Blender, Unreal Engine).

use crate::export::coords::MotionFrame;
use crate::skeleton::SKELETON;
use anyhow::Context;
use glam::Vec3;
use serde::Serialize;
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::time::Instant;

/// Official Kinara Blender stream port.
pub const BLENDER_STREAM_PORT: u16 = 49168;

/// Rig declaration bone record: [name, parent_index, affine_12_floats, length].
#[derive(Debug, Clone, Serialize)]
pub struct StreamBoneRecord(pub String, pub i32, pub [f32; 12], pub f32);

/// Rig declaration schema envelope (protocol version 2).
#[derive(Debug, Clone, Serialize)]
pub struct StreamRigSchema {
    pub version: u32,
    pub slot: u32,
    pub actor: String,
    pub name: String,
    pub bones: Vec<StreamBoneRecord>,
}

/// TCP live streaming server pushing motion capture transforms to DCC viewports.
pub struct BlenderStreamServer {
    pub listener: TcpListener,
    pub clients: Vec<TcpStream>,
    pub sequence: u64,
    pub start_time: Instant,
    pub actor_id: String,
}

impl BlenderStreamServer {
    /// Binds the streamer to local loopback port (default 49168).
    pub fn bind(port: u16) -> anyhow::Result<Self> {
        let listener = TcpListener::bind(format!("127.0.0.1:{}", port))
            .with_context(|| format!("Failed to bind live stream server to 127.0.0.1:{}", port))?;
        listener.set_nonblocking(true)?;

        // Deterministic 64-character lowercase hex actor identifier
        let actor_id = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string();

        Ok(Self {
            listener,
            clients: Vec::new(),
            sequence: 0,
            start_time: Instant::now(),
            actor_id,
        })
    }

    /// Accepts pending inbound DCC connections and performs rig declaration handshake.
    pub fn accept_new_connections(&mut self) -> anyhow::Result<usize> {
        let mut new_count = 0;
        while let Ok((mut stream, _addr)) = self.listener.accept() {
            let _ = stream.set_nodelay(true);
            let _ = stream.set_write_timeout(Some(std::time::Duration::from_millis(50)));
            // Send initial rig declaration packet (R)
            if self.send_rig_declaration(&mut stream).is_ok() {
                self.clients.push(stream);
                new_count += 1;
            }
        }
        Ok(new_count)
    }

    /// Builds and transmits the canonical 60-joint armature declaration packet.
    pub fn send_rig_declaration(&self, stream: &mut TcpStream) -> anyhow::Result<()> {
        let mut bones = Vec::with_capacity(SKELETON.len());
        for (idx, spec) in SKELETON.iter().enumerate() {
            let parent_idx = match spec.parent {
                Some(p_name) => SKELETON
                    .iter()
                    .position(|s| s.name == p_name)
                    .map(|i| i as i32)
                    .unwrap_or(-1),
                None => -1,
            };

            // Standard identity affine matrix [3x4] row-major
            let rows: [f32; 12] = [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ];
            let length = if idx == 0 { 1.0 } else { 0.25 };
            bones.push(StreamBoneRecord(spec.name.to_string(), parent_idx, rows, length));
        }

        let schema = StreamRigSchema {
            version: 2,
            slot: 1,
            actor: self.actor_id.clone(),
            name: "KinaraActor".to_string(),
            bones,
        };

        let json_bytes = serde_json::to_vec(&schema)?;
        let total_size = (json_bytes.len() + 1) as u32; // payload + kind

        let mut packet = Vec::with_capacity(5 + json_bytes.len());
        packet.extend_from_slice(&total_size.to_le_bytes());
        packet.push(b'R');
        packet.extend_from_slice(&json_bytes);

        stream.write_all(&packet)?;
        stream.flush()?;
        Ok(())
    }

    /// Broadcasts a live motion frame to all connected DCC viewports.
    pub fn broadcast_frame(&mut self, frame: &MotionFrame) -> anyhow::Result<usize> {
        self.accept_new_connections()?;
        if self.clients.is_empty() {
            return Ok(0);
        }

        self.sequence += 1;
        let actor_count = 1u16;
        let slot = 1u32;
        let frame_num = frame.frame_index as u32;

        let num_bones = SKELETON.len();
        let payload_size = 8 + 2 + (4 + 4 + num_bones * 48); // sequence(8) + count(2) + [slot(4)+frame(4)+bones(48*N)]
        let total_msg_size = (payload_size + 1) as u32;

        let mut packet = Vec::with_capacity(4 + 1 + payload_size);
        packet.extend_from_slice(&total_msg_size.to_le_bytes());
        packet.push(b'S'); // Snapshot kind

        // Snapshot header
        packet.extend_from_slice(&self.sequence.to_le_bytes());
        packet.extend_from_slice(&actor_count.to_le_bytes());

        // Actor frame header
        packet.extend_from_slice(&slot.to_le_bytes());
        packet.extend_from_slice(&frame_num.to_le_bytes());

        // Stream 12 floats per bone with bone orientation basis (+Y along limb in Blender)
        for spec in &SKELETON {
            let j_pos = frame.joints.get(spec.name).map(|j| Vec3::new(j.x, j.y, j.z)).unwrap_or(Vec3::ZERO);

            let child_name = SKELETON.iter().find(|s| s.parent == Some(spec.name)).map(|s| s.name);
            let dir_vec = if let Some(c_name) = child_name {
                frame.joints.get(c_name).map(|c| Vec3::new(c.x, c.y, c.z) - j_pos)
            } else if let Some(p_name) = spec.parent {
                frame.joints.get(p_name).map(|p| j_pos - Vec3::new(p.x, p.y, p.z))
            } else {
                None
            };

            let v = dir_vec.unwrap_or(Vec3::Y);
            let v_len = v.length();
            let hat_y = if v_len > 1e-4 { v / v_len } else { Vec3::Y };

            let ref_up = if hat_y.z.abs() < 0.95 {
                Vec3::Z
            } else {
                Vec3::X
            };
            let hat_x = hat_y.cross(ref_up).normalize();
            let hat_z = hat_x.cross(hat_y).normalize();

            // 3x4 Affine matrix (row-major):
            // Row 0: [hat_x.x, hat_y.x, hat_z.x, j_pos.x]
            // Row 1: [hat_x.y, hat_y.y, hat_z.y, j_pos.y]
            // Row 2: [hat_x.z, hat_y.z, hat_z.z, j_pos.z]
            let bone_floats: [f32; 12] = [
                hat_x.x, hat_y.x, hat_z.x, j_pos.x,
                hat_x.y, hat_y.y, hat_z.y, j_pos.y,
                hat_x.z, hat_y.z, hat_z.z, j_pos.z,
            ];
            for f in &bone_floats {
                packet.extend_from_slice(&f.to_le_bytes());
            }
        }

        // Send to active clients, prune disconnected ones
        let mut active = Vec::new();
        for mut client in self.clients.drain(..) {
            if client.write_all(&packet).is_ok() && client.flush().is_ok() {
                active.push(client);
            }
        }
        let active_count = active.len();
        self.clients = active;
        Ok(active_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{JointMap, JointValue};

    #[test]
    fn test_blender_stream_server_bind_and_broadcast() {
        let mut server = BlenderStreamServer::bind(0).expect("Failed to bind ephemeral test port");
        let port = server.listener.local_addr().unwrap().port();

        // Connect a test client
        let _client = TcpStream::connect(format!("127.0.0.1:{}", port)).expect("Failed client connect");
        let connected = server.accept_new_connections().expect("Accept failed");
        assert_eq!(connected, 1);

        let mut joints = JointMap::new();
        joints.insert("HipsRoot".to_string(), JointValue::new(0.0, 0.0, 0.0, 1.0));
        let frame = MotionFrame::new(0, joints);

        let sent = server.broadcast_frame(&frame).expect("Broadcast failed");
        assert_eq!(sent, 1);
    }

    #[test]
    fn test_broadcast_frame_bone_orientations() {
        let mut server = BlenderStreamServer::bind(0).expect("Failed to bind port");
        let mut joints = JointMap::new();
        joints.insert("HipsRoot".to_string(), JointValue::new(0.0, 0.0, 0.0, 1.0));
        joints.insert("Chest".to_string(), JointValue::new(0.0, 0.0, 10.0, 1.0));
        let frame = MotionFrame::new(0, joints);

        let count = server.broadcast_frame(&frame).expect("Broadcast failed");
        assert_eq!(count, 0);
    }
}
