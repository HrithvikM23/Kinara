//! Coordinate transformation, scale normalization, and ground-plane alignment.
//! Direct translation of `utils/exports.py`.

use crate::skeleton::{JointMap, JointValue};
use glam::Vec3;
use serde::{Deserialize, Serialize};

/// A single animation/mocap frame consisting of a frame index and 3D joint dictionary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotionFrame {
    pub frame_index: usize,
    pub joints: JointMap,
}

impl MotionFrame {
    pub fn new(frame_index: usize, joints: JointMap) -> Self {
        Self {
            frame_index,
            joints,
        }
    }
}

/// Rotates 3D Cartesian coordinates from OpenCV camera space
/// (+X right, +Y down, +Z forward) to Blender world space (+X right, +Y forward, +Z up).
#[inline]
pub fn cv_to_blender_space(p: Vec3) -> Vec3 {
    Vec3::new(p.x, p.z, -p.y)
}

/// Inverts Y coordinate from 2D screen/image space to 3D world space (+X right, +Y up, +Z depth).
#[inline]
pub fn screen_to_world_space(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, -y, z)
}

/// Measures the 3D Euclidean distance between the hip root and neck base.
#[inline]
pub fn measure_torso_length(hips: Vec3, neck: Vec3) -> f32 {
    (neck - hips).length()
}

/// Measures the effective torso length in a named joint map.
/// Looks up `HipsRoot` and `Neck`; falls back to `Chest` distance divided by 0.70.
pub fn measure_frame_torso_length(joints: &JointMap) -> Option<f32> {
    let root = joints.get("HipsRoot")?;
    if !joint_is_valid(root) {
        return None;
    }
    let root_v = Vec3::new(root.x, root.y, root.z);

    if let Some(neck) = joints.get("Neck") {
        if joint_is_valid(neck) {
            let neck_v = Vec3::new(neck.x, neck.y, neck.z);
            let dist = (neck_v - root_v).length();
            if dist > 1.0 {
                return Some(dist);
            }
        }
    }

    if let Some(chest) = joints.get("Chest") {
        if joint_is_valid(chest) {
            let chest_v = Vec3::new(chest.x, chest.y, chest.z);
            let dist = (chest_v - root_v).length();
            if dist > 1.0 {
                return Some(dist / 0.70);
            }
        }
    }

    None
}

/// Applies a zero-lag forward-backward exponential moving average (EMA) filter over a sequence of float values.
pub fn bidirectional_ema_filter(values: &[f32], alpha: f32) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let alpha = alpha.clamp(0.01, 1.0);

    // Forward pass
    let mut forward = Vec::with_capacity(values.len());
    forward.push(values[0]);
    for &val in &values[1..] {
        let prev = *forward.last().unwrap();
        forward.push(alpha * val + (1.0 - alpha) * prev);
    }

    // Backward pass
    let len = forward.len();
    let mut smoothed = vec![0.0; len];
    smoothed[len - 1] = forward[len - 1];
    for i in (0..len - 1).rev() {
        smoothed[i] = alpha * forward[i] + (1.0 - alpha) * smoothed[i + 1];
    }

    smoothed
}

/// Standardizes skeleton dimensions across animation frames so character proportions remain distance-invariant.
pub fn normalize_sequence_scale(frames: &mut [MotionFrame], target_torso_len: f32) {
    if frames.is_empty() || target_torso_len <= 0.0 {
        return;
    }

    let mut measured_lengths: Vec<Option<f32>> = Vec::with_capacity(frames.len());
    let mut valid_lengths: Vec<f32> = Vec::new();

    for frame in frames.iter() {
        let len_opt = measure_frame_torso_length(&frame.joints);
        measured_lengths.push(len_opt);
        if let Some(l) = len_opt {
            if l > 1.0 {
                valid_lengths.push(l);
            }
        }
    }

    if valid_lengths.is_empty() {
        return;
    }

    // Compute median of valid lengths
    valid_lengths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_len = valid_lengths[valid_lengths.len() / 2];

    let mut raw_scales = Vec::with_capacity(frames.len());
    let mut last_valid_scale = target_torso_len / median_len.max(1.0);

    for len_opt in measured_lengths {
        if let Some(l) = len_opt {
            if l > 1.0 {
                let s = target_torso_len / l;
                last_valid_scale = s;
                raw_scales.push(s);
                continue;
            }
        }
        raw_scales.push(last_valid_scale);
    }

    let smooth_scales = bidirectional_ema_filter(&raw_scales, 0.20);

    for (frame_idx, frame) in frames.iter_mut().enumerate() {
        let scale = smooth_scales[frame_idx];
        let root = frame.joints.get("HipsRoot").cloned();
        let (cx, cy, cz) = if let Some(r) = root {
            if joint_is_valid(&r) {
                (r.x, r.y, r.z)
            } else {
                (0.0, 0.0, 0.0)
            }
        } else {
            (0.0, 0.0, 0.0)
        };

        for joint in frame.joints.values_mut() {
            if !joint_is_valid(joint) {
                continue;
            }
            joint.x = cx + (joint.x - cx) * scale;
            joint.y = cy + (joint.y - cy) * scale;
            joint.z = cz + (joint.z - cz) * scale;
        }
    }
}

/// Offsets all frames vertically so the lowest foot contact or joint touches $Z = 0.0$.
pub fn align_to_ground_plane(frames: &mut [MotionFrame]) {
    align_to_ground_plane_on_axis(frames, "z");
}

/// Offsets all frames along a specified axis ("x", "y", or "z") so the lowest value is 0.0.
pub fn align_to_ground_plane_on_axis(frames: &mut [MotionFrame], axis: &str) {
    let mut min_val: Option<f32> = None;

    for frame in frames.iter() {
        for joint in frame.joints.values() {
            if !joint_is_valid(joint) {
                continue;
            }
            let val = match axis {
                "x" | "X" => joint.x,
                "y" | "Y" => joint.y,
                _ => joint.z,
            };
            min_val = Some(min_val.map_or(val, |m| m.min(val)));
        }
    }

    let offset = match min_val {
        Some(m) => m,
        None => return,
    };

    for frame in frames.iter_mut() {
        for joint in frame.joints.values_mut() {
            if !joint_is_valid(joint) {
                continue;
            }
            match axis {
                "x" | "X" => joint.x -= offset,
                "y" | "Y" => joint.y -= offset,
                _ => joint.z -= offset,
            }
        }
    }
}

fn joint_is_valid(joint: &JointValue) -> bool {
    joint.confidence > 0.05
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cv_to_blender_space() {
        // OpenCV (+X right, +Y down, +Z forward) -> Blender (+X right, +Y forward, +Z up)
        let cv_pt = Vec3::new(1.0, 2.0, 3.0);
        let bl_pt = cv_to_blender_space(cv_pt);
        assert_eq!(bl_pt, Vec3::new(1.0, 3.0, -2.0));
    }

    #[test]
    fn test_measure_torso_length() {
        let hips = Vec3::new(0.0, 0.0, 0.0);
        let neck = Vec3::new(0.0, 0.0, 50.0);
        assert_eq!(measure_torso_length(hips, neck), 50.0);
    }

    #[test]
    fn test_bidirectional_ema_filter_preserves_constant() {
        let vals = vec![10.0; 20];
        let smoothed = bidirectional_ema_filter(&vals, 0.2);
        assert_eq!(smoothed.len(), 20);
        for s in smoothed {
            assert!((s - 10.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_bidirectional_ema_filter_smooths_spike() {
        let mut vals = vec![10.0; 11];
        vals[5] = 20.0; // Spike in middle
        let smoothed = bidirectional_ema_filter(&vals, 0.2);
        assert!(smoothed[5] < 15.0); // Spike should be dampened
        assert!(smoothed[4] > 10.0); // Zero-phase spreading to neighbor
        assert!(smoothed[6] > 10.0);
    }

    #[test]
    fn test_align_to_ground_plane() {
        let mut joints = JointMap::new();
        joints.insert("HipsRoot".to_string(), JointValue::new(0.0, 0.0, 50.0, 0.9));
        joints.insert("LeftFoot".to_string(), JointValue::new(0.0, 0.0, 5.0, 0.9)); // lowest Z is 5.0

        let mut frames = vec![MotionFrame::new(0, joints)];
        align_to_ground_plane(&mut frames);

        assert_eq!(frames[0].joints["LeftFoot"].z, 0.0);
        assert_eq!(frames[0].joints["HipsRoot"].z, 45.0);
    }

    #[test]
    fn test_normalize_sequence_scale() {
        let mut joints = JointMap::new();
        joints.insert("HipsRoot".to_string(), JointValue::new(0.0, 0.0, 0.0, 0.9));
        joints.insert("Neck".to_string(), JointValue::new(0.0, 0.0, 25.0, 0.9)); // len = 25
        joints.insert("LeftHand".to_string(), JointValue::new(10.0, 0.0, 0.0, 0.9));

        let mut frames = vec![MotionFrame::new(0, joints)];
        // Target torso length = 50.0 (scale = 2.0)
        normalize_sequence_scale(&mut frames, 50.0);

        let neck_z = frames[0].joints["Neck"].z;
        assert!((neck_z - 50.0).abs() < 1e-3);
        let hand_x = frames[0].joints["LeftHand"].x;
        assert!((hand_x - 20.0).abs() < 1e-3);
    }
}
