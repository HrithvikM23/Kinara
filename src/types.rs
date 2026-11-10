use glam::{Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};

/// Standard humanoid body joint identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BodyJointId {
    Nose,
    LeftEye,
    RightEye,
    LeftEar,
    RightEar,
    LeftShoulder,
    RightShoulder,
    LeftElbow,
    RightElbow,
    LeftWrist,
    RightWrist,
    LeftHip,
    RightHip,
    LeftKnee,
    RightKnee,
    LeftAnkle,
    RightAnkle,
    LeftHeel,
    RightHeel,
    LeftToe,
    RightToe,
    Chest,
    Neck,
    Head,
    HipsRoot,
}

/// Standard hand finger identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FingerId {
    Thumb,
    Index,
    Middle,
    Ring,
    Pinky,
}

/// Which hand side is being tracked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HandSide {
    Left,
    Right,
}

/// A 3D joint with spatial position, local rotation, and detection confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Joint3D {
    pub position: Vec3,
    pub rotation: Quat,
    pub confidence: f32,
}

impl Default for Joint3D {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            confidence: 0.0,
        }
    }
}

/// Finger articulation angles (flexion, abduction, twist).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct FingerAngles {
    /// Flexion / extension angle in radians (curl inward).
    pub flexion: f32,
    /// Abduction / adduction angle in radians (lateral spread).
    pub abduction: f32,
    /// Longitudinal twist in radians (axial roll).
    pub twist: f32,
}

/// Complete articulation of an individual finger across all 3 phalanx joints.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FingerPose {
    pub id: Option<FingerId>,
    /// Metacarpophalangeal (MCP) - Main knuckle
    pub mcp: FingerAngles,
    /// Proximal Interphalangeal (PIP) - Middle knuckle
    pub pip: FingerAngles,
    /// Distal Interphalangeal (DIP) - Fingertip knuckle
    pub dip: FingerAngles,
}

/// 3D Hand pose with wrist root transform and all five articulated fingers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandPose {
    pub side: HandSide,
    pub wrist_position: Vec3,
    pub wrist_rotation: Quat,
    pub thumb: FingerPose,
    pub index: FingerPose,
    pub middle: FingerPose,
    pub ring: FingerPose,
    pub pinky: FingerPose,
    pub confidence: f32,
    /// Flag indicating whether the pose was estimated from high-res video
    /// or gracefully blended into an anatomical rest pose due to extreme distance/occlusion.
    pub is_fallback: bool,
}

impl HandPose {
    pub fn new(side: HandSide) -> Self {
        Self {
            side,
            wrist_position: Vec3::ZERO,
            wrist_rotation: Quat::IDENTITY,
            thumb: FingerPose {
                id: Some(FingerId::Thumb),
                ..Default::default()
            },
            index: FingerPose {
                id: Some(FingerId::Index),
                ..Default::default()
            },
            middle: FingerPose {
                id: Some(FingerId::Middle),
                ..Default::default()
            },
            ring: FingerPose {
                id: Some(FingerId::Ring),
                ..Default::default()
            },
            pinky: FingerPose {
                id: Some(FingerId::Pinky),
                ..Default::default()
            },
            confidence: 0.0,
            is_fallback: false,
        }
    }
}

/// Complete captured frame containing body skeleton and both hands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FramePose {
    pub frame_index: usize,
    pub timestamp_sec: f64,
    pub body_joints: Vec<(BodyJointId, Joint3D)>,
    pub left_hand: Option<HandPose>,
    pub right_hand: Option<HandPose>,
}

/// 2D bounding box in pixel coordinates.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl BoundingBox {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    #[allow(dead_code)]
    pub fn center(&self) -> Vec2 {
        Vec2::new(self.x + self.width * 0.5, self.y + self.height * 0.5)
    }
}
