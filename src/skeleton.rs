//! Canonical skeletal hierarchy, landmark points, and coordinate definitions for Kinara.
//! Direct translation of `utils/skeleton.py`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// FBX official ticks per second time unit constant.
pub const FBX_TIME_UNIT: i64 = 46_186_158_000;

/// Joint specification defining joint name and its parent in the armature hierarchy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JointSpec {
    pub name: &'static str,
    pub parent: Option<&'static str>,
}

/// 2D landmark point with confidence: (x, y, confidence).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: i32,
    pub y: i32,
    pub confidence: f32,
}

impl Point {
    #[inline]
    pub fn new(x: i32, y: i32, confidence: f32) -> Self {
        Self { x, y, confidence }
    }
}

impl From<(i32, i32, f32)> for Point {
    #[inline]
    fn from(t: (i32, i32, f32)) -> Self {
        Self::new(t.0, t.1, t.2)
    }
}

impl From<Point> for (i32, i32, f32) {
    #[inline]
    fn from(p: Point) -> Self {
        (p.x, p.y, p.confidence)
    }
}

/// 2D bounding box represented as (x1, y1, x2, y2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

impl BoundingBox {
    #[inline]
    pub fn new(x1: i32, y1: i32, x2: i32, y2: i32) -> Self {
        Self { x1, y1, x2, y2 }
    }
}

impl From<(i32, i32, i32, i32)> for BoundingBox {
    #[inline]
    fn from(b: (i32, i32, i32, i32)) -> Self {
        Self::new(b.0, b.1, b.2, b.3)
    }
}

impl From<BoundingBox> for (i32, i32, i32, i32) {
    #[inline]
    fn from(b: BoundingBox) -> Self {
        (b.x1, b.y1, b.x2, b.y2)
    }
}

/// 3D joint Cartesian coordinates with tracking confidence.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct JointValue {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub confidence: f32,
}

impl JointValue {
    #[inline]
    pub fn new(x: f32, y: f32, z: f32, confidence: f32) -> Self {
        Self { x, y, z, confidence }
    }
}

/// Dictionary mapping joint names to their 3D coordinates.
pub type JointMap = HashMap<String, JointValue>;

/// Indices of facial landmark points in the COCO 17-keypoint model (Nose, LeftEye, RightEye, LeftEar, RightEar).
pub const FACE_POINT_INDICES: [usize; 5] = [0, 1, 2, 3, 4];

/// Canonical 60-joint skeletal hierarchy.
pub const SKELETON: [JointSpec; 60] = [
    JointSpec { name: "HipsRoot", parent: None },
    JointSpec { name: "LeftHip", parent: Some("HipsRoot") },
    JointSpec { name: "LeftKnee", parent: Some("LeftHip") },
    JointSpec { name: "LeftAnkle", parent: Some("LeftKnee") },
    JointSpec { name: "LeftFoot", parent: Some("LeftAnkle") },
    JointSpec { name: "LeftToeBase", parent: Some("LeftFoot") },
    JointSpec { name: "RightHip", parent: Some("HipsRoot") },
    JointSpec { name: "RightKnee", parent: Some("RightHip") },
    JointSpec { name: "RightAnkle", parent: Some("RightKnee") },
    JointSpec { name: "RightFoot", parent: Some("RightAnkle") },
    JointSpec { name: "RightToeBase", parent: Some("RightFoot") },
    JointSpec { name: "Chest", parent: Some("HipsRoot") },
    JointSpec { name: "Neck", parent: Some("Chest") },
    JointSpec { name: "Head", parent: Some("Neck") },
    JointSpec { name: "LeftShoulder", parent: Some("Chest") },
    JointSpec { name: "LeftElbow", parent: Some("LeftShoulder") },
    JointSpec { name: "LeftWrist", parent: Some("LeftElbow") },
    JointSpec { name: "LeftThumb1", parent: Some("LeftWrist") },
    JointSpec { name: "LeftThumb2", parent: Some("LeftThumb1") },
    JointSpec { name: "LeftThumb3", parent: Some("LeftThumb2") },
    JointSpec { name: "LeftThumb4", parent: Some("LeftThumb3") },
    JointSpec { name: "LeftIndex1", parent: Some("LeftWrist") },
    JointSpec { name: "LeftIndex2", parent: Some("LeftIndex1") },
    JointSpec { name: "LeftIndex3", parent: Some("LeftIndex2") },
    JointSpec { name: "LeftIndex4", parent: Some("LeftIndex3") },
    JointSpec { name: "LeftMiddle1", parent: Some("LeftWrist") },
    JointSpec { name: "LeftMiddle2", parent: Some("LeftMiddle1") },
    JointSpec { name: "LeftMiddle3", parent: Some("LeftMiddle2") },
    JointSpec { name: "LeftMiddle4", parent: Some("LeftMiddle3") },
    JointSpec { name: "LeftRing1", parent: Some("LeftWrist") },
    JointSpec { name: "LeftRing2", parent: Some("LeftRing1") },
    JointSpec { name: "LeftRing3", parent: Some("LeftRing2") },
    JointSpec { name: "LeftRing4", parent: Some("LeftRing3") },
    JointSpec { name: "LeftPinky1", parent: Some("LeftWrist") },
    JointSpec { name: "LeftPinky2", parent: Some("LeftPinky1") },
    JointSpec { name: "LeftPinky3", parent: Some("LeftPinky2") },
    JointSpec { name: "LeftPinky4", parent: Some("LeftPinky3") },
    JointSpec { name: "RightShoulder", parent: Some("Chest") },
    JointSpec { name: "RightElbow", parent: Some("RightShoulder") },
    JointSpec { name: "RightWrist", parent: Some("RightElbow") },
    JointSpec { name: "RightThumb1", parent: Some("RightWrist") },
    JointSpec { name: "RightThumb2", parent: Some("RightThumb1") },
    JointSpec { name: "RightThumb3", parent: Some("RightThumb2") },
    JointSpec { name: "RightThumb4", parent: Some("RightThumb3") },
    JointSpec { name: "RightIndex1", parent: Some("RightWrist") },
    JointSpec { name: "RightIndex2", parent: Some("RightIndex1") },
    JointSpec { name: "RightIndex3", parent: Some("RightIndex2") },
    JointSpec { name: "RightIndex4", parent: Some("RightIndex3") },
    JointSpec { name: "RightMiddle1", parent: Some("RightWrist") },
    JointSpec { name: "RightMiddle2", parent: Some("RightMiddle1") },
    JointSpec { name: "RightMiddle3", parent: Some("RightMiddle2") },
    JointSpec { name: "RightMiddle4", parent: Some("RightMiddle3") },
    JointSpec { name: "RightRing1", parent: Some("RightWrist") },
    JointSpec { name: "RightRing2", parent: Some("RightRing1") },
    JointSpec { name: "RightRing3", parent: Some("RightRing2") },
    JointSpec { name: "RightRing4", parent: Some("RightRing3") },
    JointSpec { name: "RightPinky1", parent: Some("RightWrist") },
    JointSpec { name: "RightPinky2", parent: Some("RightPinky1") },
    JointSpec { name: "RightPinky3", parent: Some("RightPinky2") },
    JointSpec { name: "RightPinky4", parent: Some("RightPinky3") },
];

/// Maps COCO-17 body joint name to its detector array index.
pub fn body_name_to_index(name: &str) -> Option<usize> {
    match name {
        "LeftShoulder" => Some(5),
        "RightShoulder" => Some(6),
        "LeftElbow" => Some(7),
        "RightElbow" => Some(8),
        "LeftWrist" => Some(9),
        "RightWrist" => Some(10),
        "LeftHip" => Some(11),
        "RightHip" => Some(12),
        "LeftKnee" => Some(13),
        "RightKnee" => Some(14),
        "LeftAnkle" => Some(15),
        "RightAnkle" => Some(16),
        _ => None,
    }
}

/// Maps derived foot joint name to its array index.
pub fn body_foot_name_to_index(name: &str) -> Option<usize> {
    match name {
        "LeftFoot" => Some(17),
        "RightFoot" => Some(18),
        "LeftToeBase" => Some(19),
        "RightToeBase" => Some(20),
        _ => None,
    }
}

/// Maps 21-point hand landmark name to its array index (1-based, wrist is 0).
pub fn hand_name_to_index(name: &str) -> Option<usize> {
    match name {
        "Thumb1" => Some(1),
        "Thumb2" => Some(2),
        "Thumb3" => Some(3),
        "Thumb4" => Some(4),
        "Index1" => Some(5),
        "Index2" => Some(6),
        "Index3" => Some(7),
        "Index4" => Some(8),
        "Middle1" => Some(9),
        "Middle2" => Some(10),
        "Middle3" => Some(11),
        "Middle4" => Some(12),
        "Ring1" => Some(13),
        "Ring2" => Some(14),
        "Ring3" => Some(15),
        "Ring4" => Some(16),
        "Pinky1" => Some(17),
        "Pinky2" => Some(18),
        "Pinky3" => Some(19),
        "Pinky4" => Some(20),
        _ => None,
    }
}

/// Standard export coordinate system metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoordinateSystemConfig {
    pub space: String,
    pub right_axis: String,
    pub forward_axis: String,
    pub up_axis: String,
    pub grounded_axis: String,
}

impl Default for CoordinateSystemConfig {
    fn default() -> Self {
        Self {
            space: "kinara_normalized".to_string(),
            right_axis: "X".to_string(),
            forward_axis: "Y".to_string(),
            up_axis: "Z".to_string(),
            grounded_axis: "Z".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skeleton_hierarchy_integrity() {
        assert_eq!(SKELETON.len(), 60);
        assert_eq!(SKELETON[0].name, "HipsRoot");
        assert!(SKELETON[0].parent.is_none());

        // Verify every child references an existing parent in SKELETON
        let joint_names: Vec<&str> = SKELETON.iter().map(|j| j.name).collect();
        for spec in &SKELETON[1..] {
            let parent_name = spec.parent.expect("Child joint must have a parent");
            assert!(
                joint_names.contains(&parent_name),
                "Parent '{}' of joint '{}' not found in SKELETON",
                parent_name,
                spec.name
            );
        }
    }

    #[test]
    fn test_body_name_mappings() {
        assert_eq!(body_name_to_index("LeftWrist"), Some(9));
        assert_eq!(body_name_to_index("RightAnkle"), Some(16));
        assert_eq!(body_name_to_index("NonExistent"), None);

        assert_eq!(body_foot_name_to_index("LeftFoot"), Some(17));
        assert_eq!(body_foot_name_to_index("RightToeBase"), Some(20));

        assert_eq!(hand_name_to_index("Index1"), Some(5));
        assert_eq!(hand_name_to_index("Pinky4"), Some(20));
    }

    #[test]
    fn test_coordinate_system_default() {
        let config = CoordinateSystemConfig::default();
        assert_eq!(config.space, "kinara_normalized");
        assert_eq!(config.up_axis, "Z");
    }
}
