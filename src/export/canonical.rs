//! Canonical joint map assembly and rest-pose bind calculation.

use crate::export::coords::MotionFrame;
use crate::skeleton::{
    body_foot_name_to_index, body_name_to_index, hand_name_to_index, CoordinateSystemConfig,
    JointMap, JointValue, Point, FACE_POINT_INDICES, SKELETON,
};
use crate::solver::body_constraints::BodyKeypoints;
use crate::solver::body_geometry::derive_foot_points;
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pair of detected 21-point hand landmarks for left and right hands.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HandsPair {
    pub left: Option<Vec<Point>>,
    pub right: Option<Vec<Point>>,
}

impl HandsPair {
    pub fn new(left: Option<Vec<Point>>, right: Option<Vec<Point>>) -> Self {
        Self { left, right }
    }
}

/// Metadata envelope for exported motion files.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipMetadata {
    pub format: String,
    pub source_file: Option<String>,
    pub fps: f64,
    pub frame_count: usize,
    pub coordinate_system: CoordinateSystemConfig,
    pub skeleton: Vec<SkeletonJointMeta>,
    #[serde(flatten)]
    pub custom: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkeletonJointMeta {
    pub name: String,
    pub parent: Option<String>,
}

impl Default for ClipMetadata {
    fn default() -> Self {
        Self {
            format: "kinara-motion-json-v1".to_string(),
            source_file: None,
            fps: 30.0,
            frame_count: 0,
            coordinate_system: CoordinateSystemConfig::default(),
            skeleton: SKELETON
                .iter()
                .map(|j| SkeletonJointMeta {
                    name: j.name.to_string(),
                    parent: j.parent.map(|s| s.to_string()),
                })
                .collect(),
            custom: HashMap::new(),
        }
    }
}

/// Rest-pose bind joints and relative segment offset vectors across an armature.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RestJointOffsets {
    pub rest_joints: JointMap,
    pub rest_vectors: HashMap<String, Vec3>,
}

/// A multi-person animation frame indexing individual person tracks by label/ID.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiPersonFrame {
    pub frame_index: usize,
    pub people: HashMap<String, JointMap>,
}

impl MultiPersonFrame {
    pub fn new(frame_index: usize, people: HashMap<String, JointMap>) -> Self {
        Self {
            frame_index,
            people,
        }
    }
}

/// Assembles detected full-body and hand landmarks into the canonical 60-joint dictionary.
pub fn build_canonical_joint_map(
    body: &BodyKeypoints,
    hands: &HandsPair,
    joint_depths: Option<&HashMap<String, f32>>,
) -> JointMap {
    let mut map: JointMap = JointMap::new();
    for spec in &SKELETON {
        map.insert(spec.name.to_string(), JointValue::new(0.0, 0.0, 0.0, 0.0));
    }

    let default_depths = HashMap::new();
    let depths = joint_depths.unwrap_or(&default_depths);

    let body_pts = &body.points;
    if body_pts.len() < 17 {
        return map;
    }

    // Direct COCO-17 mapped body joints
    let body_joint_names = [
        "LeftShoulder",
        "RightShoulder",
        "LeftElbow",
        "RightElbow",
        "LeftWrist",
        "RightWrist",
        "LeftHip",
        "RightHip",
        "LeftKnee",
        "RightKnee",
        "LeftAnkle",
        "RightAnkle",
    ];

    for name in &body_joint_names {
        if let Some(idx) = body_name_to_index(name) {
            let pt = body_pts[idx];
            let z = depths.get(*name).copied().unwrap_or(0.0);
            map.insert(
                name.to_string(),
                JointValue::new(pt.x as f32, -(pt.y as f32), z, pt.confidence),
            );
        }
    }

    // Derived HipsRoot (midpoint of LeftHip and RightHip)
    let left_hip = body_pts[11];
    let right_hip = body_pts[12];
    let root_x = (left_hip.x + right_hip.x) as f32 * 0.5;
    let root_y = (left_hip.y + right_hip.y) as f32 * 0.5;
    let root_z = (depths.get("LeftHip").copied().unwrap_or(0.0)
        + depths.get("RightHip").copied().unwrap_or(0.0))
        * 0.5;
    let root_conf = (left_hip.confidence + right_hip.confidence) * 0.5;
    map.insert(
        "HipsRoot".to_string(),
        JointValue::new(root_x, -root_y, root_z, root_conf),
    );

    // Derived Chest (midpoint of LeftShoulder and RightShoulder)
    let left_shoulder = body_pts[5];
    let right_shoulder = body_pts[6];
    let chest_x = (left_shoulder.x + right_shoulder.x) as f32 * 0.5;
    let chest_y = (left_shoulder.y + right_shoulder.y) as f32 * 0.5;
    let chest_z = (depths.get("LeftShoulder").copied().unwrap_or(0.0)
        + depths.get("RightShoulder").copied().unwrap_or(0.0))
        * 0.5;
    let chest_conf = (left_shoulder.confidence + right_shoulder.confidence) * 0.5;
    map.insert(
        "Chest".to_string(),
        JointValue::new(chest_x, -chest_y, chest_z, chest_conf),
    );

    // Derived Neck and Head
    let (neck, head) = derive_head_joints(body_pts, depths);
    map.insert("Neck".to_string(), neck);
    map.insert("Head".to_string(), head);

    // Derived Feet
    let (left_foot, left_toe) = derive_foot_chain(
        body_pts,
        "Left",
        body_pts[13],
        body_pts[15],
        depths.get("LeftFoot").copied().unwrap_or(0.0),
        depths.get("LeftToeBase").copied().unwrap_or(0.0),
    );
    map.insert("LeftFoot".to_string(), left_foot);
    map.insert("LeftToeBase".to_string(), left_toe);

    let (right_foot, right_toe) = derive_foot_chain(
        body_pts,
        "Right",
        body_pts[14],
        body_pts[16],
        depths.get("RightFoot").copied().unwrap_or(0.0),
        depths.get("RightToeBase").copied().unwrap_or(0.0),
    );
    map.insert("RightFoot".to_string(), right_foot);
    map.insert("RightToeBase".to_string(), right_toe);

    // Hand Phalanges (Thumb, Index, Middle, Ring, Pinky 1-4)
    let hand_suffixes = [
        "Thumb1", "Thumb2", "Thumb3", "Thumb4", "Index1", "Index2", "Index3", "Index4",
        "Middle1", "Middle2", "Middle3", "Middle4", "Ring1", "Ring2", "Ring3", "Ring4",
        "Pinky1", "Pinky2", "Pinky3", "Pinky4",
    ];

    for (side, hand_opt) in [("Left", &hands.left), ("Right", &hands.right)] {
        if let Some(hand_pts) = hand_opt {
            if hand_pts.len() >= 21 {
                for suffix in &hand_suffixes {
                    if let Some(h_idx) = hand_name_to_index(suffix) {
                        let pt = hand_pts[h_idx];
                        let key = format!("{}{}", side, suffix);
                        let z = depths.get(&key).copied().unwrap_or(0.0);
                        map.insert(
                            key,
                            JointValue::new(pt.x as f32, -(pt.y as f32), z, pt.confidence),
                        );
                    }
                }
            }
        }
    }

    map
}

fn derive_head_joints(body_pts: &[Point], depths: &HashMap<String, f32>) -> (JointValue, JointValue) {
    let shoulder_x = (body_pts[5].x + body_pts[6].x) as f32 * 0.5;
    let shoulder_y = (body_pts[5].y + body_pts[6].y) as f32 * 0.5;
    let shoulder_conf = (body_pts[5].confidence + body_pts[6].confidence) * 0.5;

    let hip_x = (body_pts[11].x + body_pts[12].x) as f32 * 0.5;
    let hip_y = (body_pts[11].y + body_pts[12].y) as f32 * 0.5;
    let hip_conf = (body_pts[11].confidence + body_pts[12].confidence) * 0.5;

    let face_pts: Vec<Point> = FACE_POINT_INDICES
        .iter()
        .map(|&i| body_pts[i])
        .filter(|p| p.confidence > 0.0)
        .collect();

    let (neck_x, neck_y, head_x, head_y, conf) = if !face_pts.is_empty() {
        let count = face_pts.len() as f32;
        let face_x = face_pts.iter().map(|p| p.x as f32).sum::<f32>() / count;
        let face_y = face_pts.iter().map(|p| p.y as f32).sum::<f32>() / count;
        let face_conf = face_pts.iter().map(|p| p.confidence).sum::<f32>() / count;

        let nx = shoulder_x + (face_x - shoulder_x) * 0.35;
        let ny = shoulder_y + (face_y - shoulder_y) * 0.35;
        let hx = shoulder_x + (face_x - shoulder_x) * 0.85;
        let hy = shoulder_y + (face_y - shoulder_y) * 0.85;
        (nx, ny, hx, hy, (shoulder_conf + face_conf) * 0.5)
    } else {
        let dx = shoulder_x - hip_x;
        let dy = shoulder_y - hip_y;
        let mut len = dx.hypot(dy);
        let (ux, uy) = if len <= 1e-6 {
            len = 32.0;
            (0.0, -1.0)
        } else {
            (dx / len, dy / len)
        };

        let head_dist = (len * 0.35).max(18.0);
        (
            shoulder_x,
            shoulder_y,
            shoulder_x + ux * head_dist,
            shoulder_y + uy * head_dist,
            (shoulder_conf + hip_conf) * 0.5,
        )
    };

    let shoulder_depth = (depths.get("LeftShoulder").copied().unwrap_or(0.0)
        + depths.get("RightShoulder").copied().unwrap_or(0.0))
        * 0.5;

    (
        JointValue::new(neck_x, -neck_y, shoulder_depth, conf),
        JointValue::new(head_x, -head_y, shoulder_depth, conf),
    )
}

fn derive_foot_chain(
    body_pts: &[Point],
    side: &str,
    knee: Point,
    ankle: Point,
    foot_depth: f32,
    toe_depth: f32,
) -> (JointValue, JointValue) {
    let foot_name = format!("{}Foot", side);
    let toe_name = format!("{}ToeBase", side);

    if let (Some(f_idx), Some(t_idx)) = (
        body_foot_name_to_index(&foot_name),
        body_foot_name_to_index(&toe_name),
    ) {
        if body_pts.len() > t_idx && body_pts[f_idx].confidence > 0.0 && body_pts[t_idx].confidence > 0.0 {
            let fp = body_pts[f_idx];
            let tp = body_pts[t_idx];
            return (
                JointValue::new(fp.x as f32, -(fp.y as f32), foot_depth, fp.confidence),
                JointValue::new(tp.x as f32, -(tp.y as f32), toe_depth, tp.confidence),
            );
        }
    }

    let (foot_pt, toe_pt) = derive_foot_points(knee, ankle);
    (
        JointValue::new(foot_pt.x as f32, -(foot_pt.y as f32), foot_depth, foot_pt.confidence),
        JointValue::new(toe_pt.x as f32, -(toe_pt.y as f32), toe_depth, toe_pt.confidence),
    )
}

/// Computes median rest-pose bind joint positions and relative segment offset vectors across an animation clip.
pub fn compute_rest_pose_joints(frames: &[MotionFrame]) -> RestJointOffsets {
    if frames.is_empty() {
        return RestJointOffsets::default();
    }

    let mut root_xs = Vec::with_capacity(frames.len());
    let mut root_ys = Vec::with_capacity(frames.len());
    let mut root_zs = Vec::with_capacity(frames.len());
    let mut root_confs = Vec::with_capacity(frames.len());

    for frame in frames {
        if let Some(root) = frame.joints.get("HipsRoot") {
            if root.confidence > 0.05 {
                root_xs.push(root.x);
                root_ys.push(root.y);
                root_zs.push(root.z);
                root_confs.push(root.confidence);
            }
        }
    }

    let mut rest_joints = JointMap::new();
    if !root_xs.is_empty() {
        rest_joints.insert(
            "HipsRoot".to_string(),
            JointValue::new(
                median(&mut root_xs),
                median(&mut root_ys),
                median(&mut root_zs),
                median(&mut root_confs),
            ),
        );
    } else {
        rest_joints.insert("HipsRoot".to_string(), JointValue::new(0.0, 0.0, 0.0, 0.0));
    }

    let mut rest_vectors = HashMap::new();

    for spec in &SKELETON {
        let parent_name = match spec.parent {
            Some(p) => p,
            None => continue,
        };

        let mut dxs = Vec::with_capacity(frames.len());
        let mut dys = Vec::with_capacity(frames.len());
        let mut dzs = Vec::with_capacity(frames.len());
        let mut confs = Vec::with_capacity(frames.len());

        for frame in frames {
            if let (Some(pj), Some(cj)) = (frame.joints.get(parent_name), frame.joints.get(spec.name)) {
                if pj.confidence > 0.05 && cj.confidence > 0.05 {
                    dxs.push(cj.x - pj.x);
                    dys.push(cj.y - pj.y);
                    dzs.push(cj.z - pj.z);
                    confs.push((pj.confidence + cj.confidence) * 0.5);
                }
            }
        }

        let (dx, dy, dz, conf) = if !dxs.is_empty() {
            (
                median(&mut dxs),
                median(&mut dys),
                median(&mut dzs),
                median(&mut confs),
            )
        } else {
            (0.0, 0.0, 0.05, 0.0)
        };

        let vec = Vec3::new(dx, dy, dz);
        rest_vectors.insert(spec.name.to_string(), vec);

        let parent_rest = rest_joints.get(parent_name).cloned().unwrap_or_else(|| JointValue::new(0.0, 0.0, 0.0, 0.0));
        rest_joints.insert(
            spec.name.to_string(),
            JointValue::new(
                parent_rest.x + dx,
                parent_rest.y + dy,
                parent_rest.z + dz,
                conf,
            ),
        );
    }

    RestJointOffsets {
        rest_joints,
        rest_vectors,
    }
}

pub(crate) fn median(vals: &mut [f32]) -> f32 {
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals[vals.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::Point;

    fn dummy_body_keypoints() -> BodyKeypoints {
        let mut pts = vec![Point::new(0, 0, 0.9); 17];
        pts[5] = Point::new(100, 100, 0.9);
        pts[6] = Point::new(200, 100, 0.9);
        pts[7] = Point::new(90, 150, 0.9);
        pts[8] = Point::new(210, 150, 0.9);
        pts[9] = Point::new(80, 200, 0.9);
        pts[10] = Point::new(220, 200, 0.9);
        pts[11] = Point::new(120, 250, 0.9);
        pts[12] = Point::new(180, 250, 0.9);
        pts[13] = Point::new(120, 320, 0.9);
        pts[14] = Point::new(180, 320, 0.9);
        pts[15] = Point::new(120, 380, 0.9);
        pts[16] = Point::new(180, 380, 0.9);
        BodyKeypoints::new(pts)
    }

    #[test]
    fn test_build_canonical_joint_map() {
        let body = dummy_body_keypoints();
        let hands = HandsPair::default();
        let map = build_canonical_joint_map(&body, &hands, None);

        assert_eq!(map.len(), 60);
        assert!(map.contains_key("HipsRoot"));
        assert!(map.contains_key("Chest"));
        assert!(map.contains_key("Neck"));
        assert!(map.contains_key("Head"));
        assert!(map.contains_key("LeftFoot"));
        assert!(map.contains_key("RightFoot"));

        let root = map.get("HipsRoot").unwrap();
        assert_eq!(root.x, 150.0);
        assert_eq!(root.y, -250.0);
    }

    #[test]
    fn test_compute_rest_pose_joints() {
        let body = dummy_body_keypoints();
        let map = build_canonical_joint_map(&body, &HandsPair::default(), None);
        let frames = vec![MotionFrame::new(0, map)];

        let rest = compute_rest_pose_joints(&frames);
        assert!(rest.rest_joints.contains_key("HipsRoot"));
        assert!(rest.rest_vectors.contains_key("Chest"));
        assert!(rest.rest_vectors.contains_key("Neck"));
    }
}
