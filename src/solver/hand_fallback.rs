//! Hand landmark validation, wrist anchoring, and synthetic anatomical fallback generation.
//! Direct translation of `utils/hand_fallback.py`.

use crate::skeleton::Point;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

/// Normalized 2D hand landmark template offsets relative to the wrist (index 0).
/// Coordinates are (lateral, forward) relative to forearm orientation.
pub const DEFAULT_HAND_TEMPLATE: [(f32, f32); 21] = [
    (0.00, 0.00),   // 0: Wrist
    (-0.12, 0.16),  // 1: Thumb CMC
    (-0.28, 0.30),  // 2: Thumb MCP
    (-0.42, 0.48),  // 3: Thumb IP
    (-0.54, 0.68),  // 4: Thumb Tip
    (-0.20, 0.34),  // 5: Index MCP
    (-0.22, 0.58),  // 6: Index PIP
    (-0.24, 0.82),  // 7: Index DIP
    (-0.26, 1.04),  // 8: Index Tip
    (0.00, 0.40),   // 9: Middle MCP
    (0.00, 0.66),   // 10: Middle PIP
    (0.00, 0.92),   // 11: Middle DIP
    (0.00, 1.16),   // 12: Middle Tip
    (0.20, 0.38),   // 13: Ring MCP
    (0.22, 0.62),   // 14: Ring PIP
    (0.24, 0.86),   // 15: Ring DIP
    (0.26, 1.08),   // 16: Ring Tip
    (0.38, 0.28),   // 17: Pinky MCP
    (0.48, 0.46),   // 18: Pinky PIP
    (0.56, 0.64),   // 19: Pinky DIP
    (0.62, 0.82),   // 20: Pinky Tip
];

static HAND_REJECTION_COUNTS: Mutex<Option<HashMap<String, usize>>> = Mutex::new(None);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HandSide {
    Left,
    Right,
}

impl HandSide {
    pub fn from_str_case_insensitive(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "left" | "l" => Some(Self::Left),
            "right" | "r" => Some(Self::Right),
            _ => None,
        }
    }
}

/// Configuration parameters for hand detection validation and fallback generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HandValidationConfig {
    pub hand_box_min_size: f32,
    pub hand_wrist_max_offset_scale: f32,
    pub hand_kp_threshold: f32,
    pub hand_min_valid_points: usize,
    pub hand_default_scale: f32,
    pub hand_default_confidence: f32,
}

impl Default for HandValidationConfig {
    fn default() -> Self {
        Self {
            hand_box_min_size: 48.0,
            hand_wrist_max_offset_scale: 0.85,
            hand_kp_threshold: 0.35,
            hand_min_valid_points: 10,
            hand_default_scale: 0.72,
            hand_default_confidence: 0.45,
        }
    }
}

fn normalize_vec2(v: (f32, f32)) -> (f32, f32) {
    let len = v.0.hypot(v.1);
    if len <= 1e-6 {
        (0.0, -1.0)
    } else {
        (v.0 / len, v.1 / len)
    }
}

/// Translates all hand points rigidly so that the hand root (index 0) coincides with the body wrist.
pub fn anchor_hand_to_wrist(hand_points: &[Point], wrist_point: Point) -> Vec<Point> {
    if hand_points.is_empty() {
        return Vec::new();
    }

    let delta_x = wrist_point.x - hand_points[0].x;
    let delta_y = wrist_point.y - hand_points[0].y;

    hand_points
        .iter()
        .map(|p| Point::new(p.x + delta_x, p.y + delta_y, p.confidence))
        .collect()
}

/// 3D in-place anchor variant (from progress.md): translates 3D hand points so root matches body wrist.
pub fn anchor_hand_to_body_wrist(hand: &mut [Vec3], body_wrist: Vec3) {
    if hand.is_empty() {
        return;
    }
    let delta = body_wrist - hand[0];
    for p in hand.iter_mut() {
        *p += delta;
    }
}

/// Validates raw hand detection landmarks against forearm geometry and anatomical plausibility.
pub fn is_hand_detection_valid(
    hand_points: Option<&[Point]>,
    wrist_point: Point,
    elbow_point: Point,
    config: &HandValidationConfig,
    record_telemetry: bool,
) -> bool {
    let points = match hand_points {
        Some(p) if p.len() == 21 => p,
        _ => {
            if record_telemetry {
                record_hand_rejection("missing_points");
            }
            return false;
        }
    };

    let wrist_xy = (wrist_point.x as f32, wrist_point.y as f32);
    let elbow_xy = (elbow_point.x as f32, elbow_point.y as f32);
    let hand_wrist_xy = (points[0].x as f32, points[0].y as f32);

    let forearm_len = (wrist_xy.0 - elbow_xy.0)
        .hypot(wrist_xy.1 - elbow_xy.1)
        .max(1.0);

    // Check 1: Wrist offset between body detector and hand detector
    let wrist_offset = (hand_wrist_xy.0 - wrist_xy.0).hypot(hand_wrist_xy.1 - wrist_xy.1);
    let max_wrist_offset = (config.hand_box_min_size * 0.38)
        .max(forearm_len * config.hand_wrist_max_offset_scale * 1.15);
    if wrist_offset > max_wrist_offset {
        if record_telemetry {
            record_hand_rejection("wrist_offset");
        }
        return false;
    }

    // Check 2: Minimum number of confident points
    let threshold = config.hand_kp_threshold * 0.5;
    let valid_count = points.iter().filter(|p| p.confidence > threshold).count();
    if valid_count < config.hand_min_valid_points {
        if record_telemetry {
            record_hand_rejection("valid_points");
        }
        return false;
    }

    // Check 3: Radial distance bounds
    let max_radial = points[1..]
        .iter()
        .map(|p| (p.x as f32 - wrist_xy.0).hypot(p.y as f32 - wrist_xy.1))
        .fold(0.0_f32, f32::max);
    let max_allowed_radial = (forearm_len * 2.75).max(config.hand_box_min_size * 1.40);
    if max_radial > max_allowed_radial {
        if record_telemetry {
            record_hand_rejection("radial_distance");
        }
        return false;
    }

    // Check 4: Palm spread distance (MCP knuckles 5, 9, 13, 17)
    let palm_indices = [5, 9, 13, 17];
    let palm_dist_sum: f32 = palm_indices
        .iter()
        .map(|&idx| {
            let p = points[idx];
            (p.x as f32 - wrist_xy.0).hypot(p.y as f32 - wrist_xy.1)
        })
        .sum();
    let avg_palm_dist = palm_dist_sum / 4.0;
    let min_palm = forearm_len * 0.035;
    let max_palm = (forearm_len * 1.85).max(config.hand_box_min_size * 0.55);
    if avg_palm_dist < min_palm || avg_palm_dist > max_palm {
        if record_telemetry {
            record_hand_rejection("palm_distance");
        }
        return false;
    }

    true
}

/// Logs a hand rejection reason to the internal telemetry counter.
pub fn record_hand_rejection(reason: &str) {
    let mut lock = HAND_REJECTION_COUNTS.lock().unwrap();
    let counts = lock.get_or_insert_with(HashMap::new);
    *counts.entry(reason.to_string()).or_insert(0) += 1;
}

/// Extracts and clears accumulated telemetry rejection counts.
pub fn pop_hand_rejection_telemetry() -> HashMap<String, usize> {
    let mut lock = HAND_REJECTION_COUNTS.lock().unwrap();
    match lock.take() {
        Some(m) => m,
        None => HashMap::new(),
    }
}

/// Determines whether a hand detection has enough confidence to warrant attempting IK constraints.
pub fn has_usable_hand_detection(hand_points: Option<&[Point]>, config: &HandValidationConfig) -> bool {
    let points = match hand_points {
        Some(p) if p.len() == 21 => p,
        _ => return false,
    };
    let min_points = 5.max(config.hand_min_valid_points / 2);
    let thresh = config.hand_kp_threshold * 0.25;
    points.iter().filter(|p| p.confidence > thresh).count() >= min_points
}

/// Generates an anatomically proportioned synthetic 21-point rest hand aligned with the forearm.
pub fn generate_default_hand(
    wrist_point: Point,
    elbow_point: Point,
    side: HandSide,
    config: &HandValidationConfig,
) -> Vec<Point> {
    let wrist_x = wrist_point.x as f32;
    let wrist_y = wrist_point.y as f32;
    let elbow_x = elbow_point.x as f32;
    let elbow_y = elbow_point.y as f32;

    let forward_axis = normalize_vec2((wrist_x - elbow_x, wrist_y - elbow_y));
    let mut lateral_axis = (-forward_axis.1, forward_axis.0);
    if side == HandSide::Left {
        lateral_axis = (-lateral_axis.0, -lateral_axis.1);
    }

    let forearm_len = (wrist_x - elbow_x).hypot(wrist_y - elbow_y);
    let hand_scale = if forearm_len > 1e-3 {
        forearm_len * config.hand_default_scale
    } else {
        18.0
    };
    let confidence = config.hand_default_confidence.max(config.hand_kp_threshold + 0.05);

    DEFAULT_HAND_TEMPLATE
        .iter()
        .map(|&(lateral, forward)| {
            let px = wrist_x + lateral_axis.0 * lateral * hand_scale + forward_axis.0 * forward * hand_scale;
            let py = wrist_y + lateral_axis.1 * lateral * hand_scale + forward_axis.1 * forward * hand_scale;
            Point::new(px.round() as i32, py.round() as i32, confidence)
        })
        .collect()
}

/// Continuous vector version (from progress.md): generates 21-point hand landmarks from wrist and direction.
pub fn generate_fallback_hand(wrist: Vec2, dir: Vec2, len: f32) -> Vec<Vec2> {
    let forward = if dir.length() > 1e-6 { dir.normalize() } else { Vec2::new(0.0, 1.0) };
    let lateral = Vec2::new(-forward.y, forward.x);
    let scale = len.max(18.0);

    DEFAULT_HAND_TEMPLATE
        .iter()
        .map(|&(lat, fwd)| wrist + lateral * (lat * scale) + forward * (fwd * scale))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anchor_hand_to_wrist_2d() {
        let hand = vec![Point::new(100, 100, 0.9), Point::new(110, 120, 0.8)];
        let target_wrist = Point::new(200, 300, 1.0);

        let anchored = anchor_hand_to_wrist(&hand, target_wrist);
        assert_eq!(anchored[0].x, 200);
        assert_eq!(anchored[0].y, 300);
        assert_eq!(anchored[1].x, 210);
        assert_eq!(anchored[1].y, 320);
    }

    #[test]
    fn test_anchor_hand_to_body_wrist_3d() {
        let mut hand = vec![Vec3::new(1.0, 2.0, 3.0), Vec3::new(2.0, 3.0, 4.0)];
        let target_wrist = Vec3::new(10.0, 20.0, 30.0);

        anchor_hand_to_body_wrist(&mut hand, target_wrist);
        assert_eq!(hand[0], target_wrist);
        assert_eq!(hand[1], Vec3::new(11.0, 21.0, 31.0));
    }

    #[test]
    fn test_generate_default_hand_length_and_structure() {
        let wrist = Point::new(100, 200, 1.0);
        let elbow = Point::new(100, 100, 1.0);
        let config = HandValidationConfig::default();

        let hand = generate_default_hand(wrist, elbow, HandSide::Right, &config);
        assert_eq!(hand.len(), 21);
        assert_eq!(hand[0].x, 100);
        assert_eq!(hand[0].y, 200);
        // Fingertip points should extend forward (downwards in image space y > 200)
        assert!(hand[12].y > hand[0].y);
    }

    #[test]
    fn test_generate_fallback_hand_vec2() {
        let wrist = Vec2::new(100.0, 100.0);
        let dir = Vec2::new(0.0, 1.0);
        let hand = generate_fallback_hand(wrist, dir, 50.0);
        assert_eq!(hand.len(), 21);
        assert_eq!(hand[0], wrist);
    }

    #[test]
    fn test_telemetry_recording() {
        pop_hand_rejection_telemetry(); // Clear
        record_hand_rejection("test_reason");
        record_hand_rejection("test_reason");
        record_hand_rejection("another_reason");

        let stats = pop_hand_rejection_telemetry();
        assert_eq!(stats.get("test_reason"), Some(&2));
        assert_eq!(stats.get("another_reason"), Some(&1));

        // After pop, should be empty
        let empty = pop_hand_rejection_telemetry();
        assert!(empty.is_empty());
    }
}
