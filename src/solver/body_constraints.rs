//! Biomechanical full-body inverse kinematics and segment length constraint solver.
//! Prevents unnatural limb stretching and shrinking across perspective changes
//! by enforcing learned invariant segment lengths. Direct translation of `utils/body_constraints.py`.

use crate::skeleton::Point;
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// COCO-17 body segment pairings grouped by anatomical limb category.
/// Indices correspond to canonical COCO-17 detector joints:
/// - 5: LeftShoulder, 7: LeftElbow, 9: LeftWrist
/// - 6: RightShoulder, 8: RightElbow, 10: RightWrist
/// - 11: LeftHip, 13: LeftKnee, 15: LeftAnkle
/// - 12: RightHip, 14: RightKnee, 16: RightAnkle
pub const BODY_SEGMENT_GROUPS: [(&str, [(usize, usize); 2]); 4] = [
    ("upper_arm", [(5, 7), (6, 8)]),
    ("forearm", [(7, 9), (8, 10)]),
    ("thigh", [(11, 13), (12, 14)]),
    ("shin", [(13, 15), (14, 16)]),
];

/// Configuration parameters for kinematics and motion solvers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolverConfig {
    pub body_constraints_enabled: bool,
    pub body_length_smoothing_alpha: f32,
    pub body_length_correction: f32,
    pub body_conf_threshold: f32,
    pub export_foot_lock_enabled: bool,
    pub export_foot_lock_velocity: f32,
    pub export_foot_lock_max_lift: f32,
    pub min_cutoff: f32,
    pub beta: f32,
    pub d_cutoff: f32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            body_constraints_enabled: true,
            body_length_smoothing_alpha: 0.15,
            body_length_correction: 0.35,
            body_conf_threshold: 0.30,
            export_foot_lock_enabled: true,
            export_foot_lock_velocity: 8.0,
            export_foot_lock_max_lift: 16.0,
            min_cutoff: 1.0,
            beta: 0.007,
            d_cutoff: 1.0,
        }
    }
}

/// Container for detected 2D body keypoints.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BodyKeypoints {
    pub points: Vec<Point>,
}

impl BodyKeypoints {
    #[inline]
    pub fn new(points: Vec<Point>) -> Self {
        Self { points }
    }
}

impl std::ops::Deref for BodyKeypoints {
    type Target = [Point];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.points
    }
}

impl std::ops::DerefMut for BodyKeypoints {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.points
    }
}

impl From<Vec<Point>> for BodyKeypoints {
    #[inline]
    fn from(points: Vec<Point>) -> Self {
        Self::new(points)
    }
}

impl From<&[Point]> for BodyKeypoints {
    #[inline]
    fn from(slice: &[Point]) -> Self {
        Self::new(slice.to_vec())
    }
}

/// Biomechanical body kinematic constraint solver.
/// Learns stable segment lengths over time using exponential moving average (EMA)
/// and projects detected joints along limb vectors to prevent stretching.
#[derive(Debug, Clone)]
pub struct BodyIKSolver {
    config: SolverConfig,
    target_lengths: HashMap<(usize, usize), f32>,
}

impl BodyIKSolver {
    /// Initializes a new BodyIKSolver with the specified configuration.
    pub fn new(config: &SolverConfig) -> Self {
        Self {
            config: config.clone(),
            target_lengths: HashMap::new(),
        }
    }

    /// Resets all learned segment lengths.
    pub fn reset(&mut self) {
        self.target_lengths.clear();
    }

    /// Returns a reference to the active target length cache.
    pub fn target_lengths(&self) -> &HashMap<(usize, usize), f32> {
        &self.target_lengths
    }

    /// Enforces anatomical limb lengths on a mutable BodyKeypoints set.
    pub fn enforce_limb_lengths(&mut self, body: &mut BodyKeypoints) {
        self.enforce_limb_lengths_points(&mut body.points);
    }

    /// Enforces anatomical limb lengths on a mutable slice of 2D `Point` landmarks.
    pub fn enforce_limb_lengths_points(&mut self, points: &mut [Point]) {
        if !self.config.body_constraints_enabled || points.len() < 17 {
            return;
        }

        self.update_target_lengths_2d(points);
        self.apply_length_targets_2d(points);
    }

    /// Enforces anatomical limb lengths on 3D joint positions using SIMD `Vec3`.
    pub fn enforce_limb_lengths_3d(&mut self, points: &mut [Vec3], confidences: &[f32]) {
        if !self.config.body_constraints_enabled || points.len() < 17 || confidences.len() < 17 {
            return;
        }

        self.update_target_lengths_3d(points, confidences);
        self.apply_length_targets_3d(points, confidences);
    }

    fn update_target_lengths_2d(&mut self, points: &[Point]) {
        let alpha = self.config.body_length_smoothing_alpha;
        let threshold = self.config.body_conf_threshold;

        for (_group, pairs) in &BODY_SEGMENT_GROUPS {
            for &(parent_idx, child_idx) in pairs {
                let parent = points[parent_idx];
                let child = points[child_idx];
                if parent.confidence <= threshold || child.confidence <= threshold {
                    continue;
                }

                let dx = (child.x - parent.x) as f32;
                let dy = (child.y - parent.y) as f32;
                let length = dx.hypot(dy);
                if length <= 2.0 {
                    continue;
                }

                let key = (parent_idx, child_idx);
                let new_len = match self.target_lengths.get(&key) {
                    Some(&prev) => prev * (1.0 - alpha) + length * alpha,
                    None => length,
                };
                self.target_lengths.insert(key, new_len);
            }
        }
    }

    fn apply_length_targets_2d(&self, points: &mut [Point]) {
        let correction = self.config.body_length_correction;
        let threshold = self.config.body_conf_threshold;

        for (_group, pairs) in &BODY_SEGMENT_GROUPS {
            for &(parent_idx, child_idx) in pairs {
                let target_len = match self.target_lengths.get(&(parent_idx, child_idx)) {
                    Some(&len) => len,
                    None => continue,
                };

                let parent = points[parent_idx];
                let child = points[child_idx];
                if parent.confidence <= threshold || child.confidence <= threshold {
                    continue;
                }

                let dx = (child.x - parent.x) as f32;
                let dy = (child.y - parent.y) as f32;
                let current_len = dx.hypot(dy);
                if current_len <= 1e-6 {
                    continue;
                }

                let corrected_len = current_len * (1.0 - correction) + target_len * correction;
                let unit_x = dx / current_len;
                let unit_y = dy / current_len;

                let new_x = (parent.x as f32 + unit_x * corrected_len).round() as i32;
                let new_y = (parent.y as f32 + unit_y * corrected_len).round() as i32;

                points[child_idx] = Point::new(new_x, new_y, child.confidence);
            }
        }
    }

    fn update_target_lengths_3d(&mut self, points: &[Vec3], confidences: &[f32]) {
        let alpha = self.config.body_length_smoothing_alpha;
        let threshold = self.config.body_conf_threshold;

        for (_group, pairs) in &BODY_SEGMENT_GROUPS {
            for &(parent_idx, child_idx) in pairs {
                if confidences[parent_idx] <= threshold || confidences[child_idx] <= threshold {
                    continue;
                }

                let length = (points[child_idx] - points[parent_idx]).length();
                if length <= 2.0 {
                    continue;
                }

                let key = (parent_idx, child_idx);
                let new_len = match self.target_lengths.get(&key) {
                    Some(&prev) => prev * (1.0 - alpha) + length * alpha,
                    None => length,
                };
                self.target_lengths.insert(key, new_len);
            }
        }
    }

    fn apply_length_targets_3d(&self, points: &mut [Vec3], confidences: &[f32]) {
        let correction = self.config.body_length_correction;
        let threshold = self.config.body_conf_threshold;

        for (_group, pairs) in &BODY_SEGMENT_GROUPS {
            for &(parent_idx, child_idx) in pairs {
                let target_len = match self.target_lengths.get(&(parent_idx, child_idx)) {
                    Some(&len) => len,
                    None => continue,
                };

                if confidences[parent_idx] <= threshold || confidences[child_idx] <= threshold {
                    continue;
                }

                let delta = points[child_idx] - points[parent_idx];
                let current_len = delta.length();
                if current_len <= 1e-6 {
                    continue;
                }

                let corrected_len = current_len * (1.0 - correction) + target_len * correction;
                let unit = delta / current_len;
                points[child_idx] = points[parent_idx] + unit * corrected_len;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_body_points() -> Vec<Point> {
        let mut pts = vec![Point::new(0, 0, 0.9); 17];
        // Upper arm left: 5 -> 7
        pts[5] = Point::new(100, 100, 0.9);
        pts[7] = Point::new(100, 150, 0.9); // len = 50
        // Forearm left: 7 -> 9
        pts[9] = Point::new(100, 190, 0.9); // len = 40
        // Upper arm right: 6 -> 8
        pts[6] = Point::new(200, 100, 0.9);
        pts[8] = Point::new(200, 150, 0.9); // len = 50
        // Forearm right: 8 -> 10
        pts[10] = Point::new(200, 190, 0.9); // len = 40
        // Thigh left: 11 -> 13
        pts[11] = Point::new(120, 250, 0.9);
        pts[13] = Point::new(120, 320, 0.9); // len = 70
        // Shin left: 13 -> 15
        pts[15] = Point::new(120, 380, 0.9); // len = 60
        // Thigh right: 12 -> 14
        pts[12] = Point::new(180, 250, 0.9);
        pts[14] = Point::new(180, 320, 0.9); // len = 70
        // Shin right: 14 -> 16
        pts[16] = Point::new(180, 380, 0.9); // len = 60
        pts
    }

    #[test]
    fn test_body_ik_solver_initial_learning() {
        let config = SolverConfig::default();
        let mut solver = BodyIKSolver::new(&config);

        let mut body = BodyKeypoints::new(dummy_body_points());
        solver.enforce_limb_lengths(&mut body);

        // Verify initial lengths cached correctly
        assert_eq!(solver.target_lengths().get(&(5, 7)), Some(&50.0));
        assert_eq!(solver.target_lengths().get(&(7, 9)), Some(&40.0));
        assert_eq!(solver.target_lengths().get(&(11, 13)), Some(&70.0));
        assert_eq!(solver.target_lengths().get(&(13, 15)), Some(&60.0));
    }

    #[test]
    fn test_body_ik_solver_prevents_stretching() {
        let mut config = SolverConfig::default();
        config.body_length_correction = 1.0; // 100% force back to target length
        let mut solver = BodyIKSolver::new(&config);

        let mut body1 = BodyKeypoints::new(dummy_body_points());
        solver.enforce_limb_lengths(&mut body1);

        // Next frame, detector glitches and left elbow (index 7) stretches to len = 100
        let mut body2 = BodyKeypoints::new(dummy_body_points());
        body2[7] = Point::new(100, 200, 0.9); // dist(5, 7) was 50, now 100
        solver.enforce_limb_lengths(&mut body2);

        // Target length after smoothing (0.15): 50 * 0.85 + 100 * 0.15 = 57.5
        // Reprojected elbow should be clamped close to 57.5 instead of remaining 100
        let dist = ((body2[7].y - body2[5].y) as f32).abs();
        assert!((dist - 57.5).abs() < 1.0, "Expected dist ~57.5, got {}", dist);
    }

    #[test]
    fn test_body_ik_solver_respects_confidence_threshold() {
        let config = SolverConfig::default();
        let mut solver = BodyIKSolver::new(&config);

        let mut body = BodyKeypoints::new(dummy_body_points());
        body[7].confidence = 0.1; // below 0.30 threshold

        solver.enforce_limb_lengths(&mut body);
        assert!(!solver.target_lengths().contains_key(&(5, 7)));
    }

    #[test]
    fn test_body_ik_solver_3d() {
        let config = SolverConfig::default();
        let mut solver = BodyIKSolver::new(&config);

        let mut pts3d = vec![Vec3::ZERO; 17];
        pts3d[5] = Vec3::new(0.0, 10.0, 0.0);
        pts3d[7] = Vec3::new(0.0, 20.0, 0.0); // len = 10
        let confs = vec![0.9; 17];

        solver.enforce_limb_lengths_3d(&mut pts3d, &confs);
        assert_eq!(solver.target_lengths().get(&(5, 7)), Some(&10.0));
    }
}
