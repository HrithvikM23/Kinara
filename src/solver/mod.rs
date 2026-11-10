pub mod hand_ik;
pub mod smoothing;

pub use hand_ik::BiomechanicalHandIK;
pub use smoothing::RotationSmoother;

use crate::types::FramePose;

/// Master solver coordinating biomechanical inverse kinematics and temporal smoothing.
pub struct MocapSolver {
    hand_ik: BiomechanicalHandIK,
    smoother: RotationSmoother,
    enable_ik: bool,
}

impl MocapSolver {
    pub fn new(enable_ik: bool, alpha: f32, min_confidence: f32) -> Self {
        Self {
            hand_ik: BiomechanicalHandIK::new(min_confidence),
            smoother: RotationSmoother::new(alpha),
            enable_ik,
        }
    }

    /// Solves and stabilizes the full frame pose.
    pub fn solve(&mut self, pose: &mut FramePose) {
        if let Some(ref mut left_hand) = pose.left_hand {
            if self.enable_ik {
                self.hand_ik.enforce(left_hand);
            }
            self.smoother.smooth_hand(left_hand);
        }

        if let Some(ref mut right_hand) = pose.right_hand {
            if self.enable_ik {
                self.hand_ik.enforce(right_hand);
            }
            self.smoother.smooth_hand(right_hand);
        }
    }
}
