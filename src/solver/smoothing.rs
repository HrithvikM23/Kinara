use crate::types::{FingerAngles, FingerPose, HandPose};

/// Angular and Rotational Smoother.
/// Smooths joint rotations via SLERP instead of Cartesian coordinates,
/// ensuring rigid bone lengths and eliminating jitter without finger-shrinkage.
pub struct RotationSmoother {
    alpha: f32,
    prev_left_hand: Option<HandPose>,
    prev_right_hand: Option<HandPose>,
}

impl RotationSmoother {
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.01, 1.0),
            prev_left_hand: None,
            prev_right_hand: None,
        }
    }

    /// Smooths a hand pose against the previous temporal frame.
    pub fn smooth_hand(&mut self, hand: &mut HandPose) {
        let prev = match hand.side {
            crate::types::HandSide::Left => &mut self.prev_left_hand,
            crate::types::HandSide::Right => &mut self.prev_right_hand,
        };

        if let Some(prev_pose) = prev {
            // SLERP the wrist rotation
            hand.wrist_rotation = prev_pose.wrist_rotation.slerp(hand.wrist_rotation, self.alpha);
            hand.wrist_position = prev_pose.wrist_position.lerp(hand.wrist_position, self.alpha);

            // Angle damping for each finger joint
            Self::smooth_finger(&mut hand.index, &prev_pose.index, self.alpha);
            Self::smooth_finger(&mut hand.middle, &prev_pose.middle, self.alpha);
            Self::smooth_finger(&mut hand.ring, &prev_pose.ring, self.alpha);
            Self::smooth_finger(&mut hand.pinky, &prev_pose.pinky, self.alpha);
            Self::smooth_finger(&mut hand.thumb, &prev_pose.thumb, self.alpha);
        }

        *prev = Some(hand.clone());
    }

    fn smooth_finger(current: &mut FingerPose, prev: &FingerPose, alpha: f32) {
        Self::smooth_angles(&mut current.mcp, &prev.mcp, alpha);
        Self::smooth_angles(&mut current.pip, &prev.pip, alpha);
        Self::smooth_angles(&mut current.dip, &prev.dip, alpha);
    }

    fn smooth_angles(current: &mut FingerAngles, prev: &FingerAngles, alpha: f32) {
        current.flexion = prev.flexion + (current.flexion - prev.flexion) * alpha;
        current.abduction = prev.abduction + (current.abduction - prev.abduction) * alpha;
        current.twist = prev.twist + (current.twist - prev.twist) * alpha;
    }
}
