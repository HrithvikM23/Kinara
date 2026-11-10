use crate::types::{FingerAngles, FingerPose, HandPose};

/// Biomechanical Inverse Kinematics solver enforcing exact physiological joint limits.
pub struct BiomechanicalHandIK {
    min_confidence: f32,
}

impl BiomechanicalHandIK {
    pub fn new(min_confidence: f32) -> Self {
        Self { min_confidence }
    }

    /// Enforces anatomical rules across all five fingers of the hand.
    pub fn enforce(&self, hand: &mut HandPose) {
        // If detection confidence is too low or flagged as extreme distance, blend to neutral rest pose
        if hand.is_fallback || hand.confidence < self.min_confidence {
            self.apply_neutral_rest_pose(hand);
            return;
        }

        // Apply strict anatomical constraints to each digit
        self.enforce_finger(&mut hand.index, 0.05);
        self.enforce_finger(&mut hand.middle, 0.0);
        self.enforce_finger(&mut hand.ring, -0.04);
        self.enforce_finger(&mut hand.pinky, -0.08);

        // Thumb has a distinct multi-axis saddle arc
        self.enforce_thumb(&mut hand.thumb);
    }

    /// Enforces physiological limits on a finger (Index, Middle, Ring, Pinky).
    fn enforce_finger(&self, finger: &mut FingerPose, natural_rest_splay: f32) {
        // --- 1. MCP (Main Knuckle) Constraints ---
        // Flexion: 0 deg (flat open) to 90 deg (1.57 rad), max 105 deg (1.83 rad)
        // Hyperextension: max -10 deg (-0.17 rad)
        finger.mcp.flexion = finger.mcp.flexion.clamp(-0.17, 1.83);

        // Dynamic Fist Abduction Clamping:
        // As fingers curl into a fist, lateral splay drops to zero to prevent finger-interpenetration.
        let curl_ratio = (finger.mcp.flexion.max(0.0) / 1.57).clamp(0.0, 1.0);
        let max_splay = 0.26 * (1.0 - curl_ratio); // 15 deg down to 0 deg
        finger.mcp.abduction = finger.mcp.abduction.clamp(-max_splay + natural_rest_splay, max_splay + natural_rest_splay);
        finger.mcp.twist = finger.mcp.twist.clamp(-0.08, 0.08);

        // --- 2. PIP (Middle Knuckle) Constraints ---
        // STRICT HARD LIMIT: 0 deg flexion to 105 deg (1.83 rad).
        // Middle knuckles CANNOT hyperextend (never allow negative flexion).
        finger.pip.flexion = finger.pip.flexion.clamp(0.0, 1.83);
        // PIP is a pure 1-DOF hinge; lateral abduction is physically impossible.
        finger.pip.abduction = 0.0;
        finger.pip.twist = 0.0;

        // --- 3. DIP (Fingertip Knuckle) Constraints ---
        // Tendon Coupling Ratio: The FDP tendon couples DIP flexion to PIP flexion.
        // theta_DIP = 0.67 * theta_PIP (range 0.60 to 0.77).
        let coupled_dip = finger.pip.flexion * 0.67;
        finger.dip.flexion = coupled_dip.clamp(0.0, 1.22); // max 70 deg
        finger.dip.abduction = 0.0;
        finger.dip.twist = 0.0;
    }

    /// Enforces anatomical limits on the thumb complex.
    fn enforce_thumb(&self, thumb: &mut FingerPose) {
        // CMC / MCP flexion across palm
        thumb.mcp.flexion = thumb.mcp.flexion.clamp(0.0, 1.05); // max 60 deg
        thumb.mcp.abduction = thumb.mcp.abduction.clamp(0.0, 1.05); // max 60 deg away from palm

        // Middle thumb knuckle
        thumb.pip.flexion = thumb.pip.flexion.clamp(0.0, 1.40); // max 80 deg
        thumb.pip.abduction = 0.0;

        // Thumb tip (allows minor hitchhiker hyperextension)
        thumb.dip.flexion = thumb.dip.flexion.clamp(-0.26, 1.40); // -15 deg to 80 deg
        thumb.dip.abduction = 0.0;
    }

    /// Natural anatomical rest pose for extreme distance or occlusion.
    /// Eliminates snapping, broken knuckles, or jittering fingertips.
    pub fn apply_neutral_rest_pose(&self, hand: &mut HandPose) {
        let rest_angles = |mcp_flex: f32, pip_flex: f32, splay: f32| FingerPose {
            id: None,
            mcp: FingerAngles { flexion: mcp_flex, abduction: splay, twist: 0.0 },
            pip: FingerAngles { flexion: pip_flex, abduction: 0.0, twist: 0.0 },
            dip: FingerAngles { flexion: pip_flex * 0.67, abduction: 0.0, twist: 0.0 },
        };

        hand.index = rest_angles(0.35, 0.52, 0.04);
        hand.middle = rest_angles(0.38, 0.55, 0.0);
        hand.ring = rest_angles(0.42, 0.58, -0.04);
        hand.pinky = rest_angles(0.48, 0.65, -0.08);

        hand.thumb = FingerPose {
            id: None,
            mcp: FingerAngles { flexion: 0.26, abduction: 0.40, twist: 0.0 },
            pip: FingerAngles { flexion: 0.35, abduction: 0.0, twist: 0.0 },
            dip: FingerAngles { flexion: 0.20, abduction: 0.0, twist: 0.0 },
        };
        hand.is_fallback = true;
    }
}
