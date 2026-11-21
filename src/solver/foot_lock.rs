//! Foot contact detection and planted foot stabilization.
//! Prevents foot sliding, skating, and ground penetration artifacts by clamping
//! ground-contact joint velocities to zero during contact phases.
//! Direct translation of `utils/motion_cleanup.py:_lock_planted_feet`.

use glam::Vec3;
use serde::{Deserialize, Serialize};

/// 3D coordinates and contact states for the actor's feet.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FootContacts {
    pub left_ankle: Vec3,
    pub right_ankle: Vec3,
    pub left_toe: Vec3,
    pub right_toe: Vec3,
    pub left_planted: bool,
    pub right_planted: bool,
}

impl FootContacts {
    pub fn new(left_ankle: Vec3, right_ankle: Vec3, left_toe: Vec3, right_toe: Vec3) -> Self {
        Self {
            left_ankle,
            right_ankle,
            left_toe,
            right_toe,
            left_planted: false,
            right_planted: false,
        }
    }
}

/// Dynamic stabilizer that detects when feet touch the floor plane
/// and clamps horizontal $(x, y)$ coordinates to eliminate skating.
#[derive(Debug, Clone)]
pub struct FootLockStabilizer {
    pub velocity_threshold: f32,
    pub max_lift: f32,
    pub ground_z: Option<f32>,
    prev_left_ankle: Option<Vec3>,
    prev_right_ankle: Option<Vec3>,
    locked_left_ankle: Option<Vec3>,
    locked_right_ankle: Option<Vec3>,
    locked_left_toe: Option<Vec3>,
    locked_right_toe: Option<Vec3>,
}

impl FootLockStabilizer {
    /// Creates a new FootLockStabilizer.
    ///
    /// * `velocity_threshold` - Maximum velocity in units/sec below which a foot is considered planted (typically ~8.0).
    /// * `max_lift` - Maximum altitude above ground $Z=0$ to engage locking (typically ~16.0).
    pub fn new(velocity_threshold: f32, max_lift: f32) -> Self {
        Self {
            velocity_threshold,
            max_lift,
            ground_z: Some(0.0),
            prev_left_ankle: None,
            prev_right_ankle: None,
            locked_left_ankle: None,
            locked_right_ankle: None,
            locked_left_toe: None,
            locked_right_toe: None,
        }
    }

    /// Sets explicit ground height altitude.
    pub fn with_ground_height(mut self, ground_z: f32) -> Self {
        self.ground_z = Some(ground_z);
        self
    }

    /// Resets historical state.
    pub fn reset(&mut self) {
        self.prev_left_ankle = None;
        self.prev_right_ankle = None;
        self.locked_left_ankle = None;
        self.locked_right_ankle = None;
        self.locked_left_toe = None;
        self.locked_right_toe = None;
    }

    /// Detects foot floor impact and locks $(x, y)$ horizontal coordinates of planted feet.
    pub fn lock_planted_feet(&mut self, feet: &mut FootContacts, dt: f32) {
        let dt = if dt <= 1e-6 { 1.0 / 30.0 } else { dt };
        let ground_z = self.ground_z.unwrap_or(0.0);

        // --- Left Foot Processing ---
        let left_near_ground = (feet.left_ankle.z - ground_z).abs() <= self.max_lift
            || (feet.left_toe.z - ground_z).abs() <= self.max_lift;

        let left_speed = if let Some(prev) = self.prev_left_ankle {
            (feet.left_ankle - prev).length() / dt
        } else {
            0.0
        };

        if left_near_ground && (left_speed <= self.velocity_threshold || self.locked_left_ankle.is_some()) {
            if let (Some(locked_ankle), Some(locked_toe)) = (self.locked_left_ankle, self.locked_left_toe) {
                // Keep horizontal (x, y) fixed to the anchor while preserving vertical z
                feet.left_ankle.x = locked_ankle.x;
                feet.left_ankle.y = locked_ankle.y;
                feet.left_toe.x = locked_toe.x;
                feet.left_toe.y = locked_toe.y;
            } else {
                self.locked_left_ankle = Some(feet.left_ankle);
                self.locked_left_toe = Some(feet.left_toe);
            }
            feet.left_planted = true;
        } else {
            self.locked_left_ankle = None;
            self.locked_left_toe = None;
            feet.left_planted = false;
        }
        self.prev_left_ankle = Some(feet.left_ankle);

        // --- Right Foot Processing ---
        let right_near_ground = (feet.right_ankle.z - ground_z).abs() <= self.max_lift
            || (feet.right_toe.z - ground_z).abs() <= self.max_lift;

        let right_speed = if let Some(prev) = self.prev_right_ankle {
            (feet.right_ankle - prev).length() / dt
        } else {
            0.0
        };

        if right_near_ground && (right_speed <= self.velocity_threshold || self.locked_right_ankle.is_some()) {
            if let (Some(locked_ankle), Some(locked_toe)) = (self.locked_right_ankle, self.locked_right_toe) {
                feet.right_ankle.x = locked_ankle.x;
                feet.right_ankle.y = locked_ankle.y;
                feet.right_toe.x = locked_toe.x;
                feet.right_toe.y = locked_toe.y;
            } else {
                self.locked_right_ankle = Some(feet.right_ankle);
                self.locked_right_toe = Some(feet.right_toe);
            }
            feet.right_planted = true;
        } else {
            self.locked_right_ankle = None;
            self.locked_right_toe = None;
            feet.right_planted = false;
        }
        self.prev_right_ankle = Some(feet.right_ankle);
    }

    /// Evaluates downward velocity and vertical altitude to stabilize feet positions.
    /// Returns stabilized `(left_ankle, right_ankle, left_toe, right_toe)`.
    pub fn update(
        &mut self,
        ankles: (Vec3, Vec3),
        toes: (Vec3, Vec3),
        dt: f32,
    ) -> (Vec3, Vec3, Vec3, Vec3) {
        let mut contacts = FootContacts::new(ankles.0, ankles.1, toes.0, toes.1);
        self.lock_planted_feet(&mut contacts, dt);
        (
            contacts.left_ankle,
            contacts.right_ankle,
            contacts.left_toe,
            contacts.right_toe,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foot_lock_engages_near_ground() {
        let mut stabilizer = FootLockStabilizer::new(10.0, 5.0).with_ground_height(0.0);

        let dt = 1.0 / 30.0;
        let mut feet1 = FootContacts::new(
            Vec3::new(10.0, 20.0, 1.0),
            Vec3::new(30.0, 20.0, 20.0), // high up
            Vec3::new(10.0, 25.0, 0.5),
            Vec3::new(30.0, 25.0, 19.5),
        );

        stabilizer.lock_planted_feet(&mut feet1, dt);
        assert!(feet1.left_planted);
        assert!(!feet1.right_planted);

        // Next frame, slight horizontal drift on left foot while planted
        let mut feet2 = FootContacts::new(
            Vec3::new(10.5, 20.2, 1.0),
            Vec3::new(30.0, 20.0, 20.0),
            Vec3::new(10.5, 25.2, 0.5),
            Vec3::new(30.0, 25.0, 19.5),
        );

        stabilizer.lock_planted_feet(&mut feet2, dt);
        assert!(feet2.left_planted);
        // Position should be clamped to original anchor (10.0, 20.0)
        assert_eq!(feet2.left_ankle.x, 10.0);
        assert_eq!(feet2.left_ankle.y, 20.0);
        assert_eq!(feet2.left_toe.x, 10.0);
        assert_eq!(feet2.left_toe.y, 25.0);
    }

    #[test]
    fn test_foot_lock_releases_on_lift() {
        let mut stabilizer = FootLockStabilizer::new(10.0, 5.0).with_ground_height(0.0);
        let dt = 1.0 / 30.0;

        let mut feet1 = FootContacts::new(
            Vec3::new(10.0, 20.0, 1.0),
            Vec3::new(30.0, 20.0, 1.0),
            Vec3::new(10.0, 25.0, 0.5),
            Vec3::new(30.0, 25.0, 0.5),
        );
        stabilizer.lock_planted_feet(&mut feet1, dt);
        assert!(feet1.left_planted);

        // Lift foot up above max_lift (5.0)
        let mut feet2 = FootContacts::new(
            Vec3::new(15.0, 25.0, 15.0),
            Vec3::new(30.0, 20.0, 1.0),
            Vec3::new(15.0, 30.0, 14.5),
            Vec3::new(30.0, 25.0, 0.5),
        );
        stabilizer.lock_planted_feet(&mut feet2, dt);
        assert!(!feet2.left_planted);
        // Position is NOT clamped
        assert_eq!(feet2.left_ankle.x, 15.0);
    }
}
