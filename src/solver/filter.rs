use glam::{Quat, Vec3};
use std::f32::consts::PI;

/// Speed-adaptive low-pass filter (1€ Filter) for 3D landmarks.
/// Eliminates high-frequency jitter during slow movements while
/// preserving responsive, low-latency tracking during fast motions.
#[derive(Debug, Clone)]
pub struct OneEuroFilter {
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    x_prev: Option<Vec3>,
    dx_prev: Vec3,
}

impl OneEuroFilter {
    /// Initializes a new OneEuroFilter.
    ///
    /// * `min_cutoff` - Minimum cutoff frequency in Hz (typically 1.0). Lower values reduce jitter during slow movements.
    /// * `beta` - Speed coefficient (typically 0.007). Higher values reduce lag during rapid movements.
    /// * `d_cutoff` - Cutoff frequency for derivative filtering in Hz (typically 1.0).
    pub fn new(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff,
            x_prev: None,
            dx_prev: Vec3::ZERO,
        }
    }

    /// Resets the internal temporal state of the filter.
    pub fn reset(&mut self) {
        self.x_prev = None;
        self.dx_prev = Vec3::ZERO;
    }

    /// Filters a 3D coordinate point given the elapsed time `dt` in seconds.
    pub fn filter_point(&mut self, point: Vec3, dt: f32) -> Vec3 {
        let dt = if dt <= 0.0 { 1.0 / 30.0 } else { dt };

        let x_prev = match self.x_prev {
            Some(prev) => prev,
            None => {
                self.x_prev = Some(point);
                return point;
            }
        };

        // Estimate derivative (velocity)
        let dx = (point - x_prev) / dt;
        let d_alpha = alpha(dt, self.d_cutoff);
        let edx = self.dx_prev.lerp(dx, d_alpha);
        self.dx_prev = edx;

        // Adapt cutoff frequency according to speed
        let speed = edx.length();
        let cutoff = self.min_cutoff + self.beta * speed;
        let x_alpha = alpha(dt, cutoff);

        let filtered = x_prev.lerp(point, x_alpha);
        self.x_prev = Some(filtered);
        filtered
    }
}

/// Computes the exponential smoothing factor alpha: alpha = 1 / (1 + tau / dt), where tau = 1 / (2 * pi * cutoff).
#[inline]
fn alpha(dt: f32, cutoff: f32) -> f32 {
    let tau = 1.0 / (2.0 * PI * cutoff.max(1e-5));
    1.0 / (1.0 + tau / dt)
}

/// Spherical linear interpolation filter for unit rotation quaternions on S^3.
#[derive(Debug, Default, Clone, Copy)]
pub struct QuaternionFilter;

impl QuaternionFilter {
    pub fn new() -> Self {
        Self
    }

    /// Performs spherical linear interpolation (Slerp) along the shortest arc on S^3.
    ///
    /// * `q0` - Starting rotation quaternion.
    /// * `q1` - Target rotation quaternion.
    /// * `alpha` - Interpolation factor in [0.0, 1.0].
    pub fn slerp(&self, q0: Quat, q1: Quat, alpha: f32) -> Quat {
        let alpha = alpha.clamp(0.0, 1.0);
        let q0 = q0.normalize();
        let mut q1 = q1.normalize();

        // Ensure shortest path on S^3 (flip q1 if dot product is negative)
        if q0.dot(q1) < 0.0 {
            q1 = -q1;
        }

        q0.slerp(q1, alpha).normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_euro_filter_initialization() {
        let mut filter = OneEuroFilter::new(1.0, 0.007, 1.0);
        let initial_point = Vec3::new(10.0, 20.0, 30.0);
        let output = filter.filter_point(initial_point, 1.0 / 30.0);
        assert_eq!(output, initial_point);
    }

    #[test]
    fn test_one_euro_filter_jitter_smoothing() {
        let mut filter = OneEuroFilter::new(1.0, 0.0, 1.0);
        let dt = 1.0 / 30.0;
        let base = Vec3::new(5.0, 5.0, 5.0);

        filter.filter_point(base, dt);

        // Add a small jitter spike
        let jittered = base + Vec3::new(0.5, -0.5, 0.2);
        let smoothed = filter.filter_point(jittered, dt);

        // Smoothed point should be dampened significantly towards base
        assert!(smoothed.distance(base) < jittered.distance(base));
    }

    #[test]
    fn test_quaternion_slerp() {
        let filter = QuaternionFilter::new();
        let q_identity = Quat::IDENTITY;
        let q_rot_y_90 = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);

        // Test alpha = 0.0 returns starting quaternion
        let at_zero = filter.slerp(q_identity, q_rot_y_90, 0.0);
        assert!((at_zero.dot(q_identity).abs() - 1.0).abs() < 1e-5);

        // Test alpha = 1.0 returns target quaternion
        let at_one = filter.slerp(q_identity, q_rot_y_90, 1.0);
        assert!((at_one.dot(q_rot_y_90).abs() - 1.0).abs() < 1e-5);

        // Test alpha = 0.5 produces 45 degree rotation
        let at_half = filter.slerp(q_identity, q_rot_y_90, 0.5);
        let expected_half = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
        assert!((at_half.dot(expected_half).abs() - 1.0).abs() < 1e-4);
    }
}
