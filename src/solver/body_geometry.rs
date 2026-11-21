//! Biomechanical body geometry and anatomical extrapolation.
//! Direct translation of `utils/body_geometry.py`.

use crate::skeleton::Point;
use glam::Vec2;

/// Extrapolates foot and toe base landmark points from knee-to-ankle direction and shin length.
/// Applies a forward ground-plane bias and assigns minimum joint confidence.
pub fn derive_foot_points(knee_point: Point, ankle_point: Point) -> (Point, Point) {
    let knee = Vec2::new(knee_point.x as f32, knee_point.y as f32);
    let ankle = Vec2::new(ankle_point.x as f32, ankle_point.y as f32);
    let (foot, toe) = derive_foot_vec2(knee, ankle);
    let conf = knee_point.confidence.min(ankle_point.confidence);
    (
        Point::new(foot.x.round() as i32, foot.y.round() as i32, conf),
        Point::new(toe.x.round() as i32, toe.y.round() as i32, conf),
    )
}

/// Extrapolates foot and toe base positions using raw float vectors (glam::Vec2).
pub fn derive_foot_vec2(knee: Vec2, ankle: Vec2) -> (Vec2, Vec2) {
    let d = ankle - knee;
    let mut shin_length = d.length();

    let unit = if !shin_length.is_finite() || shin_length <= 1e-6 {
        shin_length = 24.0;
        Vec2::new(0.0, 1.0)
    } else {
        d / shin_length
    };

    let foot_dir_x = unit.x * 0.85 + (if unit.x >= 0.0 { 1.0 } else { -1.0 }) * 0.15;
    let foot_dir_y = (unit.y * 0.85).max(0.20);
    let dir = Vec2::new(foot_dir_x, foot_dir_y);
    let foot_dir = if dir.length() > 1e-6 {
        dir.normalize()
    } else {
        unit
    };

    let foot_length = (shin_length * 0.35).max(12.0);
    let toe_length = (shin_length * 0.25).max(10.0);

    let foot = ankle + foot_dir * foot_length;
    let toe = foot + foot_dir * toe_length;
    (foot, toe)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_foot_points_standard() {
        let knee = Point::new(100, 100, 0.95);
        let ankle = Point::new(100, 200, 0.90); // Straight down, shin length = 100

        let (foot, toe) = derive_foot_points(knee, ankle);

        // Confidence should be min(0.95, 0.90) = 0.90
        assert_eq!(foot.confidence, 0.90);
        assert_eq!(toe.confidence, 0.90);

        // Foot extends downwards and slightly laterally
        assert!(foot.y > ankle.y);
        assert!(toe.y > foot.y);
    }

    #[test]
    fn test_derive_foot_points_degenerate() {
        // Coincident points: shin length = 0 -> fallback to length 24
        let knee = Point::new(50, 50, 0.8);
        let ankle = Point::new(50, 50, 0.7);

        let (foot, toe) = derive_foot_points(knee, ankle);
        assert_eq!(foot.confidence, 0.7);
        assert!(foot.y > ankle.y);
        assert!(toe.y > foot.y);
    }

    #[test]
    fn test_derive_foot_vec2() {
        let knee = Vec2::new(0.0, 0.0);
        let ankle = Vec2::new(0.0, 100.0);

        let (foot, toe) = derive_foot_vec2(knee, ankle);
        assert!(foot.y > ankle.y);
        assert!(toe.y > foot.y);
    }
}
