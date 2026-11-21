//! Landmark motion prediction, confidence decay, and spatial box translation.
//! Direct translation of `utils/prediction.py`.

use crate::skeleton::{BoundingBox, Point};
use glam::Vec3;

/// Extrapolates 2D landmark positions using constant velocity between current and previous observations.
/// If previous points are absent or mismatched in length, applies confidence decay to current points.
pub fn predict_points(
    current_points: Option<&[Point]>,
    previous_points: Option<&[Point]>,
    confidence_decay: f32,
) -> Option<Vec<Point>> {
    let current = current_points?;

    let prev = match previous_points {
        Some(p) if p.len() == current.len() => p,
        _ => return Some(decay_points(current, confidence_decay)),
    };

    let predicted = current
        .iter()
        .zip(prev.iter())
        .map(|(curr, prev)| {
            let next_x = curr.x + (curr.x - prev.x);
            let next_y = curr.y + (curr.y - prev.y);
            Point::new(next_x, next_y, curr.confidence * confidence_decay)
        })
        .collect();

    Some(predicted)
}

/// Applies a multiplicative confidence decay factor to all points.
pub fn decay_points(points: &[Point], confidence_decay: f32) -> Vec<Point> {
    points
        .iter()
        .map(|p| Point::new(p.x, p.y, p.confidence * confidence_decay))
        .collect()
}

/// Translates 2D points by a fractional (offset_x, offset_y) and optionally decays confidence.
pub fn translate_points(
    points: &[Point],
    offset_x: f32,
    offset_y: f32,
    confidence_decay: f32,
) -> Vec<Point> {
    points
        .iter()
        .map(|p| {
            let tx = (p.x as f32 + offset_x).round() as i32;
            let ty = (p.y as f32 + offset_y).round() as i32;
            Point::new(tx, ty, p.confidence * confidence_decay)
        })
        .collect()
}

/// Translates a 2D bounding box by (offset_x, offset_y).
pub fn translate_box(
    box_opt: Option<BoundingBox>,
    offset_x: f32,
    offset_y: f32,
) -> Option<BoundingBox> {
    let b = box_opt?;
    let x1 = (b.x1 as f32 + offset_x).round() as i32;
    let y1 = (b.y1 as f32 + offset_y).round() as i32;
    let x2 = (b.x2 as f32 + offset_x).round() as i32;
    let y2 = (b.y2 as f32 + offset_y).round() as i32;
    Some(BoundingBox::new(x1, y1, x2, y2))
}

/// Extrapolates 3D landmarks along their velocity vectors with confidence decay (from progress.md).
pub fn predict_missing_landmarks(
    prev: &[Vec3],
    vel: &[Vec3],
    dt: f32,
    decay: f32,
) -> Vec<Vec3> {
    let decay_clamped = decay.clamp(0.0, 1.0);
    prev.iter()
        .zip(vel.iter())
        .map(|(&p, &v)| p + (v * decay_clamped) * dt)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_points_with_velocity() {
        let prev = vec![Point::new(10, 20, 1.0), Point::new(30, 40, 0.8)];
        let curr = vec![Point::new(15, 25, 0.9), Point::new(32, 44, 0.7)];

        let predicted = predict_points(Some(&curr), Some(&prev), 0.9).unwrap();
        assert_eq!(predicted.len(), 2);

        // Point 0: 15 + (15 - 10) = 20, 25 + (25 - 20) = 30, conf = 0.9 * 0.9 = 0.81
        assert_eq!(predicted[0].x, 20);
        assert_eq!(predicted[0].y, 30);
        assert!((predicted[0].confidence - 0.81).abs() < 1e-5);

        // Point 1: 32 + (32 - 30) = 34, 44 + (44 - 40) = 48, conf = 0.7 * 0.9 = 0.63
        assert_eq!(predicted[1].x, 34);
        assert_eq!(predicted[1].y, 48);
        assert!((predicted[1].confidence - 0.63).abs() < 1e-5);
    }

    #[test]
    fn test_predict_points_fallback_when_prev_missing_or_mismatched() {
        let curr = vec![Point::new(10, 20, 1.0)];
        let mismatched = vec![Point::new(10, 20, 1.0), Point::new(30, 40, 1.0)];

        // Prev is None -> falls back to decay
        let res1 = predict_points(Some(&curr), None, 0.5).unwrap();
        assert_eq!(res1[0].x, 10);
        assert_eq!(res1[0].y, 20);
        assert!((res1[0].confidence - 0.5).abs() < 1e-5);

        // Prev length mismatch -> falls back to decay
        let res2 = predict_points(Some(&curr), Some(&mismatched), 0.5).unwrap();
        assert_eq!(res2[0].x, 10);
        assert!((res2[0].confidence - 0.5).abs() < 1e-5);

        // Curr is None -> returns None
        assert!(predict_points(None, Some(&curr), 0.5).is_none());
    }

    #[test]
    fn test_decay_points() {
        let points = vec![Point::new(1, 2, 0.8), Point::new(3, 4, 0.4)];
        let decayed = decay_points(&points, 0.5);
        assert_eq!(decayed[0].x, 1);
        assert!((decayed[0].confidence - 0.4).abs() < 1e-5);
        assert!((decayed[1].confidence - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_translate_points() {
        let points = vec![Point::new(10, 20, 1.0)];
        let translated = translate_points(&points, 5.4, -3.6, 0.8);
        assert_eq!(translated[0].x, 15); // 10 + 5.4 = 15.4 -> 15
        assert_eq!(translated[0].y, 16); // 20 - 3.6 = 16.4 -> 16
        assert!((translated[0].confidence - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_translate_box() {
        let b = BoundingBox::new(10, 20, 50, 60);
        let translated = translate_box(Some(b), 2.5, -4.2).unwrap();
        assert_eq!(translated.x1, 13); // 10 + 2.5 = 12.5 -> 13
        assert_eq!(translated.y1, 16); // 20 - 4.2 = 15.8 -> 16
        assert_eq!(translated.x2, 53); // 50 + 2.5 = 52.5 -> 53
        assert_eq!(translated.y2, 56); // 60 - 4.2 = 55.8 -> 56

        assert!(translate_box(None, 1.0, 1.0).is_none());
    }

    #[test]
    fn test_predict_missing_landmarks_3d() {
        let prev = vec![Vec3::new(1.0, 2.0, 3.0)];
        let vel = vec![Vec3::new(0.5, -1.0, 2.0)];
        let dt = 0.5;
        let extrapolated = predict_missing_landmarks(&prev, &vel, dt, 1.0);
        assert_eq!(extrapolated[0], Vec3::new(1.25, 1.5, 4.0));
    }
}
