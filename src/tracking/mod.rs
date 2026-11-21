//! Multi-person tracking, re-identification, and hand ownership disambiguation.
//! Direct translation of `utils/multi_person.py`.

use crate::skeleton::{BoundingBox, Point};
use crate::solver::hand_fallback::generate_fallback_hand;
use crate::export::HandsPair;
use glam::Vec2;
use serde::{Deserialize, Serialize};

/// 16-bin normalized HSV color histogram representing actor clothing signature.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ColorHistogram {
    pub bins: [f32; 16],
}

impl Default for ColorHistogram {
    fn default() -> Self {
        Self { bins: [0.0; 16] }
    }
}

impl ColorHistogram {
    pub fn new(bins: [f32; 16]) -> Self {
        Self { bins }
    }

    /// Blends current histogram with another using exponential smoothing.
    pub fn blend(&self, other: &ColorHistogram, alpha: f32) -> ColorHistogram {
        let alpha = alpha.clamp(0.0, 1.0);
        let mut new_bins = [0.0; 16];
        for i in 0..16 {
            new_bins[i] = self.bins[i] * (1.0 - alpha) + other.bins[i] * alpha;
        }
        ColorHistogram { bins: new_bins }
    }
}

/// Calculates intersection-over-union (IoU) metric between two 2D bounding boxes.
pub fn compute_bbox_iou(a: BoundingBox, b: BoundingBox) -> f32 {
    let inter_x1 = a.x1.max(b.x1);
    let inter_y1 = a.y1.max(b.y1);
    let inter_x2 = a.x2.min(b.x2);
    let inter_y2 = a.y2.min(b.y2);

    if inter_x2 <= inter_x1 || inter_y2 <= inter_y1 {
        return 0.0;
    }

    let inter_area = (inter_x2 - inter_x1) as i64 * (inter_y2 - inter_y1) as i64;
    let area_a = (a.x2 - a.x1).max(0) as i64 * (a.y2 - a.y1).max(0) as i64;
    let area_b = (b.x2 - b.x1).max(0) as i64 * (b.y2 - b.y1).max(0) as i64;
    let union_area = area_a + area_b - inter_area;

    if union_area <= 0 {
        return 0.0;
    }

    inter_area as f32 / union_area as f32
}

/// Extracts a 16-bin normalized Hue histogram from the torso region of an image
/// for clothing-based actor re-identification.
pub fn extract_torso_histogram(frame: &image::RgbImage, torso: BoundingBox) -> ColorHistogram {
    let (width, height) = frame.dimensions();
        let x1 = torso.x1.clamp(0, width as i32) as u32;
        let y1 = torso.y1.clamp(0, height as i32) as u32;
        let x2 = torso.x2.clamp(0, width as i32) as u32;
        let y2 = torso.y2.clamp(0, height as i32) as u32;

        if x2 <= x1 || y2 <= y1 {
            return ColorHistogram::default();
        }

        // Focus on inner 60% horizontally and upper 50% vertically of torso
        let w = x2 - x1;
        let h = y2 - y1;
        let crop_x1 = x1 + (w as f32 * 0.20) as u32;
        let crop_x2 = x2.saturating_sub((w as f32 * 0.20) as u32).max(crop_x1 + 1);
        let crop_y1 = y1;
        let crop_y2 = y1 + (h as f32 * 0.60) as u32;

        let mut bins = [0.0f32; 16];
        let mut count = 0.0f32;

        for y in crop_y1..crop_y2.min(height) {
            for x in crop_x1..crop_x2.min(width) {
                let pixel = frame.get_pixel(x, y);
                let r = pixel[0] as f32 / 255.0;
                let g = pixel[1] as f32 / 255.0;
                let b = pixel[2] as f32 / 255.0;

                let max = r.max(g).max(b);
                let min = r.min(g).min(b);
                let delta = max - min;

                // Ignore near-grayscale pixels (low saturation)
                if delta > 0.08 {
                    let hue = if max == r {
                        60.0 * (((g - b) / delta).rem_euclid(6.0))
                    } else if max == g {
                        60.0 * (((b - r) / delta) + 2.0)
                    } else {
                        60.0 * (((r - g) / delta) + 4.0)
                    };

                    let bin_idx = ((hue / 360.0) * 16.0).floor().clamp(0.0, 15.0) as usize;
                    bins[bin_idx] += 1.0;
                    count += 1.0;
                }
            }
        }

        if count > 0.0 {
            for b in &mut bins {
                *b /= count;
            }
        }

        ColorHistogram { bins }
}

/// Measures color profile similarity via histogram intersection score in range [0.0, 1.0].
pub fn compare_color_histograms(a: &ColorHistogram, b: &ColorHistogram) -> f32 {
    let mut intersection = 0.0f32;
    for i in 0..16 {
        intersection += a.bins[i].min(b.bins[i]);
    }
    intersection.clamp(0.0, 1.0)
}

/// Raw detection for a detected person within a single frame.
#[derive(Debug, Clone, PartialEq)]
pub struct BodyDetection {
    pub box_: BoundingBox,
    pub body_points: Vec<Point>,
    pub score: f32,
    pub color_hist: Option<ColorHistogram>,
    pub tracker_id: Option<u64>,
}

/// Persistently tracked person state across video frames.
#[derive(Debug, Clone, PartialEq)]
pub struct PersonTrack {
    pub id: u64,
    pub tracker_id: Option<u64>,
    pub box_: BoundingBox,
    pub body_points: Vec<Point>,
    pub velocity: (f32, f32),
    pub color_signature: ColorHistogram,
    pub missed_frames: usize,
    pub hit_streak: usize,
    pub label: Option<String>,
    pub hands: HandsPair,
}

impl PersonTrack {
    pub fn center(&self) -> (f32, f32) {
        let cx = (self.box_.x1 + self.box_.x2) as f32 * 0.5;
        let cy = (self.box_.y1 + self.box_.y2) as f32 * 0.5;
        (cx, cy)
    }

    pub fn predicted_box(&self) -> BoundingBox {
        let (vx, vy) = self.velocity;
        BoundingBox::new(
            (self.box_.x1 as f32 + vx).round() as i32,
            (self.box_.y1 as f32 + vy).round() as i32,
            (self.box_.x2 as f32 + vx).round() as i32,
            (self.box_.y2 as f32 + vy).round() as i32,
        )
    }
}

/// Configuration parameters for multi-person tracking and association.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackerConfig {
    pub max_people: usize,
    pub track_hold_frames: usize,
    pub match_threshold: f32,
    pub cross_wrist_ratio: f32,
    pub color_blend_alpha: f32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            max_people: 4,
            track_hold_frames: 10,
            match_threshold: 0.15,
            cross_wrist_ratio: 0.90,
            color_blend_alpha: 0.35,
        }
    }
}

/// Evaluates association score between a detection and an existing track.
pub fn score_detection_to_track(det: &BodyDetection, track: &PersonTrack) -> f32 {
    if let (Some(det_id), Some(track_id)) = (det.tracker_id, track.tracker_id) {
        if det_id == track_id {
            return 10.0;
        }
    }

    let pred_box = track.predicted_box();
    let iou = compute_bbox_iou(pred_box, det.box_);

    // Center distance score
    let ax = (pred_box.x1 + pred_box.x2) as f32 * 0.5;
    let ay = (pred_box.y1 + pred_box.y2) as f32 * 0.5;
    let bx = (det.box_.x1 + det.box_.x2) as f32 * 0.5;
    let by = (det.box_.y1 + det.box_.y2) as f32 * 0.5;
    let dist = (ax - bx).hypot(ay - by);
    let scale = (pred_box.x2 - pred_box.x1).max(pred_box.y2 - pred_box.y1) as f32;
    let center_score = (1.0 - dist / (scale * 2.0).max(1.0)).max(0.0);

    // Size similarity score
    let area_a = ((pred_box.x2 - pred_box.x1).max(1) * (pred_box.y2 - pred_box.y1).max(1)) as f32;
    let area_b = ((det.box_.x2 - det.box_.x1).max(1) * (det.box_.y2 - det.box_.y1).max(1)) as f32;
    let size_score = area_a.min(area_b) / area_a.max(area_b);

    // Color histogram similarity
    let color_score = if let Some(ref hist) = det.color_hist {
        compare_color_histograms(&track.color_signature, hist)
    } else {
        0.5
    };

    iou * 0.40 + center_score * 0.20 + size_score * 0.10 + color_score * 0.30
}

/// Disambiguates hand ownership during actor crossovers, reassigning stolen hands.
pub fn resolve_cross_person_hand_ownership(tracks: &mut [PersonTrack]) {
    let len = tracks.len();
    if len <= 1 {
        return;
    }

    for track_idx in 0..len {
        for (side, wrist_idx, elbow_idx) in [("left", 9, 7), ("right", 10, 8)] {
            if tracks[track_idx].body_points.len() <= wrist_idx {
                continue;
            }

            let owner_wrist = tracks[track_idx].body_points[wrist_idx];
            let owner_elbow = tracks[track_idx].body_points[elbow_idx];

            if owner_wrist.confidence <= 0.2 || owner_elbow.confidence <= 0.2 {
                continue;
            }

            let hand_opt = match side {
                "left" => tracks[track_idx].hands.left.clone(),
                _ => tracks[track_idx].hands.right.clone(),
            };

            if let Some(ref hand_pts) = hand_opt {
                if !hand_pts.is_empty() {
                    let hand_wrist = hand_pts[0];
                    let owner_dist = ((hand_wrist.x - owner_wrist.x) as f32)
                        .hypot((hand_wrist.y - owner_wrist.y) as f32);

                    let mut best_other_score = -1.0f32;
                    for other_idx in 0..len {
                        if other_idx == track_idx || tracks[other_idx].body_points.len() <= wrist_idx {
                            continue;
                        }
                        let other_wrist = tracks[other_idx].body_points[wrist_idx];
                        let other_dist = ((hand_wrist.x - other_wrist.x) as f32)
                            .hypot((hand_wrist.y - other_wrist.y) as f32);
                        let score = (owner_dist - other_dist) / owner_dist.max(1.0);
                        if score > best_other_score {
                            best_other_score = score;
                        }
                    }

                    // Stolen hand detected: replace with synthetic fallback
                    if best_other_score > 0.40 {
                        let dir = Vec2::new(
                            (owner_wrist.x - owner_elbow.x) as f32,
                            (owner_wrist.y - owner_elbow.y) as f32,
                        );
                        let mut dir_len = dir.length();
                        let unit_dir = if dir_len > 1e-4 {
                            dir / dir_len
                        } else {
                            dir_len = 30.0;
                            Vec2::new(0.0, 1.0)
                        };

                        let fallback = generate_fallback_hand(
                            Vec2::new(owner_wrist.x as f32, owner_wrist.y as f32),
                            unit_dir,
                            dir_len,
                        );
                        let pts: Vec<Point> = fallback
                            .iter()
                            .map(|p| Point::new(p.x.round() as i32, p.y.round() as i32, 0.55))
                            .collect();
                        if side == "left" {
                            tracks[track_idx].hands.left = Some(pts);
                        } else {
                            tracks[track_idx].hands.right = Some(pts);
                        }
                    }
                }
            }
        }
    }
}

/// Multi-person tracking and trajectory association engine.
#[derive(Debug, Clone)]
pub struct MultiPersonTracker {
    config: TrackerConfig,
    tracks: Vec<PersonTrack>,
    next_id: u64,
}

impl MultiPersonTracker {
    /// Creates a new MultiPersonTracker with the given configuration.
    pub fn new(config: TrackerConfig) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            next_id: 1,
        }
    }

    /// Returns active tracks.
    pub fn tracks(&self) -> &[PersonTrack] {
        &self.tracks
    }

    /// Updates tracks from per-frame person detections and optionally color-profiles actors.
    pub fn update(&mut self, dets: &[BodyDetection], frame: &image::RgbImage) -> Vec<PersonTrack> {
        let mut detection_list: Vec<BodyDetection> = dets.to_vec();

        // Extract color histogram if not already provided
        for det in &mut detection_list {
            if det.color_hist.is_none() {
                det.color_hist = Some(extract_torso_histogram(frame, det.box_));
            }
        }

        // Bipartite matching by highest composite score
        let mut assignments: Vec<(usize, usize)> = Vec::new();
        let mut used_tracks = vec![false; self.tracks.len()];
        let mut used_dets = vec![false; detection_list.len()];

        let mut candidate_pairs = Vec::new();
        for (t_idx, track) in self.tracks.iter().enumerate() {
            for (d_idx, det) in detection_list.iter().enumerate() {
                let score = score_detection_to_track(det, track);
                if score >= self.config.match_threshold {
                    candidate_pairs.push((score, t_idx, d_idx));
                }
            }
        }

        candidate_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        for (_score, t_idx, d_idx) in candidate_pairs {
            if !used_tracks[t_idx] && !used_dets[d_idx] {
                used_tracks[t_idx] = true;
                used_dets[d_idx] = true;
                assignments.push((t_idx, d_idx));
            }
        }

        // Update matched tracks
        for (t_idx, d_idx) in assignments {
            let det = &detection_list[d_idx];
            let track = &mut self.tracks[t_idx];

            let (old_cx, old_cy) = track.center();
            let new_cx = (det.box_.x1 + det.box_.x2) as f32 * 0.5;
            let new_cy = (det.box_.y1 + det.box_.y2) as f32 * 0.5;

            track.velocity = (new_cx - old_cx, new_cy - old_cy);
            track.box_ = det.box_;
            track.body_points = det.body_points.clone();
            track.tracker_id = det.tracker_id;
            track.missed_frames = 0;
            track.hit_streak += 1;

            if let Some(ref hist) = det.color_hist {
                track.color_signature = track
                    .color_signature
                    .blend(hist, self.config.color_blend_alpha);
            }
        }

        // Increment missed frames for unmatched tracks
        for (t_idx, used) in used_tracks.iter().enumerate() {
            if !used {
                self.tracks[t_idx].missed_frames += 1;
                self.tracks[t_idx].hit_streak = 0;
            }
        }

        // Create new tracks for unmatched detections if under max_people
        for (d_idx, used) in used_dets.iter().enumerate() {
            if !used && self.tracks.len() < self.config.max_people {
                let det = &detection_list[d_idx];
                let track = PersonTrack {
                    id: self.next_id,
                    tracker_id: det.tracker_id,
                    box_: det.box_,
                    body_points: det.body_points.clone(),
                    velocity: (0.0, 0.0),
                    color_signature: det.color_hist.unwrap_or_default(),
                    missed_frames: 0,
                    hit_streak: 1,
                    label: None,
                    hands: HandsPair::default(),
                };
                self.next_id += 1;
                self.tracks.push(track);
            }
        }

        // Prune expired tracks
        let max_hold = self.config.track_hold_frames;
        self.tracks.retain(|t| t.missed_frames <= max_hold);

        // Resolve cross-person hands
        resolve_cross_person_hand_ownership(&mut self.tracks);

        self.tracks.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_bbox_iou_identical() {
        let box_a = BoundingBox::new(10, 10, 50, 50);
        let box_b = BoundingBox::new(10, 10, 50, 50);
        assert!((compute_bbox_iou(box_a, box_b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_bbox_iou_disjoint() {
        let box_a = BoundingBox::new(0, 0, 10, 10);
        let box_b = BoundingBox::new(20, 20, 30, 30);
        assert_eq!(compute_bbox_iou(box_a, box_b), 0.0);
    }

    #[test]
    fn test_color_histogram_similarity() {
        let mut h1 = ColorHistogram::default();
        let mut h2 = ColorHistogram::default();
        h1.bins[0] = 0.6;
        h1.bins[1] = 0.4;

        h2.bins[0] = 0.5;
        h2.bins[1] = 0.5;

        let sim = compare_color_histograms(&h1, &h2);
        assert!((sim - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_multi_person_tracker_lifecycle() {
        let config = TrackerConfig::default();
        let mut tracker = MultiPersonTracker::new(config);
        let frame = image::RgbImage::new(640, 480);

        let det = BodyDetection {
            box_: BoundingBox::new(100, 100, 200, 300),
            body_points: vec![Point::new(150, 150, 0.9); 17],
            score: 0.95,
            color_hist: None,
            tracker_id: Some(101),
        };

        let tracks = tracker.update(&[det.clone()], &frame);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].id, 1);
        assert_eq!(tracks[0].tracker_id, Some(101));

        // Track updates next frame with slight displacement
        let mut det2 = det.clone();
        det2.box_ = BoundingBox::new(105, 105, 205, 305);
        let tracks2 = tracker.update(&[det2], &frame);
        assert_eq!(tracks2.len(), 1);
        assert_eq!(tracks2[0].id, 1);
        assert_eq!(tracks2[0].velocity, (5.0, 5.0));
    }
}
