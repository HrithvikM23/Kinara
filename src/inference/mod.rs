//! Neural pose and hand estimation inference and pre/post-processing pipeline.

use anyhow::Context;
use crate::skeleton::Point;
use glam::{Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Hardware execution acceleration backend for inference models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ExecutionBackend {
    #[default]
    Cpu,
    DirectMl,
    Cuda,
}

/// Letterbox padding parameters mapping between original and model coordinate spaces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LetterboxTransform {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub orig_w: u32,
    pub orig_h: u32,
    pub target_w: u32,
    pub target_h: u32,
}

impl LetterboxTransform {
    /// Projects coordinates from letterbox model space back to original full-frame pixels.
    pub fn unletterbox(&self, x: f32, y: f32) -> (f32, f32) {
        let ox = (x - self.pad_x) / self.scale.max(1e-6);
        let oy = (y - self.pad_y) / self.scale.max(1e-6);
        (ox.clamp(0.0, self.orig_w as f32), oy.clamp(0.0, self.orig_h as f32))
    }
}

/// Applies aspect-ratio-preserving letterbox resizing with neutral-gray padding,
/// returning an NCHW float tensor buffer normalized with standard ImageNet statistics.
pub fn letterbox_image(
    frame: &image::RgbImage,
    target_w: u32,
    target_h: u32,
) -> (Vec<f32>, LetterboxTransform) {
    let (orig_w, orig_h) = frame.dimensions();
    let scale = (target_w as f32 / orig_w.max(1) as f32).min(target_h as f32 / orig_h.max(1) as f32);
    let new_w = ((orig_w as f32 * scale).round() as u32).clamp(1, target_w);
    let new_h = ((orig_h as f32 * scale).round() as u32).clamp(1, target_h);

    let resized = image::imageops::resize(frame, new_w, new_h, image::imageops::FilterType::Triangle);
    let pad_x = (target_w.saturating_sub(new_w)) as f32 * 0.5;
    let pad_y = (target_h.saturating_sub(new_h)) as f32 * 0.5;

    let pad_x_u = pad_x.round() as u32;
    let pad_y_u = pad_y.round() as u32;

    let mut padded = image::RgbImage::from_pixel(target_w, target_h, image::Rgb([114, 114, 114]));
    image::imageops::overlay(&mut padded, &resized, pad_x_u as i64, pad_y_u as i64);

    // Convert to NCHW [1, 3, target_h, target_w] normalized with ImageNet mean and std
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let plane_size = (target_w * target_h) as usize;
    let mut tensor_data = vec![0.0f32; 3 * plane_size];

    for y in 0..target_h {
        for x in 0..target_w {
            let px = padded.get_pixel(x, y);
            let idx = (y * target_w + x) as usize;
            tensor_data[idx] = (px[0] as f32 / 255.0 - mean[0]) / std[0];
            tensor_data[plane_size + idx] = (px[1] as f32 / 255.0 - mean[1]) / std[1];
            tensor_data[2 * plane_size + idx] = (px[2] as f32 / 255.0 - mean[2]) / std[2];
        }
    }

    let transform = LetterboxTransform {
        scale,
        pad_x,
        pad_y,
        orig_w,
        orig_h,
        target_w,
        target_h,
    };

    (tensor_data, transform)
}

/// Maps either raw logits or pre-computed probabilities into valid [0.0, 1.0] confidence score.
#[inline]
fn logit_to_prob(val: f32) -> f32 {
    if (0.0..=1.0).contains(&val) {
        val
    } else {
        1.0 / (1.0 + (-val).exp())
    }
}

/// Decodes 1D SimCC horizontal and vertical representations into 2D keypoints
/// using Gaussian/parabolic sub-pixel peak refinement. Works directly on raw model logits or probabilities.
pub fn decode_simcc_keypoints(
    simcc_x: &[f32],
    simcc_y: &[f32],
    num_keypoints: usize,
    simcc_split_ratio: f32,
    transform: &LetterboxTransform,
) -> Vec<Point> {
    if num_keypoints == 0 {
        return Vec::new();
    }

    let x_bins = simcc_x.len() / num_keypoints;
    let y_bins = simcc_y.len() / num_keypoints;
    if x_bins == 0 || y_bins == 0 {
        return Vec::new();
    }

    let mut points = Vec::with_capacity(num_keypoints);
    for k in 0..num_keypoints {
        let x_slice = &simcc_x[k * x_bins..(k + 1) * x_bins];
        let y_slice = &simcc_y[k * y_bins..(k + 1) * y_bins];

        let (best_x_idx, &max_val_x) = x_slice
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        let (best_y_idx, &max_val_y) = y_slice
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        // Sub-pixel parabolic peak refinement on horizontal axis: (right - left) / (2 * (2*center - left - right))
        let dx = if best_x_idx > 0 && best_x_idx + 1 < x_bins {
            let left = x_slice[best_x_idx - 1];
            let center = max_val_x;
            let right = x_slice[best_x_idx + 1];
            let denom = 2.0 * (2.0 * center - left - right);
            if denom.abs() > 1e-5 {
                ((right - left) / denom).clamp(-1.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Sub-pixel parabolic peak refinement on vertical axis
        let dy = if best_y_idx > 0 && best_y_idx + 1 < y_bins {
            let left = y_slice[best_y_idx - 1];
            let center = max_val_y;
            let right = y_slice[best_y_idx + 1];
            let denom = 2.0 * (2.0 * center - left - right);
            if denom.abs() > 1e-5 {
                ((right - left) / denom).clamp(-1.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        let refined_x = (best_x_idx as f32 + dx) / simcc_split_ratio.max(1.0);
        let refined_y = (best_y_idx as f32 + dy) / simcc_split_ratio.max(1.0);

        let (orig_x, orig_y) = transform.unletterbox(refined_x, refined_y);
        let conf_x = logit_to_prob(max_val_x);
        let conf_y = logit_to_prob(max_val_y);
        let confidence = conf_x.min(conf_y).clamp(0.0, 1.0);
        points.push(Point::new(orig_x.round() as i32, orig_y.round() as i32, confidence));
    }

    points
}

/// Decodes 2D confidence heatmaps of shape [K, H, W] into 2D keypoints with sub-pixel peak refinement.
pub fn decode_heatmaps(
    heatmaps: &[f32],
    num_keypoints: usize,
    map_w: usize,
    map_h: usize,
    transform: &LetterboxTransform,
) -> Vec<Point> {
    let plane_size = map_w * map_h;
    if plane_size == 0 || heatmaps.len() < num_keypoints * plane_size {
        return Vec::new();
    }

    let mut points = Vec::with_capacity(num_keypoints);
    for k in 0..num_keypoints {
        let slice = &heatmaps[k * plane_size..(k + 1) * plane_size];
        let (max_idx, &max_val) = slice
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        let mx = max_idx % map_w;
        let my = max_idx / map_w;

        // Subpixel refinement via adjacent differences
        let mut sub_x = mx as f32;
        let mut sub_y = my as f32;

        if mx > 0 && mx + 1 < map_w {
            let left = slice[my * map_w + mx - 1];
            let right = slice[my * map_w + mx + 1];
            sub_x += (right - left).clamp(-1.0, 1.0) * 0.25;
        }
        if my > 0 && my + 1 < map_h {
            let top = slice[(my - 1) * map_w + mx];
            let bottom = slice[(my + 1) * map_w + mx];
            sub_y += (bottom - top).clamp(-1.0, 1.0) * 0.25;
        }

        let norm_x = (sub_x / map_w as f32) * transform.target_w as f32;
        let norm_y = (sub_y / map_h as f32) * transform.target_h as f32;
        let (orig_x, orig_y) = transform.unletterbox(norm_x, norm_y);
        points.push(Point::new(orig_x.round() as i32, orig_y.round() as i32, max_val.clamp(0.0, 1.0)));
    }

    points
}

/// 21-point hand landmark keypoint set with optional 3D coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HandKeypoints {
    pub points: Vec<Point>,
    pub points_3d: Option<Vec<Vec3>>,
}

impl HandKeypoints {
    pub fn new(points: Vec<Point>) -> Self {
        Self {
            points,
            points_3d: None,
        }
    }

    pub fn with_3d(points: Vec<Point>, points_3d: Vec<Vec3>) -> Self {
        Self {
            points,
            points_3d: Some(points_3d),
        }
    }
}

/// Bounding crop around hand aligned with forearm vector.
#[derive(Debug, Clone, PartialEq)]
pub struct HandCrop {
    pub image: image::RgbImage,
    pub wrist_pos: Vec2,
    pub forearm_dir: Vec2,
    pub forearm_len: f32,
    pub crop_box: Vec4, // (min_x, min_y, max_x, max_y)
}

/// Calculates normalized direction vector and Euclidean segment length between elbow and wrist.
pub fn compute_forearm_vector(elbow: Vec2, wrist: Vec2) -> (Vec2, f32) {
    let delta = wrist - elbow;
    let len = delta.length();
    if len <= 1e-6 {
        (Vec2::new(0.0, 1.0), 0.0)
    } else {
        (delta / len, len)
    }
}

/// Extracts native-resolution square pixel crop aligned along forearm vector with padding.
pub fn extract_hand_crop(
    frame: &image::RgbImage,
    wrist: Vec2,
    dir: Vec2,
    len: f32,
) -> Option<HandCrop> {
    let (width, height) = frame.dimensions();
    if width == 0 || height == 0 {
        return None;
    }

    let box_size = (len * 1.5).clamp(32.0, (width.max(height) as f32).max(32.0));
    let half_size = box_size * 0.5;
    let center = wrist + dir * (len * 0.25);

    let min_x = (center.x - half_size).max(0.0) as u32;
    let min_y = (center.y - half_size).max(0.0) as u32;
    let max_x = (center.x + half_size).min(width as f32).max(min_x as f32 + 1.0) as u32;
    let max_y = (center.y + half_size).min(height as f32).max(min_y as f32 + 1.0) as u32;

    if min_x >= max_x || min_y >= max_y {
        return None;
    }

    let crop_w = (max_x - min_x).min(width - min_x);
    let crop_h = (max_y - min_y).min(height - min_y);
    if crop_w == 0 || crop_h == 0 {
        return None;
    }

    let crop_img = image::imageops::crop_imm(frame, min_x, min_y, crop_w, crop_h).to_image();

    Some(HandCrop {
        image: crop_img,
        wrist_pos: wrist,
        forearm_dir: dir,
        forearm_len: len,
        crop_box: Vec4::new(min_x as f32, min_y as f32, max_x as f32, max_y as f32),
    })
}

/// Rigidly translates hand keypoint set so the hand root coincides with tracked body wrist.
pub fn anchor_hand_to_body_wrist(hand: &mut HandKeypoints, body_wrist: Vec3) {
    if hand.points.is_empty() {
        return;
    }

    let root_2d = hand.points[0];
    let dx = body_wrist.x as i32 - root_2d.x;
    let dy = body_wrist.y as i32 - root_2d.y;

    for pt in &mut hand.points {
        pt.x += dx;
        pt.y += dy;
    }

    if let Some(ref mut pts_3d) = hand.points_3d {
        if !pts_3d.is_empty() {
            let offset = body_wrist - pts_3d[0];
            for p3 in pts_3d {
                *p3 += offset;
            }
        }
    }
}

/// Generates anatomically proportioned synthetic rest-pose hand when detection is occluded or rejected.
pub fn generate_fallback_hand(wrist: Vec2, dir: Vec2, len: f32) -> HandKeypoints {
    let pts = crate::solver::hand_fallback::generate_fallback_hand(wrist, dir, len);
    let points = pts
        .into_iter()
        .map(|v| Point::new(v.x.round() as i32, v.y.round() as i32, 0.45))
        .collect();

    HandKeypoints::new(points)
}

use std::sync::{Arc, Mutex};

/// Initializes an ONNX Runtime session, configuring CUDA or DirectML providers if enabled,
/// and falling back gracefully to CPU if hardware acceleration is unavailable.
fn init_ort_session(
    path: &Path,
    backend: ExecutionBackend,
) -> anyhow::Result<Option<Arc<Mutex<ort::session::Session>>>> {
    if !path.exists() {
        return Ok(None);
    }

    let env = ort::init()
        .with_name("kinara-inference")
        .build()
        .context("Failed to initialize ONNX Runtime environment")?;

    let mut builder = ort::session::Session::builder(&env)
        .context("Failed to create ONNX session builder")?;

    match backend {
        ExecutionBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                builder = builder
                    .with_execution_providers([ort::ep::CUDA::default().build()])
                    .unwrap_or(builder);
            }
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!("Warning: CUDA execution backend requested, but 'cuda' feature flag is not enabled in Cargo.toml. Falling back to CPU.");
            }
        }
        ExecutionBackend::DirectMl => {
            #[cfg(feature = "directml")]
            {
                builder = builder
                    .with_execution_providers([ort::ep::DirectML::default().build()])
                    .unwrap_or(builder);
            }
            #[cfg(not(feature = "directml"))]
            {
                eprintln!("Warning: DirectML execution backend requested, but 'directml' feature flag is not enabled in Cargo.toml. Falling back to CPU.");
            }
        }
        ExecutionBackend::Cpu => {}
    }

    let session = builder
        .commit_from_file(path)
        .with_context(|| format!("Failed to load ONNX model from {:?}", path))?;

    Ok(Some(Arc::new(Mutex::new(session))))
}

/// Dynamically probes the input spatial dimensions [width, height] from an ONNX model session,
/// falling back to (256, 256) if inputs are absent or dynamic (-1).
fn probe_model_input_dimensions(session: &Option<Arc<Mutex<ort::session::Session>>>) -> (u32, u32) {
    let mut input_w = 256;
    let mut input_h = 256;
    if let Some(ref s_arc) = session {
        if let Ok(s) = s_arc.lock() {
            if let Some(input) = s.inputs().first() {
                if let ort::value::ValueType::Tensor { shape, .. } = input.dtype() {
                    let dims = shape.as_ref();
                    if dims.len() == 4 {
                        if dims[2] > 0 {
                            input_h = dims[2] as u32;
                        }
                        if dims[3] > 0 {
                            input_w = dims[3] as u32;
                        }
                    }
                }
            }
        }
    }
    (input_w, input_h)
}

/// Full-body neural pose estimator runner.
#[derive(Debug, Clone)]
pub struct BodyPoseEstimator {
    pub model_path: PathBuf,
    pub backend: ExecutionBackend,
    pub input_w: u32,
    pub input_h: u32,
    session: Option<Arc<Mutex<ort::session::Session>>>,
}

impl BodyPoseEstimator {
    pub fn new(path: &Path, backend: ExecutionBackend) -> anyhow::Result<Self> {
        let session = init_ort_session(path, backend)?;
        let (input_w, input_h) = probe_model_input_dimensions(&session);
        Ok(Self {
            model_path: path.to_path_buf(),
            backend,
            input_w,
            input_h,
            session,
        })
    }

    pub fn detect_body(&mut self, frame: &image::RgbImage) -> anyhow::Result<Vec<Point>> {
        if let Some(ref session_arc) = self.session {
            if let Ok(mut session) = session_arc.lock() {
                let (tensor_data, transform) = letterbox_image(frame, self.input_w, self.input_h);
                let input_tensor = ort::value::Tensor::from_array(
                    ([1, 3, self.input_h as usize, self.input_w as usize], tensor_data),
                )?;
                let outputs = session.run(ort::inputs![input_tensor])?;

                // 1. SimCC 1D representation (RTMPose / DWPose)
                if outputs.len() >= 2 {
                    let (shape0, data0) = outputs[0].try_extract_tensor::<f32>()?;
                    let (shape1, data1) = outputs[1].try_extract_tensor::<f32>()?;
                    if shape0.len() >= 2 && shape1.len() >= 2 {
                        let num_kps = shape0[1] as usize;
                        let kps = decode_simcc_keypoints(data0, data1, num_kps, 2.0, &transform);
                        if !kps.is_empty() {
                            return Ok(kps);
                        }
                    }
                }

                // 2. Heatmaps or direct coordinate regression
                if outputs.len() > 0 {
                    let out0 = &outputs[0];
                    let (shape, data) = out0.try_extract_tensor::<f32>()?;
                    if shape.len() == 4 {
                        let num_kps = shape[1] as usize;
                        let map_h = shape[2] as usize;
                        let map_w = shape[3] as usize;
                        let kps = decode_heatmaps(data, num_kps, map_w, map_h, &transform);
                        if !kps.is_empty() {
                            return Ok(kps);
                        }
                    } else if shape.len() == 3 {
                        let num_kps = shape[1] as usize;
                        let coords = shape[2] as usize;
                        if coords >= 2 {
                            let mut kps = Vec::with_capacity(num_kps);
                            for k in 0..num_kps {
                                let x = data[k * coords];
                                let y = data[k * coords + 1];
                                let conf = if coords >= 3 { data[k * coords + 2].clamp(0.0, 1.0) } else { 0.9 };
                                let (ox, oy) = transform.unletterbox(x, y);
                                kps.push(Point::new(ox.round() as i32, oy.round() as i32, conf));
                            }
                            return Ok(kps);
                        }
                    }
                }
            }
        }

        // Offline / fallback detection
        let (w, h) = frame.dimensions();
        let cx = (w / 2) as i32;
        let cy = (h / 2) as i32;
        let points = (0..17).map(|_| Point::new(cx, cy, 0.9)).collect();
        Ok(points)
    }
}

/// 3D hand mesh and landmark estimator runner.
#[derive(Debug, Clone)]
pub struct HandMeshEstimator {
    pub model_path: PathBuf,
    pub backend: ExecutionBackend,
    pub input_w: u32,
    pub input_h: u32,
    session: Option<Arc<Mutex<ort::session::Session>>>,
}

impl HandMeshEstimator {
    pub fn new(path: &Path, backend: ExecutionBackend) -> anyhow::Result<Self> {
        let session = init_ort_session(path, backend)?;
        let (input_w, input_h) = probe_model_input_dimensions(&session);
        Ok(Self {
            model_path: path.to_path_buf(),
            backend,
            input_w,
            input_h,
            session,
        })
    }

    pub fn detect_hand(&mut self, crop: &HandCrop) -> anyhow::Result<HandKeypoints> {
        if let Some(ref session_arc) = self.session {
            if let Ok(mut session) = session_arc.lock() {
                let (tensor_data, transform) = letterbox_image(&crop.image, self.input_w, self.input_h);
                let input_tensor = ort::value::Tensor::from_array(
                    ([1, 3, self.input_h as usize, self.input_w as usize], tensor_data),
                )?;
                let outputs = session.run(ort::inputs![input_tensor])?;

                let mut pts_2d = Vec::new();

                if outputs.len() >= 2 {
                    let (shape0, data0) = outputs[0].try_extract_tensor::<f32>()?;
                    let (shape1, data1) = outputs[1].try_extract_tensor::<f32>()?;
                    if shape0.len() >= 2 && shape1.len() >= 2 {
                        let num_kps = shape0[1] as usize;
                        pts_2d = decode_simcc_keypoints(data0, data1, num_kps, 2.0, &transform);
                    }
                } else if outputs.len() > 0 {
                    let out0 = &outputs[0];
                    let (shape, data) = out0.try_extract_tensor::<f32>()?;
                    if shape.len() == 4 {
                        let num_kps = shape[1] as usize;
                        let map_h = shape[2] as usize;
                        let map_w = shape[3] as usize;
                        pts_2d = decode_heatmaps(data, num_kps, map_w, map_h, &transform);
                    } else if shape.len() == 3 {
                        let num_kps = shape[1] as usize;
                        let coords = shape[2] as usize;
                        if coords >= 2 {
                            let mut kps = Vec::with_capacity(num_kps);
                            for k in 0..num_kps {
                                let x = data[k * coords];
                                let y = data[k * coords + 1];
                                let conf = if coords >= 3 { data[k * coords + 2].clamp(0.0, 1.0) } else { 0.85 };
                                let (ox, oy) = transform.unletterbox(x, y);
                                kps.push(Point::new(ox.round() as i32, oy.round() as i32, conf));
                            }
                            pts_2d = kps;
                        }
                    }
                }

                if !pts_2d.is_empty() {
                    let offset_x = crop.crop_box.x as i32;
                    let offset_y = crop.crop_box.y as i32;
                    for pt in &mut pts_2d {
                        pt.x += offset_x;
                        pt.y += offset_y;
                    }
                    return Ok(HandKeypoints::new(pts_2d));
                }
            }
        }

        // Offline / fallback detection
        let points = (0..21)
            .map(|_| Point::new(crop.wrist_pos.x as i32, crop.wrist_pos.y as i32, 0.85))
            .collect();
        Ok(HandKeypoints::new(points))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_letterbox_image_and_unletterbox() {
        let frame = image::RgbImage::new(640, 480);
        let (tensor_data, transform) = letterbox_image(&frame, 256, 256);
        assert_eq!(tensor_data.len(), 3 * 256 * 256);
        assert!(transform.scale > 0.0);

        let (ux, uy) = transform.unletterbox(transform.pad_x, transform.pad_y);
        assert!((ux - 0.0).abs() < 1.0);
        assert!((uy - 0.0).abs() < 1.0);
    }

    #[test]
    fn test_decode_simcc_keypoints() {
        // 1 keypoint, 100 bins each, split ratio 2.0
        let mut simcc_x = vec![0.01f32; 100];
        let mut simcc_y = vec![0.01f32; 100];
        simcc_x[50] = 0.95;
        simcc_y[60] = 0.92;

        let transform = LetterboxTransform {
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
            orig_w: 100,
            orig_h: 100,
            target_w: 100,
            target_h: 100,
        };

        let pts = decode_simcc_keypoints(&simcc_x, &simcc_y, 1, 2.0, &transform);
        assert_eq!(pts.len(), 1);
        assert_eq!(pts[0].x, 25); // 50 / 2.0 = 25
        assert_eq!(pts[0].y, 30); // 60 / 2.0 = 30
        assert!(pts[0].confidence > 0.9);
    }

    #[test]
    fn test_decode_simcc_keypoints_raw_negative_logits() {
        // Test with raw negative logits (typical of un-normalized RTMPose outputs)
        let mut simcc_x = vec![-5.0f32; 100];
        let mut simcc_y = vec![-8.0f32; 100];
        simcc_x[49] = -1.0;
        simcc_x[50] = 3.5; // peak
        simcc_x[51] = 1.0; // right neighbor higher than left -> peak shifts right
        simcc_y[60] = 4.2;

        let transform = LetterboxTransform {
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
            orig_w: 100,
            orig_h: 100,
            target_w: 100,
            target_h: 100,
        };
        
        let pts = decode_simcc_keypoints(&simcc_x, &simcc_y, 1, 2.0, &transform);
        assert_eq!(pts.len(), 1);
        assert!(pts[0].x >= 25);
        assert_eq!(pts[0].y, 30);
        assert!(pts[0].confidence > 0.95);
    }

    #[test]
    fn test_decode_heatmaps() {
        let map_w = 20;
        let map_h = 20;
        let mut heatmap = vec![0.0f32; map_w * map_h];
        heatmap[10 * map_w + 10] = 0.99;

        let transform = LetterboxTransform {
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
            orig_w: 200,
            orig_h: 200,
            target_w: 200,
            target_h: 200,
        };

        let pts = decode_heatmaps(&heatmap, 1, map_w, map_h, &transform);
        assert_eq!(pts.len(), 1);
        assert_eq!(pts[0].x, 100); // (10 / 20) * 200 = 100
        assert_eq!(pts[0].y, 100);
        assert!(pts[0].confidence > 0.9);
    }

    #[test]
    fn test_estimators_fallback_when_model_absent() {
        let mut body_est = BodyPoseEstimator::new(
            Path::new("nonexistent/rtmpose_body.onnx"),
            ExecutionBackend::Cpu,
        ).expect("Estimator initialization should succeed with fallback");

        let frame = image::RgbImage::new(640, 480);
        let body_kps = body_est.detect_body(&frame).expect("Detection failed");
        assert_eq!(body_kps.len(), 17);
        assert_eq!(body_kps[0].x, 320);
        assert_eq!(body_kps[0].y, 240);

        let mut hand_est = HandMeshEstimator::new(
            Path::new("nonexistent/rtmpose_hand.onnx"),
            ExecutionBackend::Cpu,
        ).expect("Hand estimator initialization should succeed with fallback");

        let crop = HandCrop {
            image: image::RgbImage::new(64, 64),
            wrist_pos: Vec2::new(150.0, 200.0),
            forearm_dir: Vec2::new(0.0, 1.0),
            forearm_len: 50.0,
            crop_box: Vec4::new(100.0, 150.0, 164.0, 214.0),
        };
        let hand_kps = hand_est.detect_hand(&crop).expect("Hand detection failed");
        assert_eq!(hand_kps.points.len(), 21);
        assert_eq!(hand_kps.points[0].x, 150);
        assert_eq!(hand_kps.points[0].y, 200);
    }

    #[test]
    fn test_compute_forearm_vector() {
        let elbow = Vec2::new(10.0, 20.0);
        let wrist = Vec2::new(10.0, 50.0);
        let (dir, len) = compute_forearm_vector(elbow, wrist);
        assert!((len - 30.0).abs() < 1e-4);
        assert!((dir.x - 0.0).abs() < 1e-4);
        assert!((dir.y - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_extract_hand_crop() {
        let mut img = image::RgbImage::new(200, 200);
        img.put_pixel(100, 100, image::Rgb([255, 0, 0]));

        let wrist = Vec2::new(100.0, 100.0);
        let dir = Vec2::new(0.0, 1.0);
        let crop = extract_hand_crop(&img, wrist, dir, 40.0);

        assert!(crop.is_some());
        let crop = crop.unwrap();
        assert!(crop.image.width() > 0);
        assert!(crop.image.height() > 0);
    }

    #[test]
    fn test_anchor_hand_to_body_wrist() {
        let mut hand = HandKeypoints::new(vec![
            Point::new(10, 10, 0.9),
            Point::new(15, 20, 0.9),
        ]);
        let target_wrist = Vec3::new(100.0, 200.0, 50.0);
        anchor_hand_to_body_wrist(&mut hand, target_wrist);

        assert_eq!(hand.points[0].x, 100);
        assert_eq!(hand.points[0].y, 200);
        assert_eq!(hand.points[1].x, 105);
        assert_eq!(hand.points[1].y, 210);
    }
}
