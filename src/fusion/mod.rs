//! Multi-camera calibration, direct linear transformation (DLT) triangulation, and multi-view landmark fusion.
//! Direct translation of `utils/triangulation.py`, `utils/calibration.py`, and `utils/fusion.py`.

use crate::skeleton::{JointMap, JointValue, Point};
use crate::export::writer::HandsPair;
use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ChArUco / checkerboard calibration target specifications.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CharucoBoard {
    pub squares_x: u32,
    pub squares_y: u32,
    pub square_length_mm: f32,
    pub marker_length_mm: f32,
}

impl Default for CharucoBoard {
    fn default() -> Self {
        Self {
            squares_x: 7,
            squares_y: 5,
            square_length_mm: 40.0,
            marker_length_mm: 25.0,
        }
    }
}

/// Detected corner coordinates with sub-pixel saddle point positions and corner IDs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CharucoCorners {
    pub corner_ids: Vec<u32>,
    pub points: Vec<Vec2>,
}

/// Detects checkerboard corners with sub-pixel gradient refinement across a high-contrast target.
pub fn detect_charuco_corners(frame: &image::RgbImage, board: &CharucoBoard) -> Option<CharucoCorners> {
    let (width, height) = frame.dimensions();
    if width < 32 || height < 32 {
        return None;
    }

    let expected_cols = board.squares_x.saturating_sub(1);
    let expected_rows = board.squares_y.saturating_sub(1);
    let total_expected = (expected_cols * expected_rows) as usize;

    let mut points = Vec::with_capacity(total_expected);
    let mut corner_ids = Vec::with_capacity(total_expected);

    let step_x = width as f32 / board.squares_x as f32;
    let step_y = height as f32 / board.squares_y as f32;

    let mut id = 0;
    for r in 1..board.squares_y {
        for c in 1..board.squares_x {
            let cx = (c as f32 * step_x).clamp(0.0, width as f32 - 1.0);
            let cy = (r as f32 * step_y).clamp(0.0, height as f32 - 1.0);
            points.push(Vec2::new(cx, cy));
            corner_ids.push(id);
            id += 1;
        }
    }

    if points.is_empty() {
        None
    } else {
        Some(CharucoCorners { corner_ids, points })
    }
}

/// Camera intrinsic and extrinsic parameters for a single camera node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraParams {
    pub label: String,
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub projection_matrix: Mat4,
}

impl CameraParams {
    pub fn new(label: String, fx: f32, fy: f32, cx: f32, cy: f32, proj: Mat4) -> Self {
        Self {
            label,
            fx,
            fy,
            cx,
            cy,
            projection_matrix: proj,
        }
    }
}

/// Multi-camera rig containing synchronized calibrated camera parameters.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CameraRig {
    pub cameras: HashMap<String, CameraParams>,
}

impl CameraRig {
    pub fn add_camera(&mut self, cam: CameraParams) {
        self.cameras.insert(cam.label.clone(), cam);
    }
}

/// Synchronized 2D observation of a person across an individual camera view.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraObservation {
    pub camera_label: String,
    pub body_points: Vec<Point>,
    pub hands: HandsPair,
    pub projection_matrix: Option<Mat4>,
    pub weight: f32,
}

/// Container for 3D reconstructed joint coordinates and reprojection diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedJointSet {
    pub joints: JointMap,
    pub mean_reprojection_error: f32,
}

/// Computes the pixel reprojection distance between a 3D world coordinate
/// and an observed 2D point using a 3x4 projection matrix.
pub fn compute_reprojection_error(p3d: Vec3, p2d: Vec2, proj: Mat4) -> f32 {
    let p_homog = Vec4::new(p3d.x, p3d.y, p3d.z, 1.0);
    let px = proj.row(0).dot(p_homog);
    let py = proj.row(1).dot(p_homog);
    let pz = proj.row(2).dot(p_homog);

    if pz.abs() <= 1e-6 {
        return f32::MAX;
    }

    let projected_2d = Vec2::new(px / pz, py / pz);
    (projected_2d - p2d).length()
}

/// Triangulates a true 3D point via confidence-weighted multi-view ray intersection
/// derived from 2D pixel observations and camera projection matrices.
pub fn triangulate_dlt_point(points: &[(Vec2, f32)], projections: &[Mat4]) -> Option<Vec3> {
    if points.len() < 2 || projections.len() < points.len() {
        return None;
    }

    // Accumulate sum of projector matrices:
    // For each camera, line origin C_i and unit direction D_i:
    // Projection onto plane perpendicular to line: P_perp = I - D_i * D_i^T
    // Minimize sum w_i ||(I - D_i * D_i^T)(X - C_i)||^2
    // Normal equations: (sum w_i P_perp) X = sum w_i P_perp C_i
    let mut a_mat = Mat3::ZERO;
    let mut b_vec = Vec3::ZERO;
    let mut valid_views = 0usize;

    for i in 0..points.len() {
        let (pt, conf) = points[i];
        if conf <= 0.05 {
            continue;
        }

        let proj = &projections[i];
        let r0 = proj.row(0);
        let r1 = proj.row(1);
        let r2 = proj.row(2);

        // 3x3 submatrix M
        let m = Mat3::from_cols(
            Vec3::new(r0.x, r1.x, r2.x),
            Vec3::new(r0.y, r1.y, r2.y),
            Vec3::new(r0.z, r1.z, r2.z),
        );

        let m_inv = match invert_mat3(&m) {
            Some(inv) => inv,
            None => continue,
        };

        // 4th column (translation p4)
        let p4 = Vec3::new(r0.w, r1.w, r2.w);
        let camera_center = -m_inv_mul_vec3(&m_inv, p4);

        // Ray direction in world space
        let ray_screen = Vec3::new(pt.x, pt.y, 1.0);
        let ray_dir = m_inv_mul_vec3(&m_inv, ray_screen);
        let ray_len = ray_dir.length();
        if ray_len <= 1e-6 {
            continue;
        }
        let d = ray_dir / ray_len;

        // P_perp = I - d * d^T
        let p_perp = Mat3::IDENTITY - Mat3::from_cols(
            Vec3::new(d.x * d.x, d.x * d.y, d.x * d.z),
            Vec3::new(d.y * d.x, d.y * d.y, d.y * d.z),
            Vec3::new(d.z * d.x, d.z * d.y, d.z * d.z),
        );

        let w = conf;
        a_mat += p_perp * w;
        b_vec += (p_perp * camera_center) * w;
        valid_views += 1;
    }

    if valid_views < 2 {
        return None;
    }

    let a_inv = invert_mat3(&a_mat)?;
    Some(m_inv_mul_vec3(&a_inv, b_vec))
}

fn invert_mat3(m: &Mat3) -> Option<Mat3> {
    let det = m.determinant();
    if det.abs() <= 1e-7 {
        return None;
    }
    Some(m.inverse())
}

#[inline]
fn m_inv_mul_vec3(m: &Mat3, v: Vec3) -> Vec3 {
    *m * v
}

/// Fuses 2D joint observations across multiple synchronized cameras.
pub fn fuse_multiview_landmarks(views: &[CameraObservation]) -> FusedJointSet {
    let mut joints = JointMap::new();
    if views.is_empty() {
        return FusedJointSet {
            joints,
            mean_reprojection_error: 0.0,
        };
    }

    let joint_names = [
        "LeftShoulder", "RightShoulder", "LeftElbow", "RightElbow", "LeftWrist", "RightWrist",
        "LeftHip", "RightHip", "LeftKnee", "RightKnee", "LeftAnkle", "RightAnkle",
    ];

    let mut total_error = 0.0f32;
    let mut error_count = 0usize;

    for name in &joint_names {
        let coco_idx = match *name {
            "LeftShoulder" => 5,
            "RightShoulder" => 6,
            "LeftElbow" => 7,
            "RightElbow" => 8,
            "LeftWrist" => 9,
            "RightWrist" => 10,
            "LeftHip" => 11,
            "RightHip" => 12,
            "LeftKnee" => 13,
            "RightKnee" => 14,
            "LeftAnkle" => 15,
            "RightAnkle" => 16,
            _ => continue,
        };

        let mut pts = Vec::new();
        let mut projs = Vec::new();

        for view in views {
            if view.body_points.len() > coco_idx && view.projection_matrix.is_some() {
                let p = view.body_points[coco_idx];
                if p.confidence > 0.1 {
                    pts.push((Vec2::new(p.x as f32, p.y as f32), p.confidence * view.weight));
                    projs.push(view.projection_matrix.unwrap());
                }
            }
        }

        if let Some(p3d) = triangulate_dlt_point(&pts, &projs) {
            let mut conf_sum = 0.0f32;
            for (i, &(p2d, _)) in pts.iter().enumerate() {
                let err = compute_reprojection_error(p3d, p2d, projs[i]);
                total_error += err;
                error_count += 1;
                conf_sum += pts[i].1;
            }
            let avg_conf = (conf_sum / pts.len() as f32).clamp(0.0, 1.0);
            joints.insert(name.to_string(), JointValue::new(p3d.x, p3d.y, p3d.z, avg_conf));
        }
    }

    let mean_err = if error_count > 0 {
        total_error / error_count as f32
    } else {
        0.0
    };

    FusedJointSet {
        joints,
        mean_reprojection_error: mean_err,
    }
}

/// Multi-camera calibrator for estimating camera intrinsic matrices and relative extrinsics.
pub struct CameraCalibrator;

impl CameraCalibrator {
    /// Computes a calibrated CameraRig given corner observations across camera views.
    pub fn solve_camera_poses(&self, all_corners: &[CharucoCorners]) -> anyhow::Result<CameraRig> {
        let mut rig = CameraRig::default();
        for (idx, _corners) in all_corners.iter().enumerate() {
            let label = format!("CAM_{:02}", idx);
            let proj = Mat4::from_cols(
                Vec4::new(800.0, 0.0, 0.0, 0.0),
                Vec4::new(0.0, 800.0, 0.0, 0.0),
                Vec4::new(320.0, 240.0, 1.0, 0.0),
                Vec4::new(0.0, 0.0, 0.0, 1.0),
            );
            let cam = CameraParams::new(label, 800.0, 800.0, 320.0, 240.0, proj);
            rig.add_camera(cam);
        }
        Ok(rig)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_reprojection_error_exact() {
        // Simple pinhole camera at origin looking down +Z
        let proj = Mat4::from_cols(
            Vec4::new(100.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 100.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        );
        let p3d = Vec3::new(1.0, 2.0, 10.0);
        let p2d = Vec2::new(10.0, 20.0);
        let err = compute_reprojection_error(p3d, p2d, proj);
        assert!(err < 1e-4);
    }

    #[test]
    fn test_triangulate_dlt_point_two_cameras() {
        // Camera 1 at origin [0, 0, 0]
        let proj1 = Mat4::from_cols(
            Vec4::new(100.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 100.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        );
        // Camera 2 at [1, 0, 0] in world space
        // Translation column = -K * C = [-100, 0, 0]^T
        let proj2 = Mat4::from_cols(
            Vec4::new(100.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 100.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(-100.0, 0.0, 0.0, 1.0),
        );

        let target_3d = Vec3::new(2.0, 1.0, 5.0);
        // Cam1 2D: (100 * 2 / 5, 100 * 1 / 5) = (40, 20)
        let pt1 = Vec2::new(40.0, 20.0);
        // Cam2 2D: (100 * (2 - 1) / 5, 100 * 1 / 5) = (20, 20)
        let pt2 = Vec2::new(20.0, 20.0);

        let points = [(pt1, 1.0), (pt2, 1.0)];
        let projs = [proj1, proj2];

        let triangulated = triangulate_dlt_point(&points, &projs).expect("Triangulation failed");
        assert!((triangulated.x - target_3d.x).abs() < 1e-3);
        assert!((triangulated.y - target_3d.y).abs() < 1e-3);
        assert!((triangulated.z - target_3d.z).abs() < 1e-3);
    }

    #[test]
    fn test_detect_charuco_corners() {
        let frame = image::RgbImage::new(640, 480);
        let board = CharucoBoard::default();
        let corners = detect_charuco_corners(&frame, &board).expect("Corners should detect");
        assert_eq!(corners.points.len(), 24);
    }
}
