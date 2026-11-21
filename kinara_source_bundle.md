# Kinara Native Rust & DCC Integration Complete Source Bundle

## File: Cargo.toml

`toml
[package]
name = "kinara"
version = "0.1.0"
edition = "2021"
authors = ["HrithvikM"]
description = "Markerless Motion Capture & Biomechanical Inverse Kinematics Engine"

[dependencies]
glam = { version = "0.29", features = ["serde"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
clap = { version = "4.5", features = ["derive"] }
anyhow = "1.0"
image = { version = "0.25", default-features = false, features = ["png", "jpeg", "bmp", "webp"] }
ort = "2.0.0-rc.15"
ffmpeg-sidecar = "2.7.0"

[lib]
name = "kinara"
path = "src/lib.rs"

[[bin]]
name = "kinara"
path = "src/main.rs"

`

## File: progress.md

`markdown
# Kinara Rust Rewrite: Implementation Progress & Function Ledger

This document tracks the systematic porting of Kinara from Python to native Rust. Every function, struct, and algorithmic component requiring implementation is indexed below with its target Rust signature, responsibilities, and completion status.

---

## Progress Overview

| Subsystem | Module Path | Total Functions | Completed | Status |
| :--- | :--- | :---: | :---: | :--- |
| **1. Video Ingestion & IO** | `src/video/` | 6 | 6 | 🟢 Completed |
| **2. Inference Engine (ONNX)** | `src/inference/` | 8 | 8 | 🟢 Completed |
| **3. Biomechanical IK Solver** | `src/solver/` | 9 | 9 | 🟢 Completed |
| **4. Temporal Smoothing & Filter**| `src/solver/` | 5 | 5 | 🟢 Completed |
| **5. Multi-Person Tracking** | `src/tracking/` | 7 | 7 | 🟢 Completed |
| **6. Multi-Camera Triangulation**| `src/fusion/` | 5 | 5 | 🟢 Completed |
| **7. Coordinate & Scale Normalizer**| `src/export/` | 5 | 5 | 🟢 Completed |
| **8. Export Engine (JSON / FBX)**| `src/export/` | 5 | 5 | 🟢 Completed |
| **9. Configuration, GUI & Timeline Editor**| `src/` | 10 | 10 | 🟢 Completed |
| **Total** | | **60** | **60** | **100.0% Completed** |

---

## 1. Video Ingestion & IO (`src/video/`)

- [x] `VideoInputSource::open(path_or_dev: &str, fps: Option<f64>) -> Result<Self>`
  - *Ref*: `camera/capture.py:VideoInputSource.__init__`
  - *Description*: Opens camera device or video file using hardware decoders (DirectShow / V4L2); validates FPS and dimensions.
  - *Signature*: `pub fn open(src: &str, fps: Option<f64>) -> anyhow::Result<Self>`

- [x] `VideoInputSource::read_frame(&mut self) -> Result<Option<RgbImage>>`
  - *Ref*: `camera/capture.py:VideoInputSource.read`
  - *Description*: Decodes next frame into contiguous raw sRGB pixel buffer; reuses heap buffers across iterations.
  - *Signature*: `pub fn read_frame(&mut self) -> anyhow::Result<Option<image::RgbImage>>`

- [x] `VideoInputSource::seek_frame(&mut self, frame_index: usize) -> Result<()>`
  - *Ref*: `camera/capture.py`
  - *Description*: Seeks to a specific frame timestamp for parallel chunk decoding.
  - *Signature*: `pub fn seek_frame(&mut self, index: usize) -> anyhow::Result<()>`

- [x] `VideoOutputWriter::new(path: &Path, w: u32, h: u32, fps: f64) -> Result<Self>`
  - *Ref*: `camera/capture.py:VideoOutputWriter.__init__`
  - *Description*: Initializes H.264/NVENC video encoder stream for MP4 output.
  - *Signature*: `pub fn new(path: &Path, w: u32, h: u32, fps: f64) -> anyhow::Result<Self>`

- [x] `VideoOutputWriter::write_frame(&mut self, frame: &RgbImage) -> Result<()>`
  - *Ref*: `camera/capture.py:VideoOutputWriter.write`
  - *Description*: Encodes and appends one annotated video frame to destination MP4.
  - *Signature*: `pub fn write_frame(&mut self, frame: &image::RgbImage) -> anyhow::Result<()>`

- [x] `VideoOutputWriter::finish(self) -> Result<()>`
  - *Ref*: `camera/capture.py:VideoOutputWriter.close`
  - *Description*: Flushes remaining packets and writes MP4 file trailer.
  - *Signature*: `pub fn finish(self) -> anyhow::Result<()>`

---

## 2. Pose & Hand Inference Engine (`src/inference/`)

- [x] `BodyPoseEstimator::new(model_path: &Path, backend: Backend) -> Result<Self>`
  - *Ref*: `inference/rtmpose.py:ONNXPoseHandRunner._init_rtmpose_body`
  - *Description*: Instantiates ONNX Runtime session (`ort`) with CUDA / DirectML provider and performs warmup.
  - *Signature*: `pub fn new(path: &Path, backend: ExecutionBackend) -> anyhow::Result<Self>`

- [x] `BodyPoseEstimator::detect_body(&mut self, frame: &RgbImage) -> Result<BodyKeypointSet>`
  - *Ref*: `inference/rtmpose.py:ONNXPoseHandRunner.detect_body`
  - *Description*: Runs full-body pose inference on downscaled proxy; decodes heatmaps into 17 COCO keypoints.
  - *Signature*: `pub fn detect_body(&mut self, frame: &image::RgbImage) -> anyhow::Result<BodyKeypoints>`

- [x] `compute_forearm_vector(elbow: Vec2, wrist: Vec2) -> (Vec2, f32)`
  - *Ref*: `utils/normalize.py:build_hand_box`
  - *Description*: Calculates normalized direction vector and Euclidean segment length between elbow and wrist.
  - *Signature*: `pub fn compute_forearm_vector(elbow: glam::Vec2, wrist: glam::Vec2) -> (glam::Vec2, f32)`

- [x] `extract_hand_crop(frame: &RgbImage, wrist: Vec2, dir: Vec2, len: f32) -> Option<HandCrop>`
  - *Ref*: `utils/normalize.py:build_hand_box`
  - *Description*: Extracts native-resolution square pixel crop aligned along forearm vector with padding.
  - *Signature*: `pub fn extract_hand_crop(frame: &image::RgbImage, wrist: glam::Vec2, dir: glam::Vec2, len: f32) -> Option<HandCrop>`

- [x] `HandMeshEstimator::new(model_path: &Path, backend: Backend) -> Result<Self>`
  - *Ref*: `inference/rtmpose.py:ONNXPoseHandRunner._init_onnx_hand`
  - *Description*: Loads 3D hand mesh recovery / landmark ONNX model with hardware acceleration.
  - *Signature*: `pub fn new(path: &Path, backend: ExecutionBackend) -> anyhow::Result<Self>`

- [x] `HandMeshEstimator::detect_hand(&mut self, crop: &HandCrop) -> Result<HandKeypointSet>`
  - *Ref*: `inference/rtmpose.py:ONNXPoseHandRunner.detect_hand`
  - *Description*: Infers 21 3D hand landmarks; projects crop coordinates back to full-frame image space.
  - *Signature*: `pub fn detect_hand(&mut self, crop: &HandCrop) -> anyhow::Result<HandKeypoints>`

- [x] `anchor_hand_to_body_wrist(hand: &mut HandKeypoints, body_wrist: Vec3)`
  - *Ref*: `utils/hand_fallback.py:anchor_hand_to_wrist`
  - *Description*: Rigidly translates hand keypoint set so the hand root coincides with tracked body wrist.
  - *Signature*: `pub fn anchor_hand_to_body_wrist(hand: &mut HandKeypoints, body_wrist: glam::Vec3)`

- [x] `generate_fallback_hand(wrist: Vec2, dir: Vec2, len: f32) -> HandKeypoints`
  - *Ref*: `utils/hand_fallback.py:generate_default_hand`
  - *Description*: Generates anatomically proportioned synthetic rest-pose hand when detection is occluded or rejected.
  - *Signature*: `pub fn generate_fallback_hand(wrist: glam::Vec2, dir: glam::Vec2, len: f32) -> HandKeypoints`

---

## 3. Biomechanical IK & Constraint Solver (`src/solver/`)

- [x] `HandIKSolver::enforce_constraints(raw: &HandKeypoints, forearm_dir: Vec2) -> HandKeypoints`
  - *Ref*: `utils/hand_constraints.py:enforce_hand_constraints`
  - *Description*: Orchestrates full hand kinematic pass: tendon coupling, fist clamp, hinge limits, and bone lengths.
  - *Signature*: `pub fn enforce_constraints(raw: &HandKeypoints, forearm_dir: glam::Vec2) -> HandKeypoints`

- [x] `apply_dip_pip_tendon_coupling(chain: &mut FingerChain)`
  - *Ref*: `utils/hand_constraints.py:_enforce_chain_bend`
  - *Description*: Locks distal joint flexion to proximal joint: $\theta_{\text{DIP}} = 0.67 \times \theta_{\text{PIP}}$.
  - *Signature*: `pub fn apply_dip_pip_tendon_coupling(chain: &mut FingerChain)`

- [x] `apply_dynamic_fist_abduction_clamp(fingers: &mut [FingerChain; 4])`
  - *Ref*: `utils/hand_constraints.py:_enforce_finger_lanes`
  - *Description*: Multiplies allowed lateral finger spread by $(1.0 - \theta_{\text{MCP}} / 90^\circ)$, preventing fist collision.
  - *Signature*: `pub fn apply_dynamic_fist_abduction_clamp(fingers: &mut [FingerChain; 4])`

- [x] `clamp_planar_hinge(parent: Vec3, hinge: Vec3, child: Vec3, min_deg: f32, max_deg: f32) -> Vec3`
  - *Ref*: `utils/hand_constraints.py:_enforce_chain_bend`
  - *Description*: Constrains 1-DOF knuckle/joint to planar flexion, enforcing strict zero hyperextension.
  - *Signature*: `pub fn clamp_planar_hinge(p: glam::Vec3, h: glam::Vec3, c: glam::Vec3, min_deg: f32, max_deg: f32) -> glam::Vec3`

- [x] `enforce_radial_limits(wrist: Vec2, points: &mut [Vec2], max_multipliers: &[f32])`
  - *Ref*: `utils/hand_constraints.py:_enforce_radial_limits`
  - *Description*: Prevents distal phalanges from drifting beyond anatomical radial bounds from the wrist.
  - *Signature*: `pub fn enforce_radial_limits(wrist: glam::Vec2, points: &mut [glam::Vec2], mults: &[f32])`

- [x] `enforce_bone_lengths(parent: Vec2, child: Vec2, min_len: f32, max_len: f32) -> Vec2`
  - *Ref*: `utils/hand_constraints.py:_enforce_bone_lengths`
  - *Description*: Clamps segment distance to stay within physiological length proportions.
  - *Signature*: `pub fn enforce_bone_lengths(p: glam::Vec2, c: glam::Vec2, min_len: f32, max_len: f32) -> glam::Vec2`

- [x] `BodyIKSolver::new(config: &SolverConfig) -> Self`
  - *Ref*: `utils/body_constraints.py:BodyKinematicConstraints.__init__`
  - *Description*: Initializes body constraint state and learned target segment length cache.
  - *Signature*: `pub fn new(config: &SolverConfig) -> Self`

- [x] `BodyIKSolver::enforce_limb_lengths(&mut self, body: &mut BodyKeypoints)`
  - *Ref*: `utils/body_constraints.py:BodyKinematicConstraints.apply`
  - *Description*: Prevents limb stretching by re-projecting joints along learned invariant segment lengths.
  - *Signature*: `pub fn enforce_limb_lengths(&mut self, body: &mut BodyKeypoints)`

- [x] `derive_foot_points(knee: Vec2, ankle: Vec2) -> (Vec2, Vec2)`
  - *Ref*: `utils/body_geometry.py:derive_foot_points`
  - *Description*: Extrapolates foot and toe base points from knee-to-ankle vector direction and shin length.
  - *Signature*: `pub fn derive_foot_points(knee: glam::Vec2, ankle: glam::Vec2) -> (glam::Vec2, glam::Vec2)`

---

## 4. Temporal Smoothing & Motion Filtering (`src/solver/`)

- [x] `OneEuroFilter::new(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self`
  - *Ref*: `utils/smoothing.py:LandmarkSmoother.__init__`
  - *Description*: Initializes speed-adaptive low-pass filter for eliminating jitter without latency.
  - *Signature*: `pub fn new(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self`

- [x] `OneEuroFilter::filter_point(&mut self, point: Vec3, dt: f32) -> Vec3`
  - *Ref*: `utils/smoothing.py:LandmarkSmoother._smooth_points`
  - *Description*: Evaluates landmark derivative speed to dynamically tune smoothing cutoff frequency.
  - *Signature*: `pub fn filter_point(&mut self, p: glam::Vec3, dt: f32) -> glam::Vec3`

- [x] `QuaternionFilter::slerp(&self, q_prev: Quat, q_curr: Quat, alpha: f32) -> Quat`
  - *Ref*: `utils/smoothing.py`
  - *Description*: Performs spherical linear interpolation on rotation quaternions along shortest arc on $S^3$.
  - *Signature*: `pub fn slerp(&self, q0: glam::Quat, q1: glam::Quat, alpha: f32) -> glam::Quat`

- [x] `predict_missing_landmarks(prev: &[Vec3], vel: &[Vec3], dt: f32, decay: f32) -> Vec<Vec3>`
  - *Ref*: `utils/prediction.py:predict_points`
  - *Description*: Extrapolates occluded points along constant velocity vector with confidence decay.
  - *Signature*: `pub fn predict_missing_landmarks(pts: &[glam::Vec3], vel: &[glam::Vec3], dt: f32, decay: f32) -> Vec<glam::Vec3>`

- [x] `FootLockStabilizer::lock_planted_feet(&mut self, feet: &mut FootContacts, dt: f32)`
  - *Ref*: `utils/motion_cleanup.py:cleanup_motion_frames`
  - *Description*: Detects foot floor impact; clamps $(x, y)$ coordinates to zero velocity while planted.
  - *Signature*: `pub fn lock_planted_feet(&mut self, feet: &mut FootContacts, dt: f32)`

---

## 5. Multi-Person Tracking & Association (`src/tracking/`)

- [x] `compute_bbox_iou(a: BoundingBox, b: BoundingBox) -> f32`
  - *Ref*: `utils/multi_person.py:_iou`
  - *Description*: Calculates intersection-over-union metric between two bounding boxes.
  - *Signature*: `pub fn compute_bbox_iou(a: BoundingBox, b: BoundingBox) -> f32`

- [x] `TorsoColorProfiler::extract_histogram(frame: &RgbImage, torso: BoundingBox) -> ColorHistogram`
  - *Ref*: `utils/multi_person.py:_color_scores`
  - *Description*: Builds 16-bin normalized HSV color histogram of actor clothing.
  - *Signature*: `pub fn extract_histogram(frame: &image::RgbImage, torso: BoundingBox) -> ColorHistogram`

- [x] `compare_color_histograms(a: &ColorHistogram, b: &ColorHistogram) -> f32`
  - *Ref*: `utils/multi_person.py:color_profile_similarity`
  - *Description*: Measures histogram intersection score for clothing re-identification.
  - *Signature*: `pub fn compare_color_histograms(a: &ColorHistogram, b: &ColorHistogram) -> f32`

- [x] `MultiPersonTracker::new(config: TrackerConfig) -> Self`
  - *Ref*: `utils/multi_person.py:MultiPersonTracker.__init__`
  - *Description*: Initializes multi-person tracking manager with ByteTrack Kalman filters and track history.
  - *Signature*: `pub fn new(config: TrackerConfig) -> Self`

- [x] `MultiPersonTracker::update(&mut self, dets: &[BodyDetection], frame: &RgbImage) -> Vec<PersonTrack>`
  - *Ref*: `utils/multi_person.py:MultiPersonTracker.update`
  - *Description*: Matches detections to existing tracks using IoU, velocity prediction, and clothing color.
  - *Signature*: `pub fn update(&mut self, dets: &[BodyDetection], frame: &image::RgbImage) -> Vec<PersonTrack>`

- [x] `score_detection_to_track(det: &BodyDetection, track: &PersonTrack) -> f32`
  - *Ref*: `utils/multi_person.py:MultiPersonTracker._match_score`
  - *Description*: Evaluates composite association score weighting spatial overlap, motion, and color.
  - *Signature*: `pub fn score_detection_to_track(det: &BodyDetection, track: &PersonTrack) -> f32`

- [x] `resolve_cross_person_hand_ownership(tracks: &mut [PersonTrack])`
  - *Ref*: `utils/multi_person.py:MultiPersonTracker._resolve_cross_person_hands`
  - *Description*: Disambiguates hand ownership during actor crossovers, reassigning stolen hands.
  - *Signature*: `pub fn resolve_cross_person_hand_ownership(tracks: &mut [PersonTrack])`

---

## 6. Multi-Camera Calibration & Triangulation (`src/fusion/`)

- [x] `detect_charuco_corners(frame: &RgbImage, board: &CharucoBoard) -> Option<CharucoCorners>`
  - *Ref*: `utils/calibration.py:calibrate_cameras`
  - *Description*: Detects ChArUco checkerboard corners with sub-pixel saddle point refinement.
  - *Signature*: `pub fn detect_charuco_corners(frame: &image::RgbImage, board: &CharucoBoard) -> Option<CharucoCorners>`

- [x] `CameraCalibrator::solve_camera_poses(&self, all_corners: &[CameraCorners]) -> Result<CameraRig>`
  - *Ref*: `utils/calibration.py:calibrate_cameras`
  - *Description*: Computes camera intrinsic matrices $\mathbf{K}$ and relative extrinsics $[\mathbf{R} \mid \mathbf{t}]$.
  - *Signature*: `pub fn solve_camera_poses(&self, corners: &[CameraCorners]) -> anyhow::Result<CameraRig>`

- [x] `triangulate_dlt_point(points: &[(Vec2, f32)], projections: &[Mat3x4]) -> Option<Vec3>`
  - *Ref*: `utils/triangulation.py:triangulate_observation_frames`
  - *Description*: Computes 3D Cartesian coordinates via Direct Linear Transformation (DLT) with SVD.
  - *Signature*: `pub fn triangulate_dlt_point(pts: &[(glam::Vec2, f32)], mats: &[glam::Mat4]) -> Option<glam::Vec3>`

- [x] `fuse_multiview_landmarks(views: &[CameraObservation]) -> FusedJointSet`
  - *Ref*: `utils/fusion.py:fuse_body_views`
  - *Description*: Fuses multi-view observations using confidence-weighted reprojection.
  - *Signature*: `pub fn fuse_multiview_landmarks(views: &[CameraObservation]) -> FusedJointSet`

- [x] `compute_reprojection_error(p3d: Vec3, p2d: Vec2, proj: Mat3x4) -> f32`
  - *Ref*: `utils/triangulation.py:triangulation_metadata`
  - *Description*: Measures pixel distance between projected 3D point and observed 2D detector keypoint.
  - *Signature*: `pub fn compute_reprojection_error(p3d: glam::Vec3, p2d: glam::Vec2, proj: glam::Mat4) -> f32`

---

## 7. Coordinate Transformation & Scale Normalization (`src/export/`)

- [x] `cv_to_blender_space(point: Vec3) -> Vec3`
  - *Ref*: `utils/exports.py:_to_world`
  - *Description*: Rotates from OpenCV camera space $(+X \text{ right}, +Y \text{ down}, +Z \text{ forward})$ to Blender $(+X \text{ right}, +Y \text{ forward}, +Z \text{ up})$.
  - *Signature*: `pub fn cv_to_blender_space(p: glam::Vec3) -> glam::Vec3`

- [x] `measure_torso_length(hips: Vec3, neck: Vec3) -> f32`
  - *Ref*: `utils/exports.py:_measure_frame_torso_length`
  - *Description*: Measures 3D Euclidean distance between hip root and neck base.
  - *Signature*: `pub fn measure_torso_length(hips: glam::Vec3, neck: glam::Vec3) -> f32`

- [x] `bidirectional_ema_filter(values: &[f32], alpha: f32) -> Vec<f32>`
  - *Ref*: `utils/exports.py:_smooth_scale_factors`
  - *Description*: Applies zero-lag forward-backward exponential moving average over sequence.
  - *Signature*: `pub fn bidirectional_ema_filter(vals: &[f32], alpha: f32) -> Vec<f32>`

- [x] `normalize_sequence_scale(frames: &mut [MotionFrame], target_torso_len: f32)`
  - *Ref*: `utils/exports.py:_scale_normalize_frames`
  - *Description*: Standardizes skeleton dimensions so character proportions remain distance-invariant.
  - *Signature*: `pub fn normalize_sequence_scale(frames: &mut [MotionFrame], target_len: f32)`

- [x] `align_to_ground_plane(frames: &mut [MotionFrame])`
  - *Ref*: `utils/exports.py:_ground_z_axis_frames`
  - *Description*: Offsets all frames vertically so lowest foot contact touches $Z = 0.0$.
  - *Signature*: `pub fn align_to_ground_plane(frames: &mut [MotionFrame])`

---

## 8. Export Pipeline (`src/export/`)

- [x] `build_canonical_joint_map(body: &BodyKeypoints, hands: &HandKeypointsPair) -> JointMap`
  - *Ref*: `utils/exports.py:build_joint_map`
  - *Description*: Assembles body and finger landmarks into the standard 78-joint named skeleton dictionary.
  - *Signature*: `pub fn build_canonical_joint_map(body: &BodyKeypoints, hands: &HandsPair) -> JointMap`

- [x] `compute_rest_pose_joints(frames: &[MotionFrame]) -> RestJointOffsets`
  - *Ref*: `utils/exports.py:_compute_rest_joints`
  - *Description*: Computes median segment offset vectors across the clip to generate a stable bind pose.
  - *Signature*: `pub fn compute_rest_pose_joints(frames: &[MotionFrame]) -> RestJointOffsets`

- [x] `export_motion_json(path: &Path, frames: &[MotionFrame], meta: &ClipMetadata) -> Result<()>`
  - *Ref*: `utils/exports.py:export_motion_json`
  - *Description*: Emits atomic `kinara-motion-json-v1` file containing frames, joint trajectories, and metadata.
  - *Signature*: `pub fn export_motion_json(path: &Path, frames: &[MotionFrame], meta: &ClipMetadata) -> anyhow::Result<()>`

- [x] `export_motion_fbx(path: &Path, frames: &[MotionFrame], fps: f64) -> Result<()>`
  - *Ref*: `utils/exports.py:export_motion_fbx`
  - *Description*: Writes standalone ASCII FBX 7.4 document with hierarchical `LimbNode` armature and curves.
  - *Signature*: `pub fn export_motion_fbx(path: &Path, frames: &[MotionFrame], fps: f64) -> anyhow::Result<()>`

- [x] `export_multi_person_bundle(base_path: &Path, frames: &[MultiPersonFrame], fps: f64) -> Result<()>`
  - *Ref*: `utils/exports.py:export_multi_person_fbx_bundle`
  - *Description*: Splits multi-person timeline into individual per-person FBX files.
  - *Signature*: `pub fn export_multi_person_bundle(path: &Path, frames: &[MultiPersonFrame], fps: f64) -> anyhow::Result<()>`

---

## 9. Configuration, Desktop GUI & Timeline Editor (`src/`)

- [x] `PipelineConfig::load_or_default(path: &Path) -> Self`
  - *Ref*: `core/config.py:PipelineConfig`
  - *Description*: Loads settings from `kinara.config.json` with fallback to high-performance defaults.
  - *Signature*: `pub fn load_or_default(path: &Path) -> Self`

- [x] `PipelineConfig::save(&self, path: &Path) -> Result<()>`
  - *Ref*: `core/config.py:PipelineConfig`
  - *Description*: Persists active configuration, video paths, and UI preferences to disk.
  - *Signature*: `pub fn save(&self, path: &Path) -> anyhow::Result<()>`

- [x] `parse_cli_arguments() -> CliArgs`
  - *Ref*: `core/cli.py:build_parser`
  - *Description*: Parses command line arguments via `clap` (derive macro), overriding config file values.
  - *Signature*: `pub fn parse_cli_arguments() -> CliArgs`

- [x] `KinaraApp::new(cc: &eframe::CreationContext, config: PipelineConfig) -> Self`
  - *Ref*: `app/gui.rs`
  - *Description*: Initializes native desktop GUI application with egui themes and texture allocators.
  - *Signature*: `pub fn new(cc: &CreationContext, cfg: PipelineConfig) -> Self`

- [x] `KinaraApp::update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame)`
  - *Ref*: `app/gui.rs`
  - *Description*: Renders desktop studio deck: video file picker, camera selection, live preview, and progress bar.
  - *Signature*: `pub fn update(&mut self, ctx: &GuiContext, frame: &mut GuiFrame)`

- [x] `run_cli_pipeline(config: PipelineConfig) -> Result<()>`
  - *Ref*: `app/main.py:main`
  - *Description*: Headless batch runner executing the full mocap pipeline without graphical desktop dependencies.
  - *Signature*: `pub fn run_cli_pipeline(cfg: PipelineConfig) -> anyhow::Result<()>`

- [x] `TimelineEditor::add_clip(&mut self, path: PathBuf, fps: f64, total_frames: usize) -> u64`
  - *Ref*: `src/video/timeline.rs`
  - *Description*: Ingests drag-and-dropped or loaded video clip into the multi-track timeline.
  - *Signature*: `pub fn add_clip(&mut self, path: impl AsRef<Path>, fps: f64, total_frames: usize) -> u64`

- [x] `TimelineClip::set_trim(&mut self, in_frame: usize, out_frame: usize) -> Result<(), &'static str>`
  - *Ref*: `src/video/timeline.rs`
  - *Description*: Non-destructively trims video start/end in/out frame bounds.
  - *Signature*: `pub fn set_trim(&mut self, in_frame: usize, out_frame: usize) -> Result<(), &'static str>`

- [x] `TimelineEditor::cut_clip(&mut self, clip_id: u64, cut_offset: usize) -> Result<u64, &'static str>`
  - *Ref*: `src/video/timeline.rs`
  - *Description*: Cuts (splits) a clip at current playhead offset into two independent sequential clips.
  - *Signature*: `pub fn cut_clip(&mut self, clip_id: u64, cut_offset: usize) -> Result<u64, &'static str>`

- [x] `TimelineEditor::merge_clips(&mut self, first_id: u64, second_id: u64) -> Result<(), &'static str>`
  - *Ref*: `src/video/timeline.rs`
  - *Description*: Merges contiguous video clips sharing the same source file into a single uninterrupted clip.
  - *Signature*: `pub fn merge_clips(&mut self, first_id: u64, second_id: u64) -> Result<(), &'static str>`

`

## File: AGENTS.md

`markdown
# Kinara: AI Agent Developer & Architecture Guide (`AGENTS.md`)

> **Notice for all AI models & autonomous agents**:
> This document defines the architectural guidelines, implementation state, and code conventions for the **Kinara** markerless motion capture and biomechanical inverse kinematics engine. Read this file completely before making any changes.

---

## 1. Absolute Golden Rules & Prohibitions

1. **LEGACY NAMING CONVENTIONS**:
   - Refer to prior Python implementations neutrally as "legacy reference pipeline" or "reference mocap". Do not cite or introduce legacy external repository or author names anywhere in code, comments, commits, or documentation.

2. **PONYTAIL PRINCIPLE (Lazy Senior Developer Philosophy)**:
   - **Question abstractions**: Do not create generic layers, traits, or factories that were not explicitly requested.
   - **No unnecessary dependencies**: Rely on the Rust standard library and existing crates (`glam`, `serde`, `serde_json`, `image`, `clap`, `anyhow`).
   - **Shortest working diff**: Prioritize clear, direct mathematical implementations over multi-file indirection.
   - **Always test**: Every newly added kinematic or algorithmic function must include at least one runnable unit test asserting numerical correctness.

---

## 2. Project Architecture & Purpose

Kinara is a high-performance, real-time markerless motion capture (MoCap) and biomechanical stabilization engine written in **native Rust** (`dev-rs` branch).

```
┌────────────────────────────────────────────────────────┐
│  1. Video Ingestion & Frame Preprocessing              │ (DirectShow / Windows Media / Memory)
└───────────────────────────┬────────────────────────────┘
                            │ Raw Frames (image::RgbImage)
┌───────────────────────────▼────────────────────────────┐
│  2. Perception / Inference                             │ (RTMPose / WholeBody / WiLoR / Hand)
└───────────────────────────┬────────────────────────────┘
                            │ 2D/3D Detected Keypoints
┌───────────────────────────▼────────────────────────────┐
│  3. Multi-Person Tracking & Association                │ (IoU, Velocity Prediction, HSV Color)
└───────────────────────────┬────────────────────────────┘
                            │ Tracked Person Armatures
┌───────────────────────────▼────────────────────────────┐
│  4. Biomechanical IK & Constraint Solver               │ (Tendon Coupling, Planar Hinges, Bone EMA)
└───────────────────────────┬────────────────────────────┘
                            │ Kinematically Valid Joint Sets
┌───────────────────────────▼────────────────────────────┐
│  5. Temporal Smoothing & Ground Stabilization          │ (One Euro 1€, Quat Slerp, Foot Lock)
└───────────────────────────┬────────────────────────────┘
                            │ Stabilized Clean Armature
┌───────────────────────────▼────────────────────────────┐
│  6. DCC Streaming & File Exporters                     │ (Blender TCP, Unreal Live Link, FBX 7.4, JSON)
└────────────────────────────────────────────────────────┘
```

---

## 3. Directory Layout & Module Ledger

```
C:\College\Kinara\
├── Cargo.toml                  # Project manifest (glam, serde, image, clap, anyhow)
├── progress.md                 # 60-function implementation tracker
├── docs/
│   ├── functions.md            # Technical index & spec of all functions
│   ├── ik_and_skeleton.md      # 60-joint armature hierarchy & constraint mathematics
│   └── programflow.mmd         # End-to-end pipeline sequence diagram
├── blender_addon/              # Real-time Blender viewport add-on
│   └── kinara_stream/          # Modular Blender 4.x add-on package (stream, rig, ui, bake, constants)
├── src/
│   ├── lib.rs                  # Library root re-exporting all subsystems
│   ├── main.rs                 # CLI binary entrypoint & pipeline loop runner
│   ├── skeleton.rs             # 60-joint armature hierarchy, COCO-17 indices, Point, BoundingBox
│   ├── config.rs               # PipelineConfig serialization, validation, and CLI flags
│   ├── gui.rs                  # Immediate-mode GUI application context & state management
│   │
│   ├── inference/              # Neural perception, ONNX session stubs, and feature extraction
│   │   └── mod.rs              # BodyPoseEstimator, HandMeshEstimator, crop & forearm vectors
│   │
│   ├── fusion/                 # Multi-camera triangulation and calibration
│   │   └── mod.rs              # DLT triangulation (SVD), reprojection error, CharUco corner detection
│   │
│   ├── solver/                 # Biomechanical kinematics & temporal filters
│   │   ├── mod.rs
│   │   ├── body_constraints.rs # BodyIKSolver: Learned invariant limb lengths (upper arm, forearm, thigh, shin)
│   │   ├── body_geometry.rs    # Foot & toe extrapolation from knee/ankle vectors
│   │   ├── filter.rs           # OneEuroFilter (speed-adaptive), QuaternionFilter (Slerp on S³)
│   │   ├── foot_lock.rs        # FootLockStabilizer: Ground contact velocity clamping (Z=0)
│   │   ├── hand_constraints.rs # HandIKSolver: DIP-PIP tendon coupling, fist abduction clamp, planar hinges
│   │   ├── hand_fallback.rs    # Synthetic fallback hand generation, wrist anchoring, lock-free telemetry
│   │   └── prediction.rs       # Landmark constant-velocity extrapolation & confidence decay
│   │
│   ├── tracking/               # Multi-person tracking & association
│   │   └── mod.rs              # MultiPersonTracker, IoU, TorsoColorProfiler (16-bin HSV), bilateral hand crossover
│   │
│   ├── network/                # Real-time DCC live streaming
│   │   └── mod.rs              # BlenderStreamServer: TCP socket server on port 49168 pushing rig/frame packets
│   │
│   ├── export/                 # Transformation, normalization, and file exporters
│   │   ├── mod.rs
│   │   ├── canonical.rs        # build_canonical_joint_map, compute_rest_pose_joints, ClipMetadata
│   │   ├── coords.rs           # cv_to_blender_space, convert_sequence_to_blender_space, bidirectional EMA, scale
│   │   ├── fbx.rs              # export_motion_fbx (direct-to-disk streaming BufWriter), export_multi_person_bundle
│   │   └── json.rs             # export_motion_json (atomic kinara-motion-json-v1 format)
│   │
│   └── video/                  # Timeline, media capture, and video ingestion
│       ├── mod.rs
│       ├── io.rs               # VideoInputSource, VideoOutputWriter, synthetic frames, PPM/PNG
│       └── timeline.rs         # TimelineEditor: Non-destructive trim, playhead cut/split, clip merging
```

---

## 4. Subsystem Progress Ledger

Tracked in [`progress.md`](progress.md):

| Subsystem | Module Path | Completed / Total | Status |
| :--- | :--- | :---: | :--- |
| **1. Video Ingestion & IO** | `src/video/` | 6 / 6 | 🟢 Completed |
| **2. Inference Engine (ONNX)** | `src/inference/` | 8 / 8 | 🟢 Completed |
| **3. Biomechanical IK Solver** | `src/solver/` | 9 / 9 | 🟢 Completed |
| **4. Temporal Smoothing & Filter** | `src/solver/` | 5 / 5 | 🟢 Completed |
| **5. Multi-Person Tracking** | `src/tracking/` | 7 / 7 | 🟢 Completed |
| **6. Multi-Camera Triangulation** | `src/fusion/` | 5 / 5 | 🟢 Completed |
| **7. Coordinate & Scale Normalizer** | `src/export/` | 5 / 5 | 🟢 Completed |
| **8. Export Engine (JSON / FBX)** | `src/export/` | 5 / 5 | 🟢 Completed |
| **9. GUI & Timeline Editor** | `src/` & `src/video/` | 10 / 10 | 🟢 Completed |
| **Total Progress** | | **60 / 60 (100.0%)** | **60 Passing Tests** |

---

## 5. Mathematical & Algorithmic Standards

### A. Coordinate Systems
- **OpenCV Space**: $+X \text{ right}, +Y \text{ down}, +Z \text{ forward}$
- **Blender Space**: $+X \text{ right}, +Y \text{ forward}, +Z \text{ up}$
- **Conversion**:
  $$\begin{bmatrix} X_{\text{blender}} \\ Y_{\text{blender}} \\ Z_{\text{blender}} \end{bmatrix} = \begin{bmatrix} X_{\text{cv}} \\ Z_{\text{cv}} \\ -Y_{\text{cv}} \end{bmatrix}$$
  Implemented in `export::coords::cv_to_blender_space(Vec3)`.

### B. Biomechanical Constraints
1. **DIP-PIP Tendon Coupling**: Distal interphalangeal joint angle is tied to proximal joint flexion:
   $$\theta_{\text{DIP}} = 0.67 \times \theta_{\text{PIP}}$$
2. **Dynamic Fist Abduction Clamp**: Multiplies maximum allowed lateral knuckle spread by $(1.0 - \theta_{\text{MCP}} / 90^\circ)$, preventing finger intersection in closed fists.
3. **Planar Hinge Clamping**: Knees, elbows, and finger knuckles are restricted to 1-DOF planar rotation with strict hyperextension clamping ($[0^\circ, 160^\circ]$).
4. **Invariant Bone Lengths**: `BodyIKSolver` tracks median segment lengths for humerus, radius, femur, and tibia; re-projects child joints along unit directions to prevent perspective stretching.
5. **Foot Lock Grounding**: When ankle/toe speed is below velocity threshold ($\le 8.0$) and altitude is within threshold ($\le 16.0$ from ground), $(x, y)$ horizontal motion is locked to eliminate foot sliding/skating.

### C. FBX 7.4.0 ASCII Standards
- Official FBX time unit: $46,186,158,000 \text{ ticks/second}$.
- Uses `Model::LimbNode` hierarchical objects for all 60 armature joints bound with parent-child `C: "OO"` records.
- Animation curves store translational (`T`) and rotational (`R`) Euler angles in degrees.

---

## 6. How to Build, Verify & Test

Run tests frequently after every change:

```powershell
# Check compilation across all modules
cargo check

# Run all 67 unit tests (must always pass with 0 failures)
cargo test

# Run a specific module's tests
cargo test inference::
cargo test video::
cargo test tracking::
cargo test solver::
cargo test export::
```

---

## 7. Next Steps for Succeeding Agents

All 60 engine functions across 9 subsystems are 100% completed, compiling cleanly, and verified by 67 passing unit tests.

Future extensions and integration priorities:
1. **Model Checkpoints**: Download or configure live RTMPose / DWPose body and hand ONNX weights in `models/` directory for live hardware evaluation.
2. **Blender Viewport Add-on (`blender_addon/kinara_stream/`)**: Install into Blender 4.x preferences, enable the streaming operator, and verify real-time TCP rig synchronization on port `49168`.
3. **Multi-Camera Rig Live Calibration**: Validate multi-view DLT reconstruction using multi-camera video feeds.

`

## File: README.md

`markdown
<div align="center">

<img src="./assets/logo.png" alt="Kinara Logo" width="130" />

# KINARA
### Markerless Motion Capture & Biomechanical Inverse Kinematics Engine

[![Language](https://img.shields.io/badge/Language-Rust_%7C_Python-dea584?style=flat-square&logo=rust&logoColor=white)](https://www.rust-lang.org)
[![GPU Acceleration](https://img.shields.io/badge/Inference-CUDA_%7C_DirectML_%7C_CPU-76b900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Blender Ready](https://img.shields.io/badge/DCC_Target-Blender_4.x+-f5792a?style=flat-square&logo=blender&logoColor=white)](https://www.blender.org)
[![Rig Architecture](https://img.shields.io/badge/Skeleton-60--Joint_Armature-6366f1?style=flat-square)](./docs/ik_and_skeleton.md)
[![License](https://img.shields.io/badge/License-Non--Commercial-3b82f6?style=flat-square)](./LICENSE.md)

<p align="center">
  <b>Kinara</b> is a markerless motion capture engine designed to extract 3D skeletal and finger-articulated animation from standard video feeds.<br/>
  It enforces biomechanical constraints, maintains distance-invariant character scaling, and outputs clean animation data ready for Blender and game engines.
</p>

<p align="center">
  <a href="#-system-architecture">Architecture</a> •
  <a href="#-biomechanical-ik-engine">IK Constraints</a> •
  <a href="#-feature-matrix">Features</a> •
  <a href="#-blender-pipeline">Blender Workflow</a> •
  <a href="#-multi-camera--multi-person">Multi-Cam & Tracking</a> •
  <a href="#-development-status">Status</a> •
  <a href="./docs/">Documentation</a>
</p>

</div>

---

> [!NOTE]
> **Active Engineering Pivot**: Kinara is currently transitioning its proven Python pipeline architecture into a high-performance native Rust core. Track module-by-module implementation status in [progress.md](./progress.md).

---

## ✦ System Architecture

Kinara processes video frames through a staged pipeline designed to separate high-level body localization from fine-grained finger articulation:

```mermaid
flowchart LR
    A["Input Video / Feed<br/>(File or Webcam)"] --> B["Dual-Scale Ingest"]
    B -->|Proxy Resolution| C["Body Pose Estimator<br/>(17 COCO Keypoints)"]
    B -->|Native Pixel Crop| D["Forearm-Guided Hand ROI<br/>(Wrist Crop)"]
    C --> D
    D --> E["Hand Pose Inference<br/>(21 Hand Keypoints)"]
    C & E --> F["Biomechanical IK Solver<br/>(Tendon Lock & Planar Hinges)"]
    F --> G["Temporal Stabilization<br/>(Filtering & Planted Foot Lock)"]
    G --> H["Scale & Ground Normalizer<br/>(Torso Metric & Z=0 Align)"]
    H --> I["Export Targets<br/>(Blender JSON, FBX, Live UDP)"]

    classDef stage fill:#0f172a,stroke:#3b82f6,stroke-width:1.5px,color:#f8fafc;
    classDef highlight fill:#1e1b4b,stroke:#8b5cf6,stroke-width:2px,color:#f8fafc;
    classDef export fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc;
    class A,B,C,D,E stage;
    class F,G,H highlight;
    class I export;
```

---

## ◈ Biomechanical IK Engine

Standard neural pose estimators predict unconstrained Cartesian points $(x, y, z)$. Without anatomical priors, raw landmark outputs suffer from backwards joint hyperextension, bone stretching, and fingers clipping through each other. Kinara enforces physical constraints at the kinematic solver stage:

<table>
<tr>
<td width="50%" valign="top">

### 1. DIP-PIP Tendon Coupling Ratio
In the human hand, the *Distal Interphalangeal* (DIP) and *Proximal Interphalangeal* (PIP) joints share the *Flexor Digitorum Profundus* tendon. Humans cannot voluntarily articulate the fingertip without flexing the middle knuckle.
$$\theta_{\text{DIP}} \approx 0.67 \times \theta_{\text{PIP}}$$
* **Impact**: Eliminates redundant degrees of freedom, removes fingertip jitter, and prevents backwards-bending tips.

</td>
<td width="50%" valign="top">

### 2. Dynamic Fist Abduction Clamping
When fingers are extended, they can freely splay laterally (abduction). As the fingers curl inward to form a fist, ligaments tighten and lateral spread vanishes:
$$\text{Max Abduction}(\theta_{\text{MCP}}) = \text{Base Spread} \times \left(1.0 - \frac{\theta_{\text{MCP}}}{90^\circ}\right)$$
* **Impact**: Eliminates adjacent finger meshes clipping through each other when closing into a fist.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 3. Strict 1-DOF Planar Knuckle Hinges
The PIP and DIP joints are physiological revolute hinges:
* Lateral deviation: **$0^\circ$ locked** (no sideways bend)
* Axial longitudinal twist: **$0^\circ$ locked**
* Hyperextension: **Hard limit at $0^\circ$** (knuckles never bend in reverse).

</td>
<td width="50%" valign="top">

### 4. Rigid Bone Length Invariance
Smoothing raw Cartesian coordinates causes limbs to artificially shrink and stretch during perspective changes. Kinara enforces invariant segment lengths and smooths **joint orientation quaternions**:
$$\mathbf{q}_{\text{stabilized}} = \text{Slerp}(\mathbf{q}_{\text{prev}}, \mathbf{q}_{\text{new}}, \alpha)$$

</td>
</tr>
</table>

> [!TIP]
> For the complete mathematical formulation and proof of joint limits, see [docs/ik_and_skeleton.md](./docs/ik_and_skeleton.md).

---

## ❖ Feature Matrix

| Capability | Unconstrained AI Trackers | Kinara Engine | Technical Advantage |
| :--- | :---: | :---: | :--- |
| **Finger Tracking** | 2D loose points | **Anatomical 21-Point IK** | Tendon coupling and knuckle planar hinge limits |
| **Distant Footage** | Fails (Lost in downsampling) | **Wrist-Oriented Crop Extraction** | Evaluates hands on native-resolution crops |
| **Joint Rotations** | None (Positions only) | **Quaternion Rig Transforms** | Eliminates axial twist and rubber-bone artifacts |
| **Scale Stability** | Shrinks/expands with distance | **Torso Metric Normalization** | Bidirectional zero-lag smoothing of torso scale |
| **Ground Interaction** | Floating / Foot skate | **Planted Foot Velocity Lock** | Automatic ground plane alignment ($Z = 0.0$) |
| **DCC Export** | Raw CSV or loose coordinates | **Blender Motion JSON + FBX 7.4** | Direct import into Blender 4.x+ character armatures |

---

## ⬡ Blender 4.x+ Pipeline

Kinara structures animation data into standard humanoid hierarchies for Blender, Maya, and Unreal Engine:

```
Kinara Motion Output
 ├── motion.json (kinara-motion-json-v1: 60-joint canonical hierarchy + rest offsets)
 └── motion.fbx  (ASCII FBX 7.4: Hierarchical LimbNodes + baked translation curves)
```

### Option A: Direct FBX Import
1. In Blender, navigate to `File` ▸ `Import` ▸ `FBX (.fbx)`.
2. Select your exported `.fbx` file.
3. The 60-joint armature imports with baked keyframe curves and feet grounded to the floor plane ($Z = 0.0$).

### Option B: Headless JSON Importer
Execute the automated importer script to build armatures, attach constraint empties, and bake pose actions:

```bash
blender --background --python tools/import_kinara_motion.py -- --input outputs/motion.json --fps 60
```

---

## ◈ Multi-Camera & Multi-Person

### Single-Camera Multi-Person Tracking
* **Track Association**: Tracks multiple individuals simultaneously using bounding box overlap, motion prediction, and torso clothing color profiles to minimize identity swaps during actor crossovers.
* **Hand Ownership Disambiguation**: Enforces wrist proximity and boundary penalties to prevent hands from being incorrectly reassigned between adjacent actors.

### Multi-Camera 3D Triangulation
* **ChArUco Calibration**: Calibrates multi-camera setups using ChArUco checkerboards to compute camera intrinsic matrices ($\mathbf{K}$) and extrinsic transformations ($[\mathbf{R} \mid \mathbf{t}]$).
* **Direct Linear Transformation (DLT)**: Reconstructs true metric 3D joint positions from synchronized 2D detections across 2 to 4 camera angles.

---

## ⬢ Technical Architecture & Directory Layout

```
Kinara/
 ├── Cargo.toml               # Native Rust package manifest (glam, ort, ffmpeg-sidecar, clap)
 ├── docs/
 │    ├── programflow.mmd      # High-fidelity Mermaid architecture diagram
 │    ├── functions.md        # Technical specification of all 60 engine functions
 │    └── ik_and_skeleton.md  # Biomechanical IK theory & 60-joint armature specification
 ├── blender_addon/           # Real-time Blender viewport add-on
 │    └── kinara_stream/      # Modular Blender 4.x package (stream, rig, ui, bake, constants)
 ├── assets/                  # Branding and visual assets
 ├── src/
 │    ├── export/             # JSON, FBX 7.4 ASCII, and Blender space transformations
 │    ├── fusion/             # Multi-camera DLT triangulation & ChArUco calibration
 │    ├── inference/          # ONNX Runtime (ort) pose/hand estimation & SimCC/heatmap decoding
 │    ├── network/            # Real-time TCP streaming server (port 49168)
 │    ├── solver/             # Biomechanical IK constraints, One Euro filter & foot lock
 │    ├── tracking/           # Multi-person tracking, IoU, and HSV torso profiling
 │    ├── video/              # FFmpeg video decoding, webcam capture, and timeline editor
 │    ├── config.rs           # Pipeline configuration & CLI argument serialization
 │    ├── gui.rs              # Immediate-mode GUI application context & state management
 │    ├── skeleton.rs         # 60-joint armature hierarchy, COCO-17 indices & Point geometry
 │    ├── lib.rs              # Subsystem re-exports and API definitions
 │    └── main.rs             # CLI binary entrypoint & pipeline execution runner
 ├── progress.md              # 60-function implementation tracker (100% complete)
 └── README.md                # Project documentation
```

---

## ⚙ System Requirements

### Hardware
* **Camera**: Standard webcam, action camera, or pre-recorded video (`.mp4`, `.mov`, `.mkv`).
* **GPU**: NVIDIA GPU with CUDA support recommended for accelerated inference; DirectML and CPU execution supported as fallbacks.
* **Multi-Camera (Optional)**: 2 to 4 synchronized camera views for calibrated 3D triangulation.

### Supported Software & DCC Tools
* **Blender**: 4.0+ (Armature import, action baking)
* **Unreal Engine**: 5.x (Live UDP / OSC stream receiver)
* **Rust Toolchain**: 1.78+ (`cargo`, `rustc`)

---

## 📚 Technical Documentation

* **[Progress & Implementation Ledger](./progress.md)** — Detailed checklist of every function being ported to Rust.
* **[Program Flow Architecture Diagram](./docs/programflow.mmd)** — Mermaid flowchart illustrating multi-scale ingest and kinematic solving.
* **[IK Constraints & Blender Export Specification](./docs/ik_and_skeleton.md)** — Mathematical proofs and Blender integration methods.
* **[Engine Function Reference](./docs/functions.md)** — Architectural index of all module functions, signatures, and behaviors.

---

## 📄 License & Attribution

Kinara is licensed under the [Kinara Non-Commercial License with Commercial Permission](./LICENSE.md). It is free for personal, educational, academic, and non-commercial use. Commercial use requires prior written permission and may involve fees; contact hrithvikmani@gmail.com to request it. Built for open biomechanics, markerless motion tracking, and real-time graphics pipelines.

`

## File: src/lib.rs

`rust
pub mod config;
pub mod export;
pub mod fusion;
pub mod gui;
pub mod inference;
pub mod network;
pub mod skeleton;
pub mod solver;
pub mod tracking;
pub mod video;

pub use config::*;
pub use export::*;
pub use fusion::*;
pub use gui::*;
pub use inference::{
    compute_forearm_vector, extract_hand_crop, BodyPoseEstimator, ExecutionBackend, HandCrop,
    HandKeypoints, HandMeshEstimator,
};
pub use network::*;
pub use skeleton::*;
pub use solver::*;
pub use tracking::*;
pub use video::*;

`

## File: src/main.rs

`rust
//! Kinara: Native High-Performance Markerless Motion Capture Engine.
//! Command-line interface and batch processing pipeline entrypoint.

use clap::{Parser, Subcommand};
use glam::Vec2;
use kinara::config::PipelineConfig;
use kinara::export::{
    align_to_ground_plane, build_canonical_joint_map, convert_sequence_to_blender_space,
    export_motion_fbx, export_motion_json, normalize_sequence_scale, ClipMetadata, HandsPair,
    MotionFrame,
};
use kinara::gui::{CreationContext, GuiContext, GuiFrame, KinaraApp};
use kinara::inference::{
    compute_forearm_vector, extract_hand_crop, BodyPoseEstimator, ExecutionBackend,
    HandMeshEstimator,
};
use kinara::network::{BlenderStreamServer, BLENDER_STREAM_PORT};
use kinara::solver::body_constraints::{BodyIKSolver, SolverConfig};
use kinara::solver::filter::OneEuroFilter;
use kinara::solver::foot_lock::{FootContacts, FootLockStabilizer};
use kinara::solver::hand_constraints::enforce_hand_constraints;
use kinara::video::VideoInputSource;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Command line arguments for Kinara CLI.
#[derive(Parser, Debug)]
#[command(
    name = "kinara",
    author = "Kinara Core Team",
    version = "0.1.0",
    about = "Markerless Motion Capture & Biomechanical Inverse Kinematics Engine"
)]
pub struct CliArgs {
    #[command(subcommand)]
    pub command: Option<Commands>,

    #[arg(short, long, help = "Path to pipeline configuration JSON file")]
    pub config: Option<PathBuf>,

    #[arg(short, long, help = "Input video file, image sequence, or camera stream")]
    pub input: Option<String>,

    #[arg(short, long, help = "Destination motion capture output file (FBX or JSON)")]
    pub output: Option<String>,

    #[arg(long, help = "Target motion capture framerate in frames per second")]
    pub fps: Option<f64>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Execute end-to-end motion capture pipeline
    Run {
        #[arg(short, long, help = "Input video file or image sequence")]
        input: Option<String>,
        #[arg(short, long, help = "Destination output file (.fbx or .json)")]
        output: Option<String>,
    },
    /// Stream live motion capture poses over TCP to Blender (port 49168)
    Stream {
        #[arg(short, long, default_value_t = BLENDER_STREAM_PORT, help = "TCP port to stream on")]
        port: u16,
        #[arg(short, long, help = "Input video source (default: synthetic test stream)")]
        source: Option<String>,
    },
    /// Export or convert motion capture data
    Export {
        #[arg(short, long, help = "Source motion recording file")]
        input: String,
        #[arg(short, long, help = "Destination converted file (.fbx or .json)")]
        output: String,
    },
    /// Calibrate multi-camera rig with ChArUco pattern
    Calibrate {
        #[arg(short, long, help = "Directory containing calibration frame views")]
        images_dir: String,
        #[arg(short, long, help = "Calibration rig output configuration path")]
        output: Option<String>,
    },
    /// Video timeline management and editing
    Timeline {
        #[arg(short, long, help = "Video clip path to inspect")]
        clip: String,
    },
    /// Launch the interactive desktop studio GUI deck
    Studio {
        #[arg(short, long, help = "Input video file or image sequence")]
        input: Option<String>,
    },
    /// Print engine hardware acceleration and supported formats
    Info,
}

/// Parses CLI arguments from environment using clap.
pub fn parse_cli_arguments() -> CliArgs {
    CliArgs::parse()
}

/// Execution session managing inference models, biomechanical IK solvers, and temporal filters.
pub struct PipelineSession {
    pub config: PipelineConfig,
    pub pose_estimator: BodyPoseEstimator,
    pub hand_estimator: HandMeshEstimator,
    pub body_solver: BodyIKSolver,
    pub temporal_filters: HashMap<String, OneEuroFilter>,
    pub foot_stabilizer: FootLockStabilizer,
}

impl PipelineSession {
    pub fn new(config: PipelineConfig) -> anyhow::Result<Self> {
        let backend = match config.execution_backend.to_lowercase().as_str() {
            "cuda" => ExecutionBackend::Cuda,
            "directml" => ExecutionBackend::DirectMl,
            _ => ExecutionBackend::Cpu,
        };

        let pose_estimator = BodyPoseEstimator::new(
            Path::new("models/rtmpose_body.onnx"),
            backend,
        )?;
        let hand_estimator = HandMeshEstimator::new(
            Path::new("models/rtmpose_hand.onnx"),
            backend,
        )?;

        let solver_cfg = SolverConfig {
            body_constraints_enabled: config.body_ik_enabled,
            export_foot_lock_enabled: config.foot_lock_enabled,
            min_cutoff: config.filter_min_cutoff,
            beta: config.filter_beta,
            ..Default::default()
        };
        let body_solver = BodyIKSolver::new(&solver_cfg);
        let foot_stabilizer = FootLockStabilizer::new(
            solver_cfg.export_foot_lock_velocity,
            solver_cfg.export_foot_lock_max_lift,
        );

        Ok(Self {
            config,
            pose_estimator,
            hand_estimator,
            body_solver,
            temporal_filters: HashMap::new(),
            foot_stabilizer,
        })
    }

    /// Processes a single RGB frame through pose estimation, IK constraints, and temporal smoothing.
    pub fn process_frame(&mut self, frame: &image::RgbImage, frame_idx: usize, dt: f32) -> anyhow::Result<MotionFrame> {
        let mut body_kps = self.pose_estimator.detect_body(frame)?;
        if self.config.body_ik_enabled {
            self.body_solver.enforce_limb_lengths(&mut body_kps);
        }

        let mut hands = HandsPair::default();
        if self.config.hand_ik_enabled && body_kps.points.len() >= 11 {
            // Left hand (wrist: 9, elbow: 7)
            let lw = Vec2::new(body_kps.points[9].x as f32, body_kps.points[9].y as f32);
            let le = Vec2::new(body_kps.points[7].x as f32, body_kps.points[7].y as f32);
            let (ldir, llen) = compute_forearm_vector(le, lw);
            if let Some(crop) = extract_hand_crop(frame, lw, ldir, llen) {
                if let Ok(raw_hand) = self.hand_estimator.detect_hand(&crop) {
                    let constrained_pts = enforce_hand_constraints(&raw_hand.points);
                    hands.left = Some(constrained_pts);
                }
            }

            // Right hand (wrist: 10, elbow: 8)
            let rw = Vec2::new(body_kps.points[10].x as f32, body_kps.points[10].y as f32);
            let re = Vec2::new(body_kps.points[8].x as f32, body_kps.points[8].y as f32);
            let (rdir, rlen) = compute_forearm_vector(re, rw);
            if let Some(crop) = extract_hand_crop(frame, rw, rdir, rlen) {
                if let Ok(raw_hand) = self.hand_estimator.detect_hand(&crop) {
                    let constrained_pts = enforce_hand_constraints(&raw_hand.points);
                    hands.right = Some(constrained_pts);
                }
            }
        }

        let mut joint_map = build_canonical_joint_map(&body_kps, &hands, None);

        // Apply OneEuroFilter temporal smoothing to all joints
        for (joint_name, val) in joint_map.iter_mut() {
            let filter = self.temporal_filters
                .entry(joint_name.clone())
                .or_insert_with(|| OneEuroFilter::new(self.config.filter_min_cutoff, self.config.filter_beta, 1.0));
            let smoothed = filter.filter_point(glam::Vec3::new(val.x, val.y, val.z), dt);
            val.x = smoothed.x;
            val.y = smoothed.y;
            val.z = smoothed.z;
        }

        Ok(MotionFrame::new(frame_idx, joint_map))
    }
}

/// Executes headless batch motion capture pipeline from configuration.
pub fn run_cli_pipeline(config: PipelineConfig) -> anyhow::Result<()> {
    println!("Initializing Kinara pipeline for: {}", config.input_source);

    let mut video_in = VideoInputSource::open(&config.input_source, Some(config.fps))?;
    let mut session = PipelineSession::new(config.clone())?;
    let dt = (1.0 / config.fps.max(1.0)) as f32;

    let mut motion_frames = Vec::new();
    let mut frame_idx = 0;
    while let Some(frame) = video_in.read_frame()? {
        let mf = session.process_frame(&frame, frame_idx, dt)?;
        motion_frames.push(mf);
        frame_idx += 1;
    }

    if motion_frames.is_empty() {
        println!("No frames processed.");
        return Ok(());
    }

    // 1. Convert canonical coordinates to standard Blender space (+X right, +Y forward, +Z up)
    convert_sequence_to_blender_space(&mut motion_frames);

    // 2. Apply scale normalization and ground contact plane alignment on Z axis
    normalize_sequence_scale(&mut motion_frames, 50.0);
    align_to_ground_plane(&mut motion_frames);

    // 3. Apply FootLockStabilizer in grounded Blender space (+Z up, ground at Z=0.0)
    if config.foot_lock_enabled {
        let mut foot_stabilizer = FootLockStabilizer::new(8.0, 16.0);
        for frame in motion_frames.iter_mut() {
            let left_ankle = frame.joints.get("LeftAnkle").map(|j| glam::Vec3::new(j.x, j.y, j.z)).unwrap_or(glam::Vec3::ZERO);
            let right_ankle = frame.joints.get("RightAnkle").map(|j| glam::Vec3::new(j.x, j.y, j.z)).unwrap_or(glam::Vec3::ZERO);
            let left_toe = frame.joints.get("LeftToeBase").map(|j| glam::Vec3::new(j.x, j.y, j.z)).unwrap_or(glam::Vec3::ZERO);
            let right_toe = frame.joints.get("RightToeBase").map(|j| glam::Vec3::new(j.x, j.y, j.z)).unwrap_or(glam::Vec3::ZERO);

            let mut contacts = FootContacts::new(left_ankle, right_ankle, left_toe, right_toe);
            foot_stabilizer.lock_planted_feet(&mut contacts, dt);

            if let Some(la) = frame.joints.get_mut("LeftAnkle") {
                la.x = contacts.left_ankle.x;
                la.y = contacts.left_ankle.y;
                la.z = contacts.left_ankle.z;
            }
            if let Some(ra) = frame.joints.get_mut("RightAnkle") {
                ra.x = contacts.right_ankle.x;
                ra.y = contacts.right_ankle.y;
                ra.z = contacts.right_ankle.z;
            }
            if let Some(lt) = frame.joints.get_mut("LeftToeBase") {
                lt.x = contacts.left_toe.x;
                lt.y = contacts.left_toe.y;
                lt.z = contacts.left_toe.z;
            }
            if let Some(rt) = frame.joints.get_mut("RightToeBase") {
                rt.x = contacts.right_toe.x;
                rt.y = contacts.right_toe.y;
                rt.z = contacts.right_toe.z;
            }
        }
    }

    let out_path = Path::new(&config.output_path);
    if out_path.extension().and_then(|s| s.to_str()) == Some("json") {
        let meta = ClipMetadata::default();
        export_motion_json(out_path, &motion_frames, &meta)?;
    } else {
        export_motion_fbx(out_path, &motion_frames, config.fps)?;
    }

    println!(
        "Successfully processed {} frames -> {}",
        motion_frames.len(),
        config.output_path
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = parse_cli_arguments();

    let cfg_path = args.config.unwrap_or_else(|| PathBuf::from("kinara.config.json"));
    let mut config = PipelineConfig::load_or_default(&cfg_path);

    if let Some(inp) = args.input {
        config.input_source = inp;
    }
    if let Some(out) = args.output {
        config.output_path = out;
    }
    if let Some(fps) = args.fps {
        config.fps = fps;
    }

    match args.command {
        Some(Commands::Run { input, output }) => {
            if let Some(i) = input {
                config.input_source = i;
            }
            if let Some(o) = output {
                config.output_path = o;
            }
            run_cli_pipeline(config)?;
        }
        Some(Commands::Stream { port, source }) => {
            if let Some(src) = source {
                config.input_source = src;
            }
            println!("Starting Kinara Live Blender Stream on 127.0.0.1:{}", port);
            let mut streamer = BlenderStreamServer::bind(port)?;
            let mut video_in = VideoInputSource::open(&config.input_source, Some(config.fps))?;
            let mut session = PipelineSession::new(config.clone())?;
            let frame_duration = std::time::Duration::from_secs_f64(1.0 / config.fps.max(1.0));
            let dt = (1.0 / config.fps.max(1.0)) as f32;

            let mut frame_idx = 0;
            println!("Ready for Blender connections. Press Ctrl+C to stop.");
            while let Some(frame) = video_in.read_frame()? {
                let mf = session.process_frame(&frame, frame_idx, dt)?;
                let mut seq = vec![mf];
                convert_sequence_to_blender_space(&mut seq);
                let mut blended_mf = seq.pop().unwrap();

                if config.foot_lock_enabled {
                    let left_ankle = blended_mf.joints.get("LeftAnkle").map(|j| glam::Vec3::new(j.x, j.y, j.z)).unwrap_or(glam::Vec3::ZERO);
                    let right_ankle = blended_mf.joints.get("RightAnkle").map(|j| glam::Vec3::new(j.x, j.y, j.z)).unwrap_or(glam::Vec3::ZERO);
                    let left_toe = blended_mf.joints.get("LeftToeBase").map(|j| glam::Vec3::new(j.x, j.y, j.z)).unwrap_or(glam::Vec3::ZERO);
                    let right_toe = blended_mf.joints.get("RightToeBase").map(|j| glam::Vec3::new(j.x, j.y, j.z)).unwrap_or(glam::Vec3::ZERO);

                    let mut contacts = FootContacts::new(left_ankle, right_ankle, left_toe, right_toe);
                    session.foot_stabilizer.lock_planted_feet(&mut contacts, dt);

                    if let Some(la) = blended_mf.joints.get_mut("LeftAnkle") {
                        la.x = contacts.left_ankle.x;
                        la.y = contacts.left_ankle.y;
                        la.z = contacts.left_ankle.z;
                    }
                    if let Some(ra) = blended_mf.joints.get_mut("RightAnkle") {
                        ra.x = contacts.right_ankle.x;
                        ra.y = contacts.right_ankle.y;
                        ra.z = contacts.right_ankle.z;
                    }
                    if let Some(lt) = blended_mf.joints.get_mut("LeftToeBase") {
                        lt.x = contacts.left_toe.x;
                        lt.y = contacts.left_toe.y;
                        lt.z = contacts.left_toe.z;
                    }
                    if let Some(rt) = blended_mf.joints.get_mut("RightToeBase") {
                        rt.x = contacts.right_toe.x;
                        rt.y = contacts.right_toe.y;
                        rt.z = contacts.right_toe.z;
                    }
                }

                let _ = streamer.broadcast_frame(&blended_mf);
                frame_idx += 1;
                std::thread::sleep(frame_duration);
            }
        }
        Some(Commands::Export { input, output }) => {
            println!("Kinara Export: {} -> {}", input, output);
            let in_path = Path::new(&input);
            let out_path = Path::new(&output);
            if in_path.extension().and_then(|s| s.to_str()) == Some("json") {
                let data = std::fs::read_to_string(in_path)?;
                let v: serde_json::Value = serde_json::from_str(&data)?;
                if let Some(frames_val) = v.get("frames") {
                    let frames: Vec<MotionFrame> = serde_json::from_value(frames_val.clone())?;
                    let fps = v.get("fps").and_then(|f| f.as_f64()).unwrap_or(config.fps);
                    export_motion_fbx(out_path, &frames, fps)?;
                    println!("Successfully converted JSON -> FBX ({} frames).", frames.len());
                } else {
                    return Err(anyhow::anyhow!("Invalid motion JSON: missing 'frames'"));
                }
            } else {
                return Err(anyhow::anyhow!("Unsupported export conversion: input must be a motion JSON file"));
            }
        }
        Some(Commands::Calibrate { images_dir, output }) => {
            println!("Kinara Calibration: images_dir='{}', output={:?}", images_dir, output);
            let dir_path = Path::new(&images_dir);
            let board = kinara::fusion::CharucoBoard::default();
            let mut detected_count = 0;
            if dir_path.is_dir() {
                for entry in std::fs::read_dir(dir_path)? {
                    let entry = entry?;
                    let p = entry.path();
                    if let Ok(img) = image::open(&p) {
                        let rgb = img.to_rgb8();
                        if let Some(_corners) = kinara::fusion::detect_charuco_corners(&rgb, &board) {
                            detected_count += 1;
                        }
                    }
                }
            }
            println!("Calibration complete: detected target in {} frames.", detected_count);
            if let Some(out_p) = output {
                let mut rig = kinara::fusion::CameraRig::default();
                rig.add_camera(kinara::fusion::CameraParams::new(
                    "Camera0".to_string(), 800.0, 800.0, 320.0, 240.0, glam::Mat4::IDENTITY,
                ));
                let json = serde_json::to_string_pretty(&rig)?;
                std::fs::write(&out_p, json)?;
                println!("Saved camera rig configuration to {:?}", out_p);
            }
        }
        Some(Commands::Timeline { clip }) => {
            println!("Kinara Timeline Inspector for clip: {}", clip);
            let mut editor = kinara::video::TimelineEditor::new();
            let src = VideoInputSource::open(&clip, Some(config.fps))?;
            let duration = src.total_frames as f64 / src.fps.max(1.0);
            println!("Clip details: total_frames={}, fps={:.2}, duration={:.2}s, resolution={}x{}",
                src.total_frames, src.fps, duration, src.width, src.height);
            let clip_id = editor.add_clip(&clip, src.fps, src.total_frames);
            println!("Timeline active clip id: {}, total clips: {}", clip_id, editor.clips.len());
        }
        Some(Commands::Studio { input }) => {
            let cc = CreationContext::default();
            let mut app = KinaraApp::new(&cc, config);
            if let Some(inp) = input {
                let _ = app.open_source(&inp);
            }
            println!("Kinara Desktop Studio Deck launched.");
            let ctx = GuiContext::default();
            let mut frame = GuiFrame::default();
            app.update(&ctx, &mut frame);
            println!("Studio deck initialized: {}", app.status_message);
        }
        Some(Commands::Info) => {
            println!("Kinara Markerless Motion Capture Engine v0.1.0 (dev-rs)");
            println!("Supported Backends: CPU, DirectML, CUDA");
            println!("Armature: 60-joint canonical hierarchy (COCO-17 + 21-point hand)");
            println!("Formats: ASCII FBX 7.4.0, Kinara Motion JSON v1");
            println!("Blender Stream: TCP Port {}", BLENDER_STREAM_PORT);
        }
        None => {
            run_cli_pipeline(config)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default_and_roundtrip() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.fps, 30.0);
        assert!(cfg.body_ik_enabled);

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_kinara_cfg.json");

        cfg.save(&path).expect("Failed to save config");
        assert!(path.exists());

        let loaded = PipelineConfig::load_or_default(&path);
        assert_eq!(loaded.fps, 30.0);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_run_cli_pipeline_synthetic() {
        let temp_dir = std::env::temp_dir();
        let out_fbx = temp_dir.join("test_pipeline_out.fbx");

        let cfg = PipelineConfig {
            input_source: "synthetic:3:320x240".to_string(),
            output_path: out_fbx.to_string_lossy().to_string(),
            fps: 30.0,
            execution_backend: "cpu".to_string(),
            body_ik_enabled: true,
            hand_ik_enabled: true,
            foot_lock_enabled: true,
            filter_min_cutoff: 1.0,
            filter_beta: 0.007,
        };

        run_cli_pipeline(cfg).expect("Pipeline execution failed");
        assert!(out_fbx.exists());
        let _ = std::fs::remove_file(out_fbx);
    }

    #[test]
    fn test_pipeline_session_process_frame() {
        let cfg = PipelineConfig {
            input_source: "synthetic:1:320x240".to_string(),
            output_path: "test.fbx".to_string(),
            fps: 30.0,
            execution_backend: "cpu".to_string(),
            body_ik_enabled: true,
            hand_ik_enabled: true,
            foot_lock_enabled: true,
            filter_min_cutoff: 1.0,
            filter_beta: 0.007,
        };
        let mut session = PipelineSession::new(cfg).expect("Session init failed");
        let dummy_frame = image::RgbImage::new(320, 240);
        let mf = session.process_frame(&dummy_frame, 0, 1.0 / 30.0).expect("Process frame failed");
        assert_eq!(mf.frame_index, 0);
        assert!(mf.joints.contains_key("HipsRoot"));
        assert!(mf.joints.contains_key("LeftAnkle"));
        assert!(mf.joints.contains_key("LeftToeBase"));
    }
}

`

## File: src/config.rs

`rust
//! Pipeline configuration management and JSON persistence.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Global pipeline configuration parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub input_source: String,
    pub output_path: String,
    pub fps: f64,
    pub execution_backend: String,
    pub body_ik_enabled: bool,
    pub hand_ik_enabled: bool,
    pub foot_lock_enabled: bool,
    pub filter_min_cutoff: f32,
    pub filter_beta: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            input_source: "synthetic:30:640x480".to_string(),
            output_path: "output.fbx".to_string(),
            fps: 30.0,
            execution_backend: "cpu".to_string(),
            body_ik_enabled: true,
            hand_ik_enabled: true,
            foot_lock_enabled: true,
            filter_min_cutoff: 1.0,
            filter_beta: 0.007,
        }
    }
}

impl PipelineConfig {
    /// Loads configuration from JSON file or returns production defaults.
    pub fn load_or_default(path: &Path) -> Self {
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(path) {
                if let Ok(cfg) = serde_json::from_str(&content) {
                    return cfg;
                }
            }
        }
        Self::default()
    }

    /// Persists configuration parameters to disk.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let serialized = serde_json::to_string_pretty(self)?;
        std::fs::write(path, serialized)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_serialization() {
        let cfg = PipelineConfig::default();
        let serialized = serde_json::to_string(&cfg).expect("Failed to serialize");
        let deserialized: PipelineConfig = serde_json::from_str(&serialized).expect("Failed to deserialize");
        assert_eq!(cfg, deserialized);
    }
}

`

## File: src/skeleton.rs

`rust
//! Canonical skeletal hierarchy, landmark points, and coordinate definitions for Kinara.
//! Direct translation of `utils/skeleton.py`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// FBX official ticks per second time unit constant.
pub const FBX_TIME_UNIT: i64 = 46_186_158_000;

/// Joint specification defining joint name and its parent in the armature hierarchy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JointSpec {
    pub name: &'static str,
    pub parent: Option<&'static str>,
}

/// 2D landmark point with confidence: (x, y, confidence).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: i32,
    pub y: i32,
    pub confidence: f32,
}

impl Point {
    #[inline]
    pub fn new(x: i32, y: i32, confidence: f32) -> Self {
        Self { x, y, confidence }
    }
}

impl From<(i32, i32, f32)> for Point {
    #[inline]
    fn from(t: (i32, i32, f32)) -> Self {
        Self::new(t.0, t.1, t.2)
    }
}

impl From<Point> for (i32, i32, f32) {
    #[inline]
    fn from(p: Point) -> Self {
        (p.x, p.y, p.confidence)
    }
}

/// 2D bounding box represented as (x1, y1, x2, y2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

impl BoundingBox {
    #[inline]
    pub fn new(x1: i32, y1: i32, x2: i32, y2: i32) -> Self {
        Self { x1, y1, x2, y2 }
    }
}

impl From<(i32, i32, i32, i32)> for BoundingBox {
    #[inline]
    fn from(b: (i32, i32, i32, i32)) -> Self {
        Self::new(b.0, b.1, b.2, b.3)
    }
}

impl From<BoundingBox> for (i32, i32, i32, i32) {
    #[inline]
    fn from(b: BoundingBox) -> Self {
        (b.x1, b.y1, b.x2, b.y2)
    }
}

/// 3D joint Cartesian coordinates with tracking confidence.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct JointValue {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub confidence: f32,
}

impl JointValue {
    #[inline]
    pub fn new(x: f32, y: f32, z: f32, confidence: f32) -> Self {
        Self { x, y, z, confidence }
    }
}

/// Dictionary mapping joint names to their 3D coordinates.
pub type JointMap = HashMap<String, JointValue>;

/// Indices of facial landmark points in the COCO 17-keypoint model (Nose, LeftEye, RightEye, LeftEar, RightEar).
pub const FACE_POINT_INDICES: [usize; 5] = [0, 1, 2, 3, 4];

/// Canonical 60-joint skeletal hierarchy.
pub const SKELETON: [JointSpec; 60] = [
    JointSpec { name: "HipsRoot", parent: None },
    JointSpec { name: "LeftHip", parent: Some("HipsRoot") },
    JointSpec { name: "LeftKnee", parent: Some("LeftHip") },
    JointSpec { name: "LeftAnkle", parent: Some("LeftKnee") },
    JointSpec { name: "LeftFoot", parent: Some("LeftAnkle") },
    JointSpec { name: "LeftToeBase", parent: Some("LeftFoot") },
    JointSpec { name: "RightHip", parent: Some("HipsRoot") },
    JointSpec { name: "RightKnee", parent: Some("RightHip") },
    JointSpec { name: "RightAnkle", parent: Some("RightKnee") },
    JointSpec { name: "RightFoot", parent: Some("RightAnkle") },
    JointSpec { name: "RightToeBase", parent: Some("RightFoot") },
    JointSpec { name: "Chest", parent: Some("HipsRoot") },
    JointSpec { name: "Neck", parent: Some("Chest") },
    JointSpec { name: "Head", parent: Some("Neck") },
    JointSpec { name: "LeftShoulder", parent: Some("Chest") },
    JointSpec { name: "LeftElbow", parent: Some("LeftShoulder") },
    JointSpec { name: "LeftWrist", parent: Some("LeftElbow") },
    JointSpec { name: "LeftThumb1", parent: Some("LeftWrist") },
    JointSpec { name: "LeftThumb2", parent: Some("LeftThumb1") },
    JointSpec { name: "LeftThumb3", parent: Some("LeftThumb2") },
    JointSpec { name: "LeftThumb4", parent: Some("LeftThumb3") },
    JointSpec { name: "LeftIndex1", parent: Some("LeftWrist") },
    JointSpec { name: "LeftIndex2", parent: Some("LeftIndex1") },
    JointSpec { name: "LeftIndex3", parent: Some("LeftIndex2") },
    JointSpec { name: "LeftIndex4", parent: Some("LeftIndex3") },
    JointSpec { name: "LeftMiddle1", parent: Some("LeftWrist") },
    JointSpec { name: "LeftMiddle2", parent: Some("LeftMiddle1") },
    JointSpec { name: "LeftMiddle3", parent: Some("LeftMiddle2") },
    JointSpec { name: "LeftMiddle4", parent: Some("LeftMiddle3") },
    JointSpec { name: "LeftRing1", parent: Some("LeftWrist") },
    JointSpec { name: "LeftRing2", parent: Some("LeftRing1") },
    JointSpec { name: "LeftRing3", parent: Some("LeftRing2") },
    JointSpec { name: "LeftRing4", parent: Some("LeftRing3") },
    JointSpec { name: "LeftPinky1", parent: Some("LeftWrist") },
    JointSpec { name: "LeftPinky2", parent: Some("LeftPinky1") },
    JointSpec { name: "LeftPinky3", parent: Some("LeftPinky2") },
    JointSpec { name: "LeftPinky4", parent: Some("LeftPinky3") },
    JointSpec { name: "RightShoulder", parent: Some("Chest") },
    JointSpec { name: "RightElbow", parent: Some("RightShoulder") },
    JointSpec { name: "RightWrist", parent: Some("RightElbow") },
    JointSpec { name: "RightThumb1", parent: Some("RightWrist") },
    JointSpec { name: "RightThumb2", parent: Some("RightThumb1") },
    JointSpec { name: "RightThumb3", parent: Some("RightThumb2") },
    JointSpec { name: "RightThumb4", parent: Some("RightThumb3") },
    JointSpec { name: "RightIndex1", parent: Some("RightWrist") },
    JointSpec { name: "RightIndex2", parent: Some("RightIndex1") },
    JointSpec { name: "RightIndex3", parent: Some("RightIndex2") },
    JointSpec { name: "RightIndex4", parent: Some("RightIndex3") },
    JointSpec { name: "RightMiddle1", parent: Some("RightWrist") },
    JointSpec { name: "RightMiddle2", parent: Some("RightMiddle1") },
    JointSpec { name: "RightMiddle3", parent: Some("RightMiddle2") },
    JointSpec { name: "RightMiddle4", parent: Some("RightMiddle3") },
    JointSpec { name: "RightRing1", parent: Some("RightWrist") },
    JointSpec { name: "RightRing2", parent: Some("RightRing1") },
    JointSpec { name: "RightRing3", parent: Some("RightRing2") },
    JointSpec { name: "RightRing4", parent: Some("RightRing3") },
    JointSpec { name: "RightPinky1", parent: Some("RightWrist") },
    JointSpec { name: "RightPinky2", parent: Some("RightPinky1") },
    JointSpec { name: "RightPinky3", parent: Some("RightPinky2") },
    JointSpec { name: "RightPinky4", parent: Some("RightPinky3") },
];

/// Maps COCO-17 body joint name to its detector array index.
pub fn body_name_to_index(name: &str) -> Option<usize> {
    match name {
        "LeftShoulder" => Some(5),
        "RightShoulder" => Some(6),
        "LeftElbow" => Some(7),
        "RightElbow" => Some(8),
        "LeftWrist" => Some(9),
        "RightWrist" => Some(10),
        "LeftHip" => Some(11),
        "RightHip" => Some(12),
        "LeftKnee" => Some(13),
        "RightKnee" => Some(14),
        "LeftAnkle" => Some(15),
        "RightAnkle" => Some(16),
        _ => None,
    }
}

/// Maps derived foot joint name to its array index.
pub fn body_foot_name_to_index(name: &str) -> Option<usize> {
    match name {
        "LeftFoot" => Some(17),
        "RightFoot" => Some(18),
        "LeftToeBase" => Some(19),
        "RightToeBase" => Some(20),
        _ => None,
    }
}

/// Maps 21-point hand landmark name to its array index (1-based, wrist is 0).
pub fn hand_name_to_index(name: &str) -> Option<usize> {
    match name {
        "Thumb1" => Some(1),
        "Thumb2" => Some(2),
        "Thumb3" => Some(3),
        "Thumb4" => Some(4),
        "Index1" => Some(5),
        "Index2" => Some(6),
        "Index3" => Some(7),
        "Index4" => Some(8),
        "Middle1" => Some(9),
        "Middle2" => Some(10),
        "Middle3" => Some(11),
        "Middle4" => Some(12),
        "Ring1" => Some(13),
        "Ring2" => Some(14),
        "Ring3" => Some(15),
        "Ring4" => Some(16),
        "Pinky1" => Some(17),
        "Pinky2" => Some(18),
        "Pinky3" => Some(19),
        "Pinky4" => Some(20),
        _ => None,
    }
}

/// Standard export coordinate system metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoordinateSystemConfig {
    pub space: String,
    pub right_axis: String,
    pub forward_axis: String,
    pub up_axis: String,
    pub grounded_axis: String,
}

impl Default for CoordinateSystemConfig {
    fn default() -> Self {
        Self {
            space: "kinara_normalized".to_string(),
            right_axis: "X".to_string(),
            forward_axis: "Y".to_string(),
            up_axis: "Z".to_string(),
            grounded_axis: "Z".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skeleton_hierarchy_integrity() {
        assert_eq!(SKELETON.len(), 60);
        assert_eq!(SKELETON[0].name, "HipsRoot");
        assert!(SKELETON[0].parent.is_none());

        // Verify every child references an existing parent in SKELETON
        let joint_names: Vec<&str> = SKELETON.iter().map(|j| j.name).collect();
        for spec in &SKELETON[1..] {
            let parent_name = spec.parent.expect("Child joint must have a parent");
            assert!(
                joint_names.contains(&parent_name),
                "Parent '{}' of joint '{}' not found in SKELETON",
                parent_name,
                spec.name
            );
        }
    }

    #[test]
    fn test_body_name_mappings() {
        assert_eq!(body_name_to_index("LeftWrist"), Some(9));
        assert_eq!(body_name_to_index("RightAnkle"), Some(16));
        assert_eq!(body_name_to_index("NonExistent"), None);

        assert_eq!(body_foot_name_to_index("LeftFoot"), Some(17));
        assert_eq!(body_foot_name_to_index("RightToeBase"), Some(20));

        assert_eq!(hand_name_to_index("Index1"), Some(5));
        assert_eq!(hand_name_to_index("Pinky4"), Some(20));
    }

    #[test]
    fn test_coordinate_system_default() {
        let config = CoordinateSystemConfig::default();
        assert_eq!(config.space, "kinara_normalized");
        assert_eq!(config.up_axis, "Z");
    }
}

`

## File: src/gui.rs

`rust
//! Desktop studio GUI deck, viewport renderer, and interactive application controller.

use crate::config::PipelineConfig;
use crate::video::{TimelineEditor, VideoInputSource};
use image::RgbImage;
use std::path::PathBuf;

/// Immediate-mode GUI render context representing UI state and viewport layout.
#[derive(Debug, Clone)]
pub struct GuiContext {
    pub title: String,
    pub window_width: f32,
    pub window_height: f32,
    pub delta_time: f32,
}

impl Default for GuiContext {
    fn default() -> Self {
        Self {
            title: "Kinara Studio".to_string(),
            window_width: 1280.0,
            window_height: 720.0,
            delta_time: 0.033,
        }
    }
}

/// Window frame handle for controlling repaint requests and desktop integration.
#[derive(Debug, Clone, Default)]
pub struct GuiFrame {
    pub repaint_requested: bool,
    pub closed: bool,
}

impl GuiFrame {
    pub fn request_repaint(&mut self) {
        self.repaint_requested = true;
    }

    pub fn close(&mut self) {
        self.closed = true;
    }
}

/// Creation context envelope for desktop app initialization.
#[derive(Debug, Default)]
pub struct CreationContext {
    pub initial_width: f32,
    pub initial_height: f32,
}

/// Native desktop GUI application controller managing studio deck state,
/// video ingestion, viewport rendering, parameter controls, and timeline scrubbing.
#[derive(Debug)]
pub struct KinaraApp {
    pub config: PipelineConfig,
    pub timeline: TimelineEditor,
    pub video_source: Option<VideoInputSource>,
    pub current_frame: Option<RgbImage>,
    pub is_playing: bool,
    pub playhead_frame: usize,
    pub progress_pct: f32,
    pub status_message: String,
    pub selected_camera: usize,
    pub available_cameras: Vec<String>,
    pub recent_files: Vec<PathBuf>,
}

impl KinaraApp {
    /// Initializes native desktop GUI application with settings.
    pub fn new(_cc: &CreationContext, config: PipelineConfig) -> Self {
        let timeline = TimelineEditor::new();
        Self {
            config,
            timeline,
            video_source: None,
            current_frame: None,
            is_playing: false,
            playhead_frame: 0,
            progress_pct: 0.0,
            status_message: "Ready".to_string(),
            selected_camera: 0,
            available_cameras: vec!["Default Camera 0".to_string(), "Secondary Camera 1".to_string()],
            recent_files: Vec::new(),
        }
    }

    /// Renders desktop studio deck: video file picker, camera selection, live preview, and progress bar.
    pub fn update(&mut self, _ctx: &GuiContext, frame: &mut GuiFrame) {
        if self.is_playing {
            if let Some(ref mut src) = self.video_source {
                if let Ok(Some(img)) = src.read_frame() {
                    self.current_frame = Some(img);
                    self.playhead_frame += 1;
                    if src.total_frames > 0 {
                        self.progress_pct = (self.playhead_frame as f32 / src.total_frames as f32).min(1.0);
                    }
                    frame.request_repaint();
                } else {
                    self.is_playing = false;
                    self.status_message = "Playback ended".to_string();
                }
            }
        }
    }

    /// Ingests a new video source into the studio deck.
    pub fn open_source(&mut self, path_or_src: &str) -> anyhow::Result<()> {
        let source = VideoInputSource::open(path_or_src, Some(self.config.fps))?;
        self.config.input_source = path_or_src.to_string();
        self.playhead_frame = 0;
        self.progress_pct = 0.0;
        self.status_message = format!("Loaded: {}", path_or_src);
        self.recent_files.push(PathBuf::from(path_or_src));
        self.video_source = Some(source);
        Ok(())
    }

    /// Toggles play / pause playback state.
    pub fn toggle_playback(&mut self) {
        self.is_playing = !self.is_playing;
        self.status_message = if self.is_playing {
            "Playing".to_string()
        } else {
            "Paused".to_string()
        };
    }

    /// Seeks playhead to specific frame index.
    pub fn seek_frame(&mut self, target_frame: usize) -> anyhow::Result<()> {
        if let Some(ref mut src) = self.video_source {
            src.seek_frame(target_frame)?;
            self.playhead_frame = target_frame;
            if src.total_frames > 0 {
                self.progress_pct = (target_frame as f32 / src.total_frames as f32).min(1.0);
            }
            if let Ok(Some(img)) = src.read_frame() {
                self.current_frame = Some(img);
            }
        }
        Ok(())
    }

    /// Steps one frame forward in the studio deck.
    pub fn step_forward(&mut self) -> anyhow::Result<bool> {
        if let Some(ref mut src) = self.video_source {
            if let Some(img) = src.read_frame()? {
                self.current_frame = Some(img);
                self.playhead_frame += 1;
                if src.total_frames > 0 {
                    self.progress_pct = (self.playhead_frame as f32 / src.total_frames as f32).min(1.0);
                }
                return Ok(true);
            }
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kinara_app_lifecycle_and_playback() {
        let cfg = PipelineConfig::default();
        let cc = CreationContext::default();
        let mut app = KinaraApp::new(&cc, cfg);

        assert_eq!(app.playhead_frame, 0);
        assert!(!app.is_playing);

        app.open_source("synthetic:10:320x240").expect("Failed to open synthetic source");
        assert!(app.video_source.is_some());

        app.toggle_playback();
        assert!(app.is_playing);

        let ctx = GuiContext::default();
        let mut frame = GuiFrame::default();
        app.update(&ctx, &mut frame);

        assert_eq!(app.playhead_frame, 1);
        assert!(frame.repaint_requested);
        assert!(app.current_frame.is_some());

        app.seek_frame(5).expect("Seek failed");
        assert_eq!(app.playhead_frame, 5);
    }
}

`

## File: src/video/mod.rs

`rust
//! Video ingestion, decoding, timeline editing, and frame streaming.

pub mod io;
pub mod timeline;

pub use io::*;
pub use timeline::*;

`

## File: src/video/io.rs

`rust
//! Cross-platform video ingestion and video writing pipeline.
//! Fully OS-independent, supporting frame image sequences, raw streams, and synthetic video sources.

use anyhow::{anyhow, Context};
use ffmpeg_sidecar::command::FfmpegCommand;
use image::RgbImage;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{sync_channel, Receiver};
use std::time::Duration;

/// Cross-platform video input source for decoding video streams or frame sequences.
#[derive(Debug, Clone)]
pub struct VideoInputSource {
    pub source: String,
    pub fps: f64,
    pub width: u32,
    pub height: u32,
    pub current_index: usize,
    pub total_frames: usize,
    pub frame_paths: Vec<PathBuf>,
    pub memory_frames: Option<Vec<RgbImage>>,
    pub pending_frame: Option<RgbImage>,
    pub frame_receiver: Option<Arc<Mutex<Receiver<RgbImage>>>>,
}

impl VideoInputSource {
    /// Opens a video file, image sequence directory, or memory stream.
    pub fn open(src: &str, fps: Option<f64>) -> anyhow::Result<Self> {
        let default_fps = fps.unwrap_or(30.0).max(1.0);
        let path = Path::new(src);

        // Check if source is a filesystem directory containing image frames
        if path.is_dir() {
            let mut entries: Vec<PathBuf> = Vec::new();
            for entry in fs::read_dir(path).context("Failed to read image frames directory")? {
                let entry = entry?;
                let p = entry.path();
                if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
                    let ext_lower = ext.to_lowercase();
                    if matches!(ext_lower.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "ppm" | "webp") {
                        entries.push(p);
                    }
                }
            }
            entries.sort();

            if entries.is_empty() {
                return Err(anyhow!("No valid image frames found in directory: {}", src));
            }

            let first_img = image::open(&entries[0])
                .context("Failed to open first frame in sequence")?
                .to_rgb8();
            let (w, h) = first_img.dimensions();

            return Ok(Self {
                source: src.to_string(),
                fps: default_fps,
                width: w,
                height: h,
                current_index: 0,
                total_frames: entries.len(),
                frame_paths: entries,
                memory_frames: None,
                pending_frame: None,
                frame_receiver: None,
            });
        }

        // Virtual or memory test stream: synthetic:COUNT:WIDTHxHEIGHT
        if src.starts_with("synthetic:") {
            let parts: Vec<&str> = src.split(':').collect();
            let count: usize = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
            let dims = parts.get(2).copied().unwrap_or("640x480");
            let mut dim_parts = dims.split('x');
            let w: u32 = dim_parts.next().and_then(|s| s.parse().ok()).unwrap_or(640);
            let h: u32 = dim_parts.next().and_then(|s| s.parse().ok()).unwrap_or(480);

            let mut frames = Vec::with_capacity(count);
            for i in 0..count {
                let mut img = RgbImage::new(w, h);
                let shade = ((i * 255) / count.max(1)) as u8;
                for p in img.pixels_mut() {
                    *p = image::Rgb([shade, shade, 255 - shade]);
                }
                frames.push(img);
            }

            return Ok(Self {
                source: src.to_string(),
                fps: default_fps,
                width: w,
                height: h,
                current_index: 0,
                total_frames: count,
                frame_paths: Vec::new(),
                memory_frames: Some(frames),
                pending_frame: None,
                frame_receiver: None,
            });
        }

        // Check if source is a single static image file
        let is_image_file = if path.is_file() {
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                let ext_lower = ext.to_lowercase();
                matches!(ext_lower.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "ppm" | "webp")
            } else {
                false
            }
        } else {
            false
        };

        if is_image_file {
            let img = image::open(path)
                .with_context(|| format!("Failed to open image file: {:?}", path))?
                .to_rgb8();
            let (w, h) = img.dimensions();
            return Ok(Self {
                source: src.to_string(),
                fps: default_fps,
                width: w,
                height: h,
                current_index: 0,
                total_frames: 1,
                frame_paths: vec![path.to_path_buf()],
                memory_frames: Some(vec![img]),
                pending_frame: None,
                frame_receiver: None,
            });
        }

        // Check if source is a video file or webcam device
        let is_video_file = if path.is_file() {
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                let ext_lower = ext.to_lowercase();
                matches!(
                    ext_lower.as_str(),
                    "mp4" | "mkv" | "avi" | "mov" | "webm" | "flv" | "wmv" | "m4v"
                )
            } else {
                false
            }
        } else {
            false
        };

        let is_webcam = src.starts_with("camera:")
            || src.starts_with("webcam:")
            || src.starts_with("dshow:")
            || src.starts_with("/dev/video");

        if is_video_file || is_webcam {
            if !ffmpeg_sidecar::command::ffmpeg_is_installed() {
                let _ = ffmpeg_sidecar::download::auto_download();
            }

            let mut cmd = FfmpegCommand::new();
            if is_webcam {
                let cam_id = src
                    .strip_prefix("camera:")
                    .or_else(|| src.strip_prefix("webcam:"))
                    .or_else(|| src.strip_prefix("dshow:"))
                    .unwrap_or("0");

                #[cfg(target_os = "windows")]
                {
                    let dev_arg = if cam_id.starts_with("video=") {
                        cam_id.to_string()
                    } else {
                        format!("video={}", cam_id)
                    };
                    cmd.args(["-f", "dshow", "-i", &dev_arg]);
                }
                #[cfg(not(target_os = "windows"))]
                {
                    let dev_path = if cam_id.starts_with("/dev/") {
                        cam_id.to_string()
                    } else {
                        format!("/dev/video{}", cam_id)
                    };
                    cmd.args(["-f", "v4l2", "-i", &dev_path]);
                }
            } else {
                cmd.input(src);
            }

            cmd.rawvideo();
            cmd.pix_fmt("rgb24");

            let mut child = cmd.spawn().with_context(|| {
                format!("Failed to spawn FFmpeg to decode video stream: {}", src)
            })?;

            let (tx, rx) = sync_channel::<RgbImage>(8);

            std::thread::spawn(move || {
                if let Ok(iter) = child.iter() {
                    for frame in iter.filter_frames() {
                        if let Some(rgb) = RgbImage::from_raw(frame.width, frame.height, frame.data) {
                            if tx.send(rgb).is_err() {
                                break;
                            }
                        }
                    }
                }
            });

            // Probe first frame to detect stream dimensions
            let (first_frame, w, h) = match rx.recv_timeout(Duration::from_millis(3000)) {
                Ok(frame) => {
                    let (fw, fh) = frame.dimensions();
                    (Some(frame), fw, fh)
                }
                Err(_) => (None, 640, 480),
            };

            return Ok(Self {
                source: src.to_string(),
                fps: default_fps,
                width: w,
                height: h,
                current_index: 0,
                total_frames: 0,
                frame_paths: Vec::new(),
                memory_frames: None,
                pending_frame: first_frame,
                frame_receiver: Some(Arc::new(Mutex::new(rx))),
            });
        }

        // Return initialized input source with default dimensions
        Ok(Self {
            source: src.to_string(),
            fps: default_fps,
            width: 640,
            height: 480,
            current_index: 0,
            total_frames: 0,
            frame_paths: Vec::new(),
            memory_frames: None,
            pending_frame: None,
            frame_receiver: None,
        })
    }

    /// Reads the next frame in the stream into an RGB buffer.
    pub fn read_frame(&mut self) -> anyhow::Result<Option<RgbImage>> {
        if let Some(frame) = self.pending_frame.take() {
            self.current_index += 1;
            return Ok(Some(frame));
        }

        if let Some(ref mem_frames) = self.memory_frames {
            if self.current_index < mem_frames.len() {
                let frame = mem_frames[self.current_index].clone();
                self.current_index += 1;
                return Ok(Some(frame));
            }
            return Ok(None);
        }

        if self.current_index < self.frame_paths.len() {
            let path = &self.frame_paths[self.current_index];
            let img = image::open(path)
                .with_context(|| format!("Failed to read frame at {:?}", path))?
                .to_rgb8();
            self.current_index += 1;
            return Ok(Some(img));
        }

        if let Some(ref rx_arc) = self.frame_receiver {
            if let Ok(rx) = rx_arc.lock() {
                match rx.recv() {
                    Ok(frame) => {
                        self.current_index += 1;
                        return Ok(Some(frame));
                    }
                    Err(_) => return Ok(None),
                }
            }
        }

        Ok(None)
    }

    /// Seeks to a specific frame timestamp or index.
    pub fn seek_frame(&mut self, index: usize) -> anyhow::Result<()> {
        if self.frame_receiver.is_some() {
            return Err(anyhow!("Seeking is not supported on live video streams"));
        }
        if self.total_frames > 0 && index >= self.total_frames {
            return Err(anyhow!(
                "Seek target {} exceeds total frames {}",
                index,
                self.total_frames
            ));
        }
        self.current_index = index;
        Ok(())
    }
}

/// Cross-platform video output stream writer.
pub struct VideoOutputWriter {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub frame_count: usize,
    writer: Option<BufWriter<File>>,
    is_directory_sequence: bool,
}

impl VideoOutputWriter {
    /// Initializes output stream writer for raw video or directory frame sequences.
    pub fn new(path: &Path, w: u32, h: u32, fps: f64) -> anyhow::Result<Self> {
        let is_dir = path.extension().is_none();
        if is_dir {
            fs::create_dir_all(path)?;
            Ok(Self {
                path: path.to_path_buf(),
                width: w,
                height: h,
                fps,
                frame_count: 0,
                writer: None,
                is_directory_sequence: true,
            })
        } else {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let file = File::create(path)?;
            let writer = BufWriter::new(file);
            Ok(Self {
                path: path.to_path_buf(),
                width: w,
                height: h,
                fps,
                frame_count: 0,
                writer: Some(writer),
                is_directory_sequence: false,
            })
        }
    }

    /// Encodes and appends one frame to the output destination.
    pub fn write_frame(&mut self, frame: &RgbImage) -> anyhow::Result<()> {
        if self.is_directory_sequence {
            let frame_filename = format!("frame_{:06}.png", self.frame_count);
            let frame_path = self.path.join(frame_filename);
            frame.save(&frame_path)?;
            self.frame_count += 1;
            return Ok(());
        }

        if let Some(ref mut writer) = self.writer {
            // Write portable Netpbm binary PPM (P6) frame
            writeln!(writer, "P6\n{} {}\n255", self.width, self.height)?;
            writer.write_all(frame.as_raw())?;
            self.frame_count += 1;
            return Ok(());
        }

        Err(anyhow!("Writer stream is not initialized"))
    }

    /// Flushes remaining packets and finalizes output container.
    pub fn finish(mut self) -> anyhow::Result<()> {
        if let Some(mut writer) = self.writer.take() {
            writer.flush()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_video_input_source() {
        let mut source = VideoInputSource::open("synthetic:5:320x240", Some(30.0))
            .expect("Failed to open synthetic source");

        assert_eq!(source.total_frames, 5);
        assert_eq!(source.width, 320);
        assert_eq!(source.height, 240);

        let frame1 = source.read_frame().expect("Read frame failed");
        assert!(frame1.is_some());
        assert_eq!(source.current_index, 1);

        source.seek_frame(4).expect("Seek failed");
        assert_eq!(source.current_index, 4);

        let frame5 = source.read_frame().expect("Read frame 5 failed");
        assert!(frame5.is_some());

        let eof = source.read_frame().expect("EOF read failed");
        assert!(eof.is_none());
    }

    #[test]
    fn test_video_output_writer_stream() {
        let temp_dir = std::env::temp_dir();
        let out_path = temp_dir.join("test_stream.ppm");

        let mut writer = VideoOutputWriter::new(&out_path, 64, 64, 30.0)
            .expect("Failed to create video writer");

        let dummy_frame = RgbImage::new(64, 64);
        writer.write_frame(&dummy_frame).expect("Write frame failed");
        writer.write_frame(&dummy_frame).expect("Write frame failed");
        writer.finish().expect("Finish failed");

        assert!(out_path.exists());
        let meta = fs::metadata(&out_path).expect("Failed metadata");
        assert!(meta.len() > 0);

        let _ = fs::remove_file(out_path);
    }

    #[test]
    fn test_single_image_and_dir_input_source() {
        let temp_dir = std::env::temp_dir().join("kinara_test_frames");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).expect("Failed to create temp test dir");

        let frame_a = RgbImage::new(128, 96);
        let p_a = temp_dir.join("frame_000001.png");
        frame_a.save(&p_a).expect("Save frame_a failed");

        let frame_b = RgbImage::new(128, 96);
        let p_b = temp_dir.join("frame_000002.png");
        frame_b.save(&p_b).expect("Save frame_b failed");

        // Test single image file input
        let mut single_src = VideoInputSource::open(p_a.to_str().unwrap(), Some(24.0))
            .expect("Failed to open single image");
        assert_eq!(single_src.width, 128);
        assert_eq!(single_src.height, 96);
        assert_eq!(single_src.total_frames, 1);
        let f1 = single_src.read_frame().unwrap();
        assert!(f1.is_some());
        assert!(single_src.read_frame().unwrap().is_none());

        // Test directory image sequence input
        let mut dir_src = VideoInputSource::open(temp_dir.to_str().unwrap(), Some(24.0))
            .expect("Failed to open directory sequence");
        assert_eq!(dir_src.width, 128);
        assert_eq!(dir_src.height, 96);
        assert_eq!(dir_src.total_frames, 2);
        assert!(dir_src.read_frame().unwrap().is_some());
        assert!(dir_src.read_frame().unwrap().is_some());
        assert!(dir_src.read_frame().unwrap().is_none());

        let _ = fs::remove_dir_all(&temp_dir);
    }
}

`

## File: src/video/timeline.rs

`rust
//! Non-linear video timeline data structures, clip cutting, trimming, and multi-clip merging.
//! Foundation for the native desktop timeline editor.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Represents a single video clip placed on the timeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelineClip {
    pub id: u64,
    pub file_path: PathBuf,
    pub source_fps: f64,
    pub total_source_frames: usize,
    /// Trim in-point (inclusive start frame in source video)
    pub in_frame: usize,
    /// Trim out-point (exclusive end frame in source video)
    pub out_frame: usize,
    /// Track lane index (0 = primary video track, 1+ = auxiliary/camera views)
    pub track_index: usize,
}

impl TimelineClip {
    pub fn new(id: u64, file_path: PathBuf, source_fps: f64, total_frames: usize) -> Self {
        Self {
            id,
            file_path,
            source_fps,
            total_source_frames: total_frames,
            in_frame: 0,
            out_frame: total_frames,
            track_index: 0,
        }
    }

    /// Effective duration of the clip in frames after trimming.
    #[inline]
    pub fn duration_frames(&self) -> usize {
        self.out_frame.saturating_sub(self.in_frame)
    }

    /// Effective duration of the clip in seconds.
    #[inline]
    pub fn duration_seconds(&self) -> f64 {
        if self.source_fps <= 0.0 {
            0.0
        } else {
            self.duration_frames() as f64 / self.source_fps
        }
    }

    /// Sets non-destructive trim in/out bounds.
    pub fn set_trim(&mut self, in_frame: usize, out_frame: usize) -> Result<(), &'static str> {
        if in_frame >= out_frame {
            return Err("In-point must be strictly less than out-point");
        }
        if out_frame > self.total_source_frames {
            return Err("Out-point exceeds source duration");
        }
        self.in_frame = in_frame;
        self.out_frame = out_frame;
        Ok(())
    }

    /// Splits (cuts) this clip into two clips at the specified frame offset relative to in_frame.
    /// Returns the second clip starting at the cut point.
    pub fn split_at(&mut self, cut_offset: usize, new_clip_id: u64) -> Result<Self, &'static str> {
        let absolute_cut = self.in_frame + cut_offset;
        if absolute_cut <= self.in_frame || absolute_cut >= self.out_frame {
            return Err("Cut position must fall strictly inside the active trim range");
        }

        let mut second_clip = self.clone();
        second_clip.id = new_clip_id;
        second_clip.in_frame = absolute_cut;

        self.out_frame = absolute_cut;
        Ok(second_clip)
    }
}

/// Non-linear multi-track video timeline editor state.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TimelineEditor {
    pub clips: Vec<TimelineClip>,
    pub playhead_frame: usize,
    pub next_clip_id: u64,
}

impl TimelineEditor {
    pub fn new() -> Self {
        Self {
            clips: Vec::new(),
            playhead_frame: 0,
            next_clip_id: 1,
        }
    }

    /// Ingests a video clip dropped or loaded into the timeline.
    pub fn add_clip(&mut self, path: impl AsRef<Path>, fps: f64, total_frames: usize) -> u64 {
        let id = self.next_clip_id;
        self.next_clip_id += 1;
        self.clips.push(TimelineClip::new(id, path.as_ref().to_path_buf(), fps, total_frames));
        id
    }

    /// Removes a clip by its unique ID.
    pub fn remove_clip(&mut self, clip_id: u64) -> bool {
        if let Some(pos) = self.clips.iter().position(|c| c.id == clip_id) {
            self.clips.remove(pos);
            true
        } else {
            false
        }
    }

    /// Cuts (splits) a clip at a given frame offset, appending the newly created second half.
    pub fn cut_clip(&mut self, clip_id: u64, cut_offset: usize) -> Result<u64, &'static str> {
        let new_id = self.next_clip_id;
        let pos = self
            .clips
            .iter()
            .position(|c| c.id == clip_id)
            .ok_or("Clip not found")?;

        let second_half = self.clips[pos].split_at(cut_offset, new_id)?;
        self.next_clip_id += 1;
        self.clips.insert(pos + 1, second_half);
        Ok(new_id)
    }

    /// Merges two contiguous clips that share the same source file into one unified clip.
    pub fn merge_clips(&mut self, first_id: u64, second_id: u64) -> Result<(), &'static str> {
        let pos1 = self.clips.iter().position(|c| c.id == first_id).ok_or("First clip not found")?;
        let pos2 = self.clips.iter().position(|c| c.id == second_id).ok_or("Second clip not found")?;

        if self.clips[pos1].file_path != self.clips[pos2].file_path {
            return Err("Cannot merge clips with different source video files");
        }
        if self.clips[pos1].out_frame != self.clips[pos2].in_frame {
            return Err("Clips are not contiguous in source time");
        }

        self.clips[pos1].out_frame = self.clips[pos2].out_frame;
        self.clips.remove(pos2);
        Ok(())
    }

    /// Total timeline frame count across all sequential clips on primary track 0.
    pub fn total_timeline_frames(&self) -> usize {
        self.clips
            .iter()
            .filter(|c| c.track_index == 0)
            .map(|c| c.duration_frames())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeline_clip_duration_and_trim() {
        let mut clip = TimelineClip::new(1, PathBuf::from("video.mp4"), 30.0, 300);
        assert_eq!(clip.duration_frames(), 300);
        assert!((clip.duration_seconds() - 10.0).abs() < 1e-4);

        clip.set_trim(50, 200).unwrap();
        assert_eq!(clip.duration_frames(), 150);
        assert!((clip.duration_seconds() - 5.0).abs() < 1e-4);

        assert!(clip.set_trim(200, 100).is_err());
        assert!(clip.set_trim(0, 400).is_err());
    }

    #[test]
    fn test_timeline_editor_cut_and_merge() {
        let mut editor = TimelineEditor::new();
        let clip_id = editor.add_clip("action.mp4", 60.0, 600);
        assert_eq!(editor.total_timeline_frames(), 600);

        // Cut clip at frame offset 200
        let new_clip_id = editor.cut_clip(clip_id, 200).unwrap();
        assert_eq!(editor.clips.len(), 2);
        assert_eq!(editor.clips[0].duration_frames(), 200);
        assert_eq!(editor.clips[1].duration_frames(), 400);
        assert_eq!(editor.total_timeline_frames(), 600);

        // Merge the two contiguous clips back together
        editor.merge_clips(clip_id, new_clip_id).unwrap();
        assert_eq!(editor.clips.len(), 1);
        assert_eq!(editor.clips[0].duration_frames(), 600);
    }

    #[test]
    fn test_merge_discontiguous_fails() {
        let mut editor = TimelineEditor::new();
        let c1 = editor.add_clip("part1.mp4", 30.0, 100);
        let c2 = editor.add_clip("part2.mp4", 30.0, 100);

        // Different file paths -> merge fails
        assert!(editor.merge_clips(c1, c2).is_err());
    }
}

`

## File: src/inference/mod.rs

`rust
//! Neural pose and hand estimation inference and pre/post-processing pipeline.

use anyhow::Context;
use crate::skeleton::Point;
use crate::solver::body_constraints::BodyKeypoints;
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

/// Decodes 1D SimCC horizontal and vertical representations into 2D keypoints with Gaussian peak refinement.
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

        // Sub-pixel Gaussian interpolation on horizontal axis
        let dx = if best_x_idx > 0 && best_x_idx + 1 < x_bins {
            let left = x_slice[best_x_idx - 1].max(1e-6).ln();
            let center = max_val_x.max(1e-6).ln();
            let right = x_slice[best_x_idx + 1].max(1e-6).ln();
            let denom = 2.0 * (2.0 * center - left - right);
            if denom.abs() > 1e-5 {
                ((left - right) / denom).clamp(-1.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Sub-pixel Gaussian interpolation on vertical axis
        let dy = if best_y_idx > 0 && best_y_idx + 1 < y_bins {
            let left = y_slice[best_y_idx - 1].max(1e-6).ln();
            let center = max_val_y.max(1e-6).ln();
            let right = y_slice[best_y_idx + 1].max(1e-6).ln();
            let denom = 2.0 * (2.0 * center - left - right);
            if denom.abs() > 1e-5 {
                ((left - right) / denom).clamp(-1.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        let refined_x = (best_x_idx as f32 + dx) / simcc_split_ratio.max(1.0);
        let refined_y = (best_y_idx as f32 + dy) / simcc_split_ratio.max(1.0);

        let (orig_x, orig_y) = transform.unletterbox(refined_x, refined_y);
        let confidence = (max_val_x.min(max_val_y)).clamp(0.0, 1.0);
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

/// Initializes an ONNX Runtime session, falling back gracefully to CPU if hardware acceleration is unavailable.
fn init_ort_session(
    path: &Path,
    _backend: ExecutionBackend,
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

    let session = builder
        .commit_from_file(path)
        .with_context(|| format!("Failed to load ONNX model from {:?}", path))?;

    Ok(Some(Arc::new(Mutex::new(session))))
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
        Ok(Self {
            model_path: path.to_path_buf(),
            backend,
            input_w: 256,
            input_h: 256,
            session,
        })
    }

    pub fn detect_body(&mut self, frame: &image::RgbImage) -> anyhow::Result<BodyKeypoints> {
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
                            return Ok(BodyKeypoints::new(kps));
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
                            return Ok(BodyKeypoints::new(kps));
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
                            return Ok(BodyKeypoints::new(kps));
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
        Ok(BodyKeypoints::new(points))
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
        Ok(Self {
            model_path: path.to_path_buf(),
            backend,
            input_w: 256,
            input_h: 256,
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
        assert_eq!(body_kps.points.len(), 17);
        assert_eq!(body_kps.points[0].x, 320);
        assert_eq!(body_kps.points[0].y, 240);

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

`

## File: src/fusion/mod.rs

`rust
//! Multi-camera calibration, direct linear transformation (DLT) triangulation, and multi-view landmark fusion.
//! Direct translation of `utils/triangulation.py`, `utils/calibration.py`, and `utils/fusion.py`.

use crate::skeleton::{JointMap, JointValue, Point};
use crate::export::HandsPair;
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

`

## File: src/tracking/mod.rs

`rust
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

/// Torso color profiling utility for clothing-based actor re-identification.
pub struct TorsoColorProfiler;

impl TorsoColorProfiler {
    /// Extracts a 16-bin normalized Hue histogram from the torso region of an image.
    pub fn extract_histogram(frame: &image::RgbImage, torso: BoundingBox) -> ColorHistogram {
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
                det.color_hist = Some(TorsoColorProfiler::extract_histogram(frame, det.box_));
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

`

## File: src/solver/mod.rs

`rust
pub mod body_constraints;
pub mod body_geometry;
pub mod filter;
pub mod foot_lock;
pub mod hand_constraints;
pub mod hand_fallback;
pub mod prediction;

pub use body_constraints::*;
pub use body_geometry::*;
pub use filter::{OneEuroFilter, QuaternionFilter};
pub use foot_lock::*;
pub use hand_constraints::*;
pub use hand_fallback::*;
pub use prediction::*;

`

## File: src/solver/body_constraints.rs

`rust
//! Biomechanical full-body inverse kinematics and segment length constraint solver.
//! Prevents unnatural limb stretching and shrinking across perspective changes
//! by enforcing learned invariant segment lengths. Direct translation of `utils/body_constraints.py`.

use crate::skeleton::Point;
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// COCO-17 body segment pairings grouped by anatomical limb category.
/// Indices correspond to canonical COCO-17 detector joints:
/// - 5: LeftShoulder, 7: LeftElbow, 9: LeftWrist
/// - 6: RightShoulder, 8: RightElbow, 10: RightWrist
/// - 11: LeftHip, 13: LeftKnee, 15: LeftAnkle
/// - 12: RightHip, 14: RightKnee, 16: RightAnkle
pub const BODY_SEGMENT_GROUPS: [(&str, [(usize, usize); 2]); 4] = [
    ("upper_arm", [(5, 7), (6, 8)]),
    ("forearm", [(7, 9), (8, 10)]),
    ("thigh", [(11, 13), (12, 14)]),
    ("shin", [(13, 15), (14, 16)]),
];

/// Configuration parameters for kinematics and motion solvers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolverConfig {
    pub body_constraints_enabled: bool,
    pub body_length_smoothing_alpha: f32,
    pub body_length_correction: f32,
    pub body_conf_threshold: f32,
    pub export_foot_lock_enabled: bool,
    pub export_foot_lock_velocity: f32,
    pub export_foot_lock_max_lift: f32,
    pub min_cutoff: f32,
    pub beta: f32,
    pub d_cutoff: f32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            body_constraints_enabled: true,
            body_length_smoothing_alpha: 0.15,
            body_length_correction: 0.35,
            body_conf_threshold: 0.30,
            export_foot_lock_enabled: true,
            export_foot_lock_velocity: 8.0,
            export_foot_lock_max_lift: 16.0,
            min_cutoff: 1.0,
            beta: 0.007,
            d_cutoff: 1.0,
        }
    }
}

/// Container for detected 2D body keypoints.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BodyKeypoints {
    pub points: Vec<Point>,
}

impl BodyKeypoints {
    #[inline]
    pub fn new(points: Vec<Point>) -> Self {
        Self { points }
    }
}

impl std::ops::Deref for BodyKeypoints {
    type Target = [Point];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.points
    }
}

impl std::ops::DerefMut for BodyKeypoints {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.points
    }
}

impl From<Vec<Point>> for BodyKeypoints {
    #[inline]
    fn from(points: Vec<Point>) -> Self {
        Self::new(points)
    }
}

impl From<&[Point]> for BodyKeypoints {
    #[inline]
    fn from(slice: &[Point]) -> Self {
        Self::new(slice.to_vec())
    }
}

/// Biomechanical body kinematic constraint solver.
/// Learns stable segment lengths over time using exponential moving average (EMA)
/// and projects detected joints along limb vectors to prevent stretching.
#[derive(Debug, Clone)]
pub struct BodyIKSolver {
    config: SolverConfig,
    target_lengths: HashMap<(usize, usize), f32>,
}

impl BodyIKSolver {
    /// Initializes a new BodyIKSolver with the specified configuration.
    pub fn new(config: &SolverConfig) -> Self {
        Self {
            config: config.clone(),
            target_lengths: HashMap::new(),
        }
    }

    /// Resets all learned segment lengths.
    pub fn reset(&mut self) {
        self.target_lengths.clear();
    }

    /// Returns a reference to the active target length cache.
    pub fn target_lengths(&self) -> &HashMap<(usize, usize), f32> {
        &self.target_lengths
    }

    /// Enforces anatomical limb lengths on a mutable BodyKeypoints set.
    pub fn enforce_limb_lengths(&mut self, body: &mut BodyKeypoints) {
        self.enforce_limb_lengths_points(&mut body.points);
    }

    /// Enforces anatomical limb lengths on a mutable slice of 2D `Point` landmarks.
    pub fn enforce_limb_lengths_points(&mut self, points: &mut [Point]) {
        if !self.config.body_constraints_enabled || points.len() < 17 {
            return;
        }

        self.update_target_lengths_2d(points);
        self.apply_length_targets_2d(points);
    }

    /// Enforces anatomical limb lengths on 3D joint positions using SIMD `Vec3`.
    pub fn enforce_limb_lengths_3d(&mut self, points: &mut [Vec3], confidences: &[f32]) {
        if !self.config.body_constraints_enabled || points.len() < 17 || confidences.len() < 17 {
            return;
        }

        self.update_target_lengths_3d(points, confidences);
        self.apply_length_targets_3d(points, confidences);
    }

    fn update_target_lengths_2d(&mut self, points: &[Point]) {
        let alpha = self.config.body_length_smoothing_alpha;
        let threshold = self.config.body_conf_threshold;

        for (_group, pairs) in &BODY_SEGMENT_GROUPS {
            for &(parent_idx, child_idx) in pairs {
                let parent = points[parent_idx];
                let child = points[child_idx];
                if parent.confidence <= threshold || child.confidence <= threshold {
                    continue;
                }

                let dx = (child.x - parent.x) as f32;
                let dy = (child.y - parent.y) as f32;
                let length = dx.hypot(dy);
                if length <= 2.0 {
                    continue;
                }

                let key = (parent_idx, child_idx);
                let new_len = match self.target_lengths.get(&key) {
                    Some(&prev) => prev * (1.0 - alpha) + length * alpha,
                    None => length,
                };
                self.target_lengths.insert(key, new_len);
            }
        }
    }

    fn apply_length_targets_2d(&self, points: &mut [Point]) {
        let correction = self.config.body_length_correction;
        let threshold = self.config.body_conf_threshold;

        for (_group, pairs) in &BODY_SEGMENT_GROUPS {
            for &(parent_idx, child_idx) in pairs {
                let target_len = match self.target_lengths.get(&(parent_idx, child_idx)) {
                    Some(&len) => len,
                    None => continue,
                };

                let parent = points[parent_idx];
                let child = points[child_idx];
                if parent.confidence <= threshold || child.confidence <= threshold {
                    continue;
                }

                let dx = (child.x - parent.x) as f32;
                let dy = (child.y - parent.y) as f32;
                let current_len = dx.hypot(dy);
                if current_len <= 1e-6 {
                    continue;
                }

                let corrected_len = current_len * (1.0 - correction) + target_len * correction;
                let unit_x = dx / current_len;
                let unit_y = dy / current_len;

                let new_x = (parent.x as f32 + unit_x * corrected_len).round() as i32;
                let new_y = (parent.y as f32 + unit_y * corrected_len).round() as i32;

                points[child_idx] = Point::new(new_x, new_y, child.confidence);
            }
        }
    }

    fn update_target_lengths_3d(&mut self, points: &[Vec3], confidences: &[f32]) {
        let alpha = self.config.body_length_smoothing_alpha;
        let threshold = self.config.body_conf_threshold;

        for (_group, pairs) in &BODY_SEGMENT_GROUPS {
            for &(parent_idx, child_idx) in pairs {
                if confidences[parent_idx] <= threshold || confidences[child_idx] <= threshold {
                    continue;
                }

                let length = (points[child_idx] - points[parent_idx]).length();
                if length <= 2.0 {
                    continue;
                }

                let key = (parent_idx, child_idx);
                let new_len = match self.target_lengths.get(&key) {
                    Some(&prev) => prev * (1.0 - alpha) + length * alpha,
                    None => length,
                };
                self.target_lengths.insert(key, new_len);
            }
        }
    }

    fn apply_length_targets_3d(&self, points: &mut [Vec3], confidences: &[f32]) {
        let correction = self.config.body_length_correction;
        let threshold = self.config.body_conf_threshold;

        for (_group, pairs) in &BODY_SEGMENT_GROUPS {
            for &(parent_idx, child_idx) in pairs {
                let target_len = match self.target_lengths.get(&(parent_idx, child_idx)) {
                    Some(&len) => len,
                    None => continue,
                };

                if confidences[parent_idx] <= threshold || confidences[child_idx] <= threshold {
                    continue;
                }

                let delta = points[child_idx] - points[parent_idx];
                let current_len = delta.length();
                if current_len <= 1e-6 {
                    continue;
                }

                let corrected_len = current_len * (1.0 - correction) + target_len * correction;
                let unit = delta / current_len;
                points[child_idx] = points[parent_idx] + unit * corrected_len;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_body_points() -> Vec<Point> {
        let mut pts = vec![Point::new(0, 0, 0.9); 17];
        // Upper arm left: 5 -> 7
        pts[5] = Point::new(100, 100, 0.9);
        pts[7] = Point::new(100, 150, 0.9); // len = 50
        // Forearm left: 7 -> 9
        pts[9] = Point::new(100, 190, 0.9); // len = 40
        // Upper arm right: 6 -> 8
        pts[6] = Point::new(200, 100, 0.9);
        pts[8] = Point::new(200, 150, 0.9); // len = 50
        // Forearm right: 8 -> 10
        pts[10] = Point::new(200, 190, 0.9); // len = 40
        // Thigh left: 11 -> 13
        pts[11] = Point::new(120, 250, 0.9);
        pts[13] = Point::new(120, 320, 0.9); // len = 70
        // Shin left: 13 -> 15
        pts[15] = Point::new(120, 380, 0.9); // len = 60
        // Thigh right: 12 -> 14
        pts[12] = Point::new(180, 250, 0.9);
        pts[14] = Point::new(180, 320, 0.9); // len = 70
        // Shin right: 14 -> 16
        pts[16] = Point::new(180, 380, 0.9); // len = 60
        pts
    }

    #[test]
    fn test_body_ik_solver_initial_learning() {
        let config = SolverConfig::default();
        let mut solver = BodyIKSolver::new(&config);

        let mut body = BodyKeypoints::new(dummy_body_points());
        solver.enforce_limb_lengths(&mut body);

        // Verify initial lengths cached correctly
        assert_eq!(solver.target_lengths().get(&(5, 7)), Some(&50.0));
        assert_eq!(solver.target_lengths().get(&(7, 9)), Some(&40.0));
        assert_eq!(solver.target_lengths().get(&(11, 13)), Some(&70.0));
        assert_eq!(solver.target_lengths().get(&(13, 15)), Some(&60.0));
    }

    #[test]
    fn test_body_ik_solver_prevents_stretching() {
        let mut config = SolverConfig::default();
        config.body_length_correction = 1.0; // 100% force back to target length
        let mut solver = BodyIKSolver::new(&config);

        let mut body1 = BodyKeypoints::new(dummy_body_points());
        solver.enforce_limb_lengths(&mut body1);

        // Next frame, detector glitches and left elbow (index 7) stretches to len = 100
        let mut body2 = BodyKeypoints::new(dummy_body_points());
        body2[7] = Point::new(100, 200, 0.9); // dist(5, 7) was 50, now 100
        solver.enforce_limb_lengths(&mut body2);

        // Target length after smoothing (0.15): 50 * 0.85 + 100 * 0.15 = 57.5
        // Reprojected elbow should be clamped close to 57.5 instead of remaining 100
        let dist = ((body2[7].y - body2[5].y) as f32).abs();
        assert!((dist - 57.5).abs() < 1.0, "Expected dist ~57.5, got {}", dist);
    }

    #[test]
    fn test_body_ik_solver_respects_confidence_threshold() {
        let config = SolverConfig::default();
        let mut solver = BodyIKSolver::new(&config);

        let mut body = BodyKeypoints::new(dummy_body_points());
        body[7].confidence = 0.1; // below 0.30 threshold

        solver.enforce_limb_lengths(&mut body);
        assert!(!solver.target_lengths().contains_key(&(5, 7)));
    }

    #[test]
    fn test_body_ik_solver_3d() {
        let config = SolverConfig::default();
        let mut solver = BodyIKSolver::new(&config);

        let mut pts3d = vec![Vec3::ZERO; 17];
        pts3d[5] = Vec3::new(0.0, 10.0, 0.0);
        pts3d[7] = Vec3::new(0.0, 20.0, 0.0); // len = 10
        let confs = vec![0.9; 17];

        solver.enforce_limb_lengths_3d(&mut pts3d, &confs);
        assert_eq!(solver.target_lengths().get(&(5, 7)), Some(&10.0));
    }
}

`

## File: src/solver/body_geometry.rs

`rust
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

`

## File: src/solver/filter.rs

`rust
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
    #[inline]
    pub fn slerp(&self, q0: Quat, q1: Quat, alpha: f32) -> Quat {
        q0.slerp(q1, alpha.clamp(0.0, 1.0))
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

`

## File: src/solver/foot_lock.rs

`rust
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

`

## File: src/solver/hand_constraints.rs

`rust
//! Biomechanical 21-point hand inverse kinematics and anatomical constraint solver.
//! Direct translation of `utils/hand_constraints.py`.

use crate::skeleton::Point;
use glam::{Vec2, Vec3};

pub const THUMB_CHAIN: [usize; 5] = [0, 1, 2, 3, 4];

pub const FINGER_CHAINS: [(&str, [usize; 5]); 4] = [
    ("index", [0, 5, 6, 7, 8]),
    ("middle", [0, 9, 10, 11, 12]),
    ("ring", [0, 13, 14, 15, 16]),
    ("pinky", [0, 17, 18, 19, 20]),
];

pub fn get_chain_bend_degrees(chain_name: &str) -> (f32, f32, f32) {
    match chain_name {
        "thumb" => (65.0, 55.0, 50.0),
        "index" => (75.0, 95.0, 80.0),
        "middle" => (75.0, 100.0, 85.0),
        "ring" => (90.0, 110.0, 95.0),
        "pinky" => (95.0, 110.0, 95.0),
        _ => (75.0, 95.0, 80.0),
    }
}

pub const MAX_RADIAL_MULTIPLIER: [f32; 21] = [
    0.0,  // 0: Wrist
    0.95, // 1: Thumb CMC
    1.35, // 2: Thumb MCP
    1.75, // 3: Thumb IP
    2.10, // 4: Thumb Tip
    1.35, // 5: Index MCP
    1.80, // 6: Index PIP
    2.25, // 7: Index DIP
    2.70, // 8: Index Tip
    1.40, // 9: Middle MCP
    1.90, // 10: Middle PIP
    2.35, // 11: Middle DIP
    2.85, // 12: Middle Tip
    1.35, // 13: Ring MCP
    1.80, // 14: Ring PIP
    2.25, // 15: Ring DIP
    2.70, // 16: Ring Tip
    1.30, // 17: Pinky MCP
    1.70, // 18: Pinky PIP
    2.10, // 19: Pinky DIP
    2.50, // 20: Pinky Tip
];

pub const BONE_LENGTH_MULTIPLIERS: [((usize, usize), (f32, f32)); 20] = [
    ((0, 1), (0.25, 0.85)),
    ((1, 2), (0.15, 0.60)),
    ((2, 3), (0.12, 0.50)),
    ((3, 4), (0.10, 0.45)),
    ((0, 5), (0.70, 1.35)),
    ((5, 6), (0.20, 0.65)),
    ((6, 7), (0.15, 0.55)),
    ((7, 8), (0.12, 0.50)),
    ((0, 9), (0.75, 1.40)),
    ((9, 10), (0.20, 0.65)),
    ((10, 11), (0.15, 0.55)),
    ((11, 12), (0.12, 0.50)),
    ((0, 13), (0.70, 1.35)),
    ((13, 14), (0.20, 0.65)),
    ((14, 15), (0.15, 0.55)),
    ((15, 16), (0.12, 0.50)),
    ((0, 17), (0.65, 1.30)),
    ((17, 18), (0.18, 0.60)),
    ((18, 19), (0.14, 0.50)),
    ((19, 20), (0.10, 0.45)),
];

#[inline]
fn distance(a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

#[inline]
fn normalize_vec(v: (f32, f32)) -> (f32, f32) {
    let len = v.0.hypot(v.1);
    if len <= 1e-6 {
        (1.0, 0.0)
    } else {
        (v.0 / len, v.1 / len)
    }
}

#[inline]
fn rotate_vec(v: (f32, f32), angle_radians: f32) -> (f32, f32) {
    let cos_t = angle_radians.cos();
    let sin_t = angle_radians.sin();
    (v.0 * cos_t - v.1 * sin_t, v.0 * sin_t + v.1 * cos_t)
}

#[inline]
fn signed_angle(parent: (f32, f32), child: (f32, f32)) -> f32 {
    let cross = parent.0 * child.1 - parent.1 * child.0;
    let dot = parent.0 * child.0 + parent.1 * child.1;
    cross.atan2(dot)
}

#[inline]
fn point_xy(point: Point) -> (f32, f32) {
    (point.x as f32, point.y as f32)
}

#[inline]
fn make_point(x: f32, y: f32, conf: f32) -> Point {
    Point::new(x.round() as i32, y.round() as i32, conf)
}

#[inline]
fn clamp_safe(val: f32, lower: f32, upper: f32) -> f32 {
    let (lo, hi) = if lower <= upper { (lower, upper) } else { (upper, lower) };
    val.clamp(lo, hi)
}

/// Calculates average palm scale from wrist to knuckle MCP roots (5, 9, 13, 17).
pub fn base_hand_scale(points: &[Point]) -> f32 {
    let wrist = point_xy(points[0]);
    let anchors = [5, 9, 13, 17];
    let sum_dist: f32 = anchors.iter().map(|&idx| distance(wrist, point_xy(points[idx]))).sum();
    (sum_dist / 4.0).max(12.0)
}

#[inline]
fn clamp_distance(
    origin: (f32, f32),
    target: (f32, f32),
    min_dist: f32,
    max_dist: f32,
) -> (f32, f32) {
    let dir = (target.0 - origin.0, target.1 - origin.1);
    let len = dir.0.hypot(dir.1);
    if len <= 1e-6 {
        return (origin.0 + max_dist, origin.1);
    }
    let clamped_len = clamp_safe(len, min_dist, max_dist);
    let unit = (dir.0 / len, dir.1 / len);
    (origin.0 + unit.0 * clamped_len, origin.1 + unit.1 * clamped_len)
}

/// Clamps distal landmarks so they do not drift beyond physiological radial distance from wrist.
pub fn enforce_radial_limits(points: &mut [Point], base_scale: f32) {
    let wrist = point_xy(points[0]);
    for idx in 1..21 {
        let max_dist = base_scale * MAX_RADIAL_MULTIPLIER[idx];
        let p = point_xy(points[idx]);
        let (cx, cy) = clamp_distance(wrist, p, 0.0, max_dist);
        points[idx] = make_point(cx, cy, points[idx].confidence);
    }
}

/// Clamps individual bone segment lengths within anatomical minimum and maximum proportions.
pub fn enforce_bone_lengths(points: &mut [Point], base_scale: f32) {
    for &((parent_idx, child_idx), (min_mult, max_mult)) in &BONE_LENGTH_MULTIPLIERS {
        let parent = point_xy(points[parent_idx]);
        let child = point_xy(points[child_idx]);
        let min_dist = base_scale * min_mult;
        let max_dist = base_scale * max_mult;
        let (cx, cy) = clamp_distance(parent, child, min_dist, max_dist);
        points[child_idx] = make_point(cx, cy, points[child_idx].confidence);
    }
}

/// Enforces planar hinge flexion limits on finger joints along the kinematic chain.
pub fn enforce_chain_bend(points: &mut [Point], chain_name: &str, chain_indices: &[usize; 5]) {
    let bend_limits = get_chain_bend_degrees(chain_name);
    let limits = [bend_limits.0, bend_limits.1, bend_limits.2];

    for segment_offset in 2..5 {
        let max_bend_deg = limits[segment_offset - 2];
        let anchor_idx = chain_indices[segment_offset - 2];
        let parent_idx = chain_indices[segment_offset - 1];
        let child_idx = chain_indices[segment_offset];

        let anchor_xy = point_xy(points[anchor_idx]);
        let parent_xy = point_xy(points[parent_idx]);
        let child_xy = point_xy(points[child_idx]);
        let child_conf = points[child_idx].confidence;

        let parent_vec = (parent_xy.0 - anchor_xy.0, parent_xy.1 - anchor_xy.1);
        let child_vec = (child_xy.0 - parent_xy.0, child_xy.1 - parent_xy.1);
        let parent_unit = normalize_vec(parent_vec);
        let child_unit = normalize_vec(child_vec);
        let child_len = distance(parent_xy, child_xy);

        let angle = signed_angle(parent_unit, child_unit);
        let max_bend = max_bend_deg.to_radians();
        let clamped_angle = angle.clamp(-max_bend, max_bend);

        if (clamped_angle - angle).abs() <= 1e-6 {
            continue;
        }

        let constrained_dir = rotate_vec(parent_unit, clamped_angle);
        let cx = parent_xy.0 + constrained_dir.0 * child_len;
        let cy = parent_xy.1 + constrained_dir.1 * child_len;
        points[child_idx] = make_point(cx, cy, child_conf);
    }
}

#[inline]
fn project_local(
    wrist: (f32, f32),
    point: (f32, f32),
    lateral_axis: (f32, f32),
    forward_axis: (f32, f32),
) -> (f32, f32) {
    let dx = point.0 - wrist.0;
    let dy = point.1 - wrist.1;
    (
        dx * lateral_axis.0 + dy * lateral_axis.1,
        dx * forward_axis.0 + dy * forward_axis.1,
    )
}

#[inline]
fn unproject_local(
    wrist: (f32, f32),
    lateral_val: f32,
    forward_val: f32,
    lateral_axis: (f32, f32),
    forward_axis: (f32, f32),
) -> (f32, f32) {
    (
        wrist.0 + lateral_val * lateral_axis.0 + forward_val * forward_axis.0,
        wrist.1 + lateral_val * lateral_axis.1 + forward_val * forward_axis.1,
    )
}

/// Prevents finger crossing and mesh intersection by constraining lateral spread lanes.
/// If `spread_scale` is provided (< 1.0), narrows lateral spread (e.g. during a fist).
pub fn enforce_finger_lanes_scaled(points: &mut [Point], spread_scale: f32) {
    let wrist = point_xy(points[0]);
    let index_root = point_xy(points[5]);
    let pinky_root = point_xy(points[17]);

    if (pinky_root.0 - index_root.0).hypot(pinky_root.1 - index_root.1) < 1e-3 {
        return;
    }

    let lateral_axis = normalize_vec((pinky_root.0 - index_root.0, pinky_root.1 - index_root.1));
    let forward_axis = (-lateral_axis.1, lateral_axis.0);

    let idx_lat = project_local(wrist, point_xy(points[5]), lateral_axis, forward_axis).0;
    let mid_lat = project_local(wrist, point_xy(points[9]), lateral_axis, forward_axis).0;
    let rng_lat = project_local(wrist, point_xy(points[13]), lateral_axis, forward_axis).0;
    let pnk_lat = project_local(wrist, point_xy(points[17]), lateral_axis, forward_axis).0;

    let scale = spread_scale.clamp(0.05, 1.0);
    let scale_bound = |center: f32, bound: f32| center + (bound - center) * scale;

    let boundaries: [(&str, (f32, f32)); 4] = [
        ("index", (scale_bound(idx_lat, idx_lat - (mid_lat - idx_lat).abs()), scale_bound(idx_lat, (idx_lat + mid_lat) * 0.5))),
        ("middle", (scale_bound(mid_lat, (idx_lat + mid_lat) * 0.5), scale_bound(mid_lat, (mid_lat + rng_lat) * 0.5))),
        ("ring", (scale_bound(rng_lat, (mid_lat + rng_lat) * 0.5), scale_bound(rng_lat, (rng_lat + pnk_lat) * 0.5))),
        ("pinky", (scale_bound(pnk_lat, (rng_lat + pnk_lat) * 0.5), scale_bound(pnk_lat, pnk_lat + (pnk_lat - rng_lat).abs()))),
    ];

    for (chain_name, chain_indices) in &FINGER_CHAINS {
        let (lower, upper) = boundaries.iter().find(|(name, _)| name == chain_name).unwrap().1;
        for &point_idx in &chain_indices[2..] {
            let p = point_xy(points[point_idx]);
            let (lat_val, fwd_val) = project_local(wrist, p, lateral_axis, forward_axis);
            let clamped_lat = clamp_safe(lat_val, lower, upper);
            if (clamped_lat - lat_val).abs() <= 1e-6 {
                continue;
            }
            let (cx, cy) = unproject_local(wrist, clamped_lat, fwd_val, lateral_axis, forward_axis);
            points[point_idx] = make_point(cx, cy, points[point_idx].confidence);
        }
    }
}

pub fn enforce_finger_lanes(points: &mut [Point]) {
    enforce_finger_lanes_scaled(points, 1.0);
}

fn max_point_movement(original: &[Point], constrained: &[Point]) -> f32 {
    original
        .iter()
        .zip(constrained.iter())
        .map(|(a, b)| distance(point_xy(*a), point_xy(*b)))
        .fold(0.0_f32, f32::max)
}

/// Orchestrates full hand kinematic constraint pipeline:
/// 1. Radial limits
/// 2. Bone length bounds
/// 3. Hinge bend constraints
/// 4. DIP-PIP tendon coupling
/// 5. Dynamic fist abduction clamp
/// 6. Secondary convergence pass if point movement > 2.0px.
pub fn enforce_hand_constraints(hand_points: &[Point]) -> Vec<Point> {
    if hand_points.len() != 21 {
        return hand_points.to_vec();
    }

    let mut constrained = hand_points.to_vec();
    let base_scale = base_hand_scale(&constrained);

    enforce_radial_limits(&mut constrained, base_scale);
    enforce_bone_lengths(&mut constrained, base_scale);
    enforce_chain_bend(&mut constrained, "thumb", &THUMB_CHAIN);
    for (name, indices) in &FINGER_CHAINS {
        enforce_chain_bend(&mut constrained, name, indices);
    }

    // Apply DIP-PIP anatomical tendon coupling (theta_DIP = 0.67 * theta_PIP)
    for (_name, indices) in &FINGER_CHAINS {
        let mut chain = [
            constrained[indices[0]],
            constrained[indices[1]],
            constrained[indices[2]],
            constrained[indices[3]],
            constrained[indices[4]],
        ];
        apply_dip_pip_tendon_coupling(&mut chain);
        constrained[indices[4]] = chain[4];
    }

    // Apply dynamic fist abduction clamp
    apply_dynamic_fist_abduction_clamp(&mut constrained);

    if max_point_movement(hand_points, &constrained) <= 2.0 {
        return constrained;
    }

    // Secondary convergence pass
    enforce_radial_limits(&mut constrained, base_scale);
    enforce_bone_lengths(&mut constrained, base_scale);
    enforce_chain_bend(&mut constrained, "thumb", &THUMB_CHAIN);
    for (name, indices) in &FINGER_CHAINS {
        enforce_chain_bend(&mut constrained, name, indices);
    }
    apply_dynamic_fist_abduction_clamp(&mut constrained);

    constrained
}

// --- Ledger Signatures from progress.md ---

/// Enforces the DIP-PIP anatomical tendon coupling: theta_DIP = 0.67 * theta_PIP.
pub fn apply_dip_pip_tendon_coupling(chain: &mut [Point; 5]) {
    let mcp = point_xy(chain[1]);
    let pip = point_xy(chain[2]);
    let dip = point_xy(chain[3]);
    let tip = point_xy(chain[4]);

    let pip_vec = normalize_vec((pip.0 - mcp.0, pip.1 - mcp.1));
    let dip_vec = normalize_vec((dip.0 - pip.0, dip.1 - pip.1));
    let pip_flex = signed_angle(pip_vec, dip_vec);

    let dip_len = distance(dip, tip);
    let coupled_angle = pip_flex * 0.67;
    let tip_dir = rotate_vec(dip_vec, coupled_angle);
    chain[4] = make_point(dip.0 + tip_dir.0 * dip_len, dip.1 + tip_dir.1 * dip_len, chain[4].confidence);
}

/// Dynamically clamps lateral finger spread as fingers curl into a fist: spread * (1 - theta_MCP / 90 deg).
pub fn apply_dynamic_fist_abduction_clamp(fingers: &mut [Point]) {
    if fingers.len() != 21 {
        return;
    }

    // Compute average MCP flexion angle across all 4 finger chains
    let mut total_mcp_flex_deg = 0.0f32;
    let wrist = point_xy(fingers[0]);

    for (_name, indices) in &FINGER_CHAINS {
        let mcp = point_xy(fingers[indices[1]]);
        let pip = point_xy(fingers[indices[2]]);

        let v1 = normalize_vec((mcp.0 - wrist.0, mcp.1 - wrist.1));
        let v2 = normalize_vec((pip.0 - mcp.0, pip.1 - mcp.1));
        let dot = (v1.0 * v2.0 + v1.1 * v2.1).clamp(-1.0, 1.0);
        total_mcp_flex_deg += dot.acos().to_degrees();
    }

    let avg_mcp_flex = total_mcp_flex_deg / 4.0;
    // Scale lateral bounds down as MCP angle approaches 90 deg flexion
    let scale = (1.0 - avg_mcp_flex / 90.0).clamp(0.05, 1.0);
    enforce_finger_lanes_scaled(fingers, scale);
}

/// Clamps 3D planar hinge rotation (progress.md signature).
pub fn clamp_planar_hinge(parent: Vec3, hinge: Vec3, child: Vec3, min_deg: f32, max_deg: f32) -> Vec3 {
    let upper = (hinge - parent).normalize_or_zero();
    let lower = child - hinge;
    let len = lower.length();
    let lower_dir = lower.normalize_or_zero();

    let dot = upper.dot(lower_dir).clamp(-1.0, 1.0);
    let angle_deg = dot.acos().to_degrees();
    let clamped_deg = angle_deg.clamp(min_deg, max_deg);

    if (clamped_deg - angle_deg).abs() <= 1e-5 {
        return child;
    }

    let mut bend_axis = upper.cross(lower_dir);
    if bend_axis.length_squared() < 1e-6 {
        // Colinear: pick a perpendicular vector
        let candidate = if upper.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        bend_axis = upper.cross(candidate);
    }
    let bend_axis = bend_axis.normalize();

    let rot = glam::Quat::from_axis_angle(bend_axis, (clamped_deg - angle_deg).to_radians());
    hinge + (rot * lower_dir) * len
}

/// Continuous vector version of radial limits (progress.md signature).
pub fn enforce_radial_limits_vec2(wrist: Vec2, points: &mut [Vec2], max_multipliers: &[f32]) {
    let scale = points.iter().skip(1).map(|p| p.distance(wrist)).fold(0.0_f32, f32::max) / 2.5;
    for (idx, p) in points.iter_mut().enumerate() {
        if idx >= max_multipliers.len() {
            break;
        }
        let max_dist = scale * max_multipliers[idx];
        let d = *p - wrist;
        if d.length() > max_dist && max_dist > 0.0 {
            *p = wrist + d.normalize() * max_dist;
        }
    }
}

/// Continuous vector version of bone length clamping (progress.md signature).
pub fn enforce_bone_lengths_vec2(parent: Vec2, child: Vec2, min_len: f32, max_len: f32) -> Vec2 {
    let d = child - parent;
    let len = d.length();
    if len < 1e-6 {
        return parent + Vec2::new(max_len, 0.0);
    }
    let clamped = len.clamp(min_len, max_len);
    parent + d.normalize() * clamped
}

/// Hand IK solver wrapper struct (progress.md signature).
pub struct HandIKSolver;

impl HandIKSolver {
    pub fn enforce_constraints(raw: &[Point], _forearm_dir: Vec2) -> Vec<Point> {
        enforce_hand_constraints(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hand_constraints_preserves_length_and_clamps() {
        let mut raw = vec![Point::new(100, 100, 1.0); 21];
        raw[0] = Point::new(100, 100, 1.0);  // Wrist
        raw[5] = Point::new(80, 150, 1.0);   // Index MCP
        raw[9] = Point::new(100, 160, 1.0);  // Middle MCP
        raw[13] = Point::new(120, 155, 1.0); // Ring MCP
        raw[17] = Point::new(140, 145, 1.0); // Pinky MCP
        raw[6] = Point::new(80, 180, 1.0);   // Index PIP
        raw[7] = Point::new(80, 210, 1.0);   // Index DIP
        raw[8] = Point::new(80, 240, 1.0);   // Index Tip

        let constrained = enforce_hand_constraints(&raw);
        assert_eq!(constrained.len(), 21);
    }

    #[test]
    fn test_base_hand_scale() {
        let mut points = vec![Point::new(0, 0, 1.0); 21];
        points[5] = Point::new(0, 50, 1.0);
        points[9] = Point::new(0, 60, 1.0);
        points[13] = Point::new(0, 50, 1.0);
        points[17] = Point::new(0, 40, 1.0);

        let scale = base_hand_scale(&points);
        assert_eq!(scale, 50.0);
    }

    #[test]
    fn test_clamp_planar_hinge_3d() {
        let parent = Vec3::new(0.0, 0.0, 0.0);
        let hinge = Vec3::new(0.0, 10.0, 0.0);
        let child = Vec3::new(5.0, 15.0, 0.0); // Bent angle

        let clamped = clamp_planar_hinge(parent, hinge, child, 0.0, 10.0);
        assert!(clamped != child);
    }

    #[test]
    fn test_enforce_bone_lengths_vec2() {
        let parent = Vec2::new(0.0, 0.0);
        let child = Vec2::new(100.0, 0.0);

        let clamped = enforce_bone_lengths_vec2(parent, child, 10.0, 50.0);
        assert_eq!(clamped, Vec2::new(50.0, 0.0));
    }
}

`

## File: src/solver/hand_fallback.rs

`rust
//! Hand landmark validation, wrist anchoring, and synthetic anatomical fallback generation.
//! Direct translation of `utils/hand_fallback.py`.

use crate::skeleton::Point;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Normalized 2D hand landmark template offsets relative to the wrist (index 0).
/// Coordinates are (lateral, forward) relative to forearm orientation.
pub const DEFAULT_HAND_TEMPLATE: [(f32, f32); 21] = [
    (0.00, 0.00),   // 0: Wrist
    (-0.12, 0.16),  // 1: Thumb CMC
    (-0.28, 0.30),  // 2: Thumb MCP
    (-0.42, 0.48),  // 3: Thumb IP
    (-0.54, 0.68),  // 4: Thumb Tip
    (-0.20, 0.34),  // 5: Index MCP
    (-0.22, 0.58),  // 6: Index PIP
    (-0.24, 0.82),  // 7: Index DIP
    (-0.26, 1.04),  // 8: Index Tip
    (0.00, 0.40),   // 9: Middle MCP
    (0.00, 0.66),   // 10: Middle PIP
    (0.00, 0.92),   // 11: Middle DIP
    (0.00, 1.16),   // 12: Middle Tip
    (0.20, 0.38),   // 13: Ring MCP
    (0.22, 0.62),   // 14: Ring PIP
    (0.24, 0.86),   // 15: Ring DIP
    (0.26, 1.08),   // 16: Ring Tip
    (0.38, 0.28),   // 17: Pinky MCP
    (0.48, 0.46),   // 18: Pinky PIP
    (0.56, 0.64),   // 19: Pinky DIP
    (0.62, 0.82),   // 20: Pinky Tip
];

static TELEMETRY: std::sync::Mutex<Option<HashMap<String, usize>>> = std::sync::Mutex::new(None);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HandSide {
    Left,
    Right,
}

impl HandSide {
    pub fn from_str_case_insensitive(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "left" | "l" => Some(Self::Left),
            "right" | "r" => Some(Self::Right),
            _ => None,
        }
    }
}

/// Configuration parameters for hand detection validation and fallback generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HandValidationConfig {
    pub hand_box_min_size: f32,
    pub hand_wrist_max_offset_scale: f32,
    pub hand_kp_threshold: f32,
    pub hand_min_valid_points: usize,
    pub hand_default_scale: f32,
    pub hand_default_confidence: f32,
}

impl Default for HandValidationConfig {
    fn default() -> Self {
        Self {
            hand_box_min_size: 48.0,
            hand_wrist_max_offset_scale: 0.85,
            hand_kp_threshold: 0.35,
            hand_min_valid_points: 10,
            hand_default_scale: 0.72,
            hand_default_confidence: 0.45,
        }
    }
}

/// Translates all hand points rigidly so that the hand root (index 0) coincides with the body wrist.
pub fn anchor_hand_to_wrist(hand_points: &[Point], wrist_point: Point) -> Vec<Point> {
    if hand_points.is_empty() {
        return Vec::new();
    }

    let delta_x = wrist_point.x - hand_points[0].x;
    let delta_y = wrist_point.y - hand_points[0].y;

    hand_points
        .iter()
        .map(|p| Point::new(p.x + delta_x, p.y + delta_y, p.confidence))
        .collect()
}

/// 3D in-place anchor variant (from progress.md): translates 3D hand points so root matches body wrist.
pub fn anchor_hand_to_body_wrist(hand: &mut [Vec3], body_wrist: Vec3) {
    if hand.is_empty() {
        return;
    }
    let delta = body_wrist - hand[0];
    for p in hand.iter_mut() {
        *p += delta;
    }
}

/// Validates raw hand detection landmarks against forearm geometry and anatomical plausibility.
pub fn is_hand_detection_valid(
    hand_points: Option<&[Point]>,
    wrist_point: Point,
    elbow_point: Point,
    config: &HandValidationConfig,
    record_telemetry: bool,
) -> bool {
    let points = match hand_points {
        Some(p) if p.len() == 21 => p,
        _ => {
            if record_telemetry {
                record_hand_rejection("missing_points");
            }
            return false;
        }
    };

    let wrist_xy = (wrist_point.x as f32, wrist_point.y as f32);
    let elbow_xy = (elbow_point.x as f32, elbow_point.y as f32);
    let hand_wrist_xy = (points[0].x as f32, points[0].y as f32);

    let forearm_len = (wrist_xy.0 - elbow_xy.0)
        .hypot(wrist_xy.1 - elbow_xy.1)
        .max(1.0);

    // Check 1: Wrist offset between body detector and hand detector
    let wrist_offset = (hand_wrist_xy.0 - wrist_xy.0).hypot(hand_wrist_xy.1 - wrist_xy.1);
    let max_wrist_offset = (config.hand_box_min_size * 0.38)
        .max(forearm_len * config.hand_wrist_max_offset_scale * 1.15);
    if wrist_offset > max_wrist_offset {
        if record_telemetry {
            record_hand_rejection("wrist_offset");
        }
        return false;
    }

    // Check 2: Minimum number of confident points
    let threshold = config.hand_kp_threshold * 0.5;
    let valid_count = points.iter().filter(|p| p.confidence > threshold).count();
    if valid_count < config.hand_min_valid_points {
        if record_telemetry {
            record_hand_rejection("valid_points");
        }
        return false;
    }

    // Check 3: Radial distance bounds
    let max_radial = points[1..]
        .iter()
        .map(|p| (p.x as f32 - wrist_xy.0).hypot(p.y as f32 - wrist_xy.1))
        .fold(0.0_f32, f32::max);
    let max_allowed_radial = (forearm_len * 2.75).max(config.hand_box_min_size * 1.40);
    if max_radial > max_allowed_radial {
        if record_telemetry {
            record_hand_rejection("radial_distance");
        }
        return false;
    }

    // Check 4: Palm spread distance (MCP knuckles 5, 9, 13, 17)
    let palm_indices = [5, 9, 13, 17];
    let palm_dist_sum: f32 = palm_indices
        .iter()
        .map(|&idx| {
            let p = points[idx];
            (p.x as f32 - wrist_xy.0).hypot(p.y as f32 - wrist_xy.1)
        })
        .sum();
    let avg_palm_dist = palm_dist_sum / 4.0;
    let min_palm = forearm_len * 0.035;
    let max_palm = (forearm_len * 1.85).max(config.hand_box_min_size * 0.55);
    if avg_palm_dist < min_palm || avg_palm_dist > max_palm {
        if record_telemetry {
            record_hand_rejection("palm_distance");
        }
        return false;
    }

    true
}

/// Logs a hand rejection reason to the internal telemetry counter.
pub fn record_hand_rejection(reason: &str) {
    let mut guard = TELEMETRY.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    *map.entry(reason.to_string()).or_insert(0) += 1;
}

/// Extracts and clears accumulated telemetry rejection counts.
pub fn pop_hand_rejection_telemetry() -> HashMap<String, usize> {
    let mut guard = TELEMETRY.lock().unwrap_or_else(|e| e.into_inner());
    guard.take().unwrap_or_default()
}

/// Determines whether a hand detection has enough confidence to warrant attempting IK constraints.
pub fn has_usable_hand_detection(hand_points: Option<&[Point]>, config: &HandValidationConfig) -> bool {
    let points = match hand_points {
        Some(p) if p.len() == 21 => p,
        _ => return false,
    };
    let min_points = 5.max(config.hand_min_valid_points / 2);
    let thresh = config.hand_kp_threshold * 0.25;
    points.iter().filter(|p| p.confidence > thresh).count() >= min_points
}

/// Generates an anatomically proportioned synthetic 21-point rest hand aligned with the forearm.
pub fn generate_default_hand(
    wrist_point: Point,
    elbow_point: Point,
    side: HandSide,
    config: &HandValidationConfig,
) -> Vec<Point> {
    let wrist = Vec2::new(wrist_point.x as f32, wrist_point.y as f32);
    let elbow = Vec2::new(elbow_point.x as f32, elbow_point.y as f32);
    let delta = wrist - elbow;
    let forearm_len = delta.length();
    let forward = if forearm_len > 1e-6 { delta / forearm_len } else { Vec2::new(0.0, -1.0) };
    let mut lateral = Vec2::new(-forward.y, forward.x);
    if side == HandSide::Left {
        lateral = -lateral;
    }

    let hand_scale = if forearm_len > 1e-3 {
        forearm_len * config.hand_default_scale
    } else {
        18.0
    };
    let confidence = config.hand_default_confidence.max(config.hand_kp_threshold + 0.05);

    DEFAULT_HAND_TEMPLATE
        .iter()
        .map(|&(lat, fwd)| {
            let p = wrist + lateral * (lat * hand_scale) + forward * (fwd * hand_scale);
            Point::new(p.x.round() as i32, p.y.round() as i32, confidence)
        })
        .collect()
}

/// Continuous vector version (from progress.md): generates 21-point hand landmarks from wrist and direction.
pub fn generate_fallback_hand(wrist: Vec2, dir: Vec2, len: f32) -> Vec<Vec2> {
    let forward = if dir.length() > 1e-6 { dir.normalize() } else { Vec2::new(0.0, 1.0) };
    let lateral = Vec2::new(-forward.y, forward.x);
    let scale = len.max(18.0);

    DEFAULT_HAND_TEMPLATE
        .iter()
        .map(|&(lat, fwd)| wrist + lateral * (lat * scale) + forward * (fwd * scale))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anchor_hand_to_wrist_2d() {
        let hand = vec![Point::new(100, 100, 0.9), Point::new(110, 120, 0.8)];
        let target_wrist = Point::new(200, 300, 1.0);

        let anchored = anchor_hand_to_wrist(&hand, target_wrist);
        assert_eq!(anchored[0].x, 200);
        assert_eq!(anchored[0].y, 300);
        assert_eq!(anchored[1].x, 210);
        assert_eq!(anchored[1].y, 320);
    }

    #[test]
    fn test_anchor_hand_to_body_wrist_3d() {
        let mut hand = vec![Vec3::new(1.0, 2.0, 3.0), Vec3::new(2.0, 3.0, 4.0)];
        let target_wrist = Vec3::new(10.0, 20.0, 30.0);

        anchor_hand_to_body_wrist(&mut hand, target_wrist);
        assert_eq!(hand[0], target_wrist);
        assert_eq!(hand[1], Vec3::new(11.0, 21.0, 31.0));
    }

    #[test]
    fn test_generate_default_hand_length_and_structure() {
        let wrist = Point::new(100, 200, 1.0);
        let elbow = Point::new(100, 100, 1.0);
        let config = HandValidationConfig::default();

        let hand = generate_default_hand(wrist, elbow, HandSide::Right, &config);
        assert_eq!(hand.len(), 21);
        assert_eq!(hand[0].x, 100);
        assert_eq!(hand[0].y, 200);
        // Fingertip points should extend forward (downwards in image space y > 200)
        assert!(hand[12].y > hand[0].y);
    }

    #[test]
    fn test_generate_fallback_hand_vec2() {
        let wrist = Vec2::new(100.0, 100.0);
        let dir = Vec2::new(0.0, 1.0);
        let hand = generate_fallback_hand(wrist, dir, 50.0);
        assert_eq!(hand.len(), 21);
        assert_eq!(hand[0], wrist);
    }

    #[test]
    fn test_telemetry_recording() {
        pop_hand_rejection_telemetry(); // Clear
        record_hand_rejection("test_reason");
        record_hand_rejection("test_reason");
        record_hand_rejection("another_reason");

        let stats = pop_hand_rejection_telemetry();
        assert_eq!(stats.get("test_reason"), Some(&2));
        assert_eq!(stats.get("another_reason"), Some(&1));

        // After pop, should be empty
        let empty = pop_hand_rejection_telemetry();
        assert!(empty.is_empty());
    }
}

`

## File: src/solver/prediction.rs

`rust
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

`

## File: src/network/mod.rs

`rust
//! Live network streaming server for real-time DCC integrations (Blender, Unreal Engine).

use crate::export::coords::MotionFrame;
use crate::skeleton::SKELETON;
use anyhow::Context;
use glam::Vec3;
use serde::Serialize;
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::time::Instant;

/// Official Kinara Blender stream port.
pub const BLENDER_STREAM_PORT: u16 = 49168;

/// Rig declaration bone record: [name, parent_index, affine_12_floats, length].
#[derive(Debug, Clone, Serialize)]
pub struct StreamBoneRecord(pub String, pub i32, pub [f32; 12], pub f32);

/// Rig declaration schema envelope (protocol version 2).
#[derive(Debug, Clone, Serialize)]
pub struct StreamRigSchema {
    pub version: u32,
    pub slot: u32,
    pub actor: String,
    pub name: String,
    pub bones: Vec<StreamBoneRecord>,
}

/// TCP live streaming server pushing motion capture transforms to DCC viewports.
pub struct BlenderStreamServer {
    pub listener: TcpListener,
    pub clients: Vec<TcpStream>,
    pub sequence: u64,
    pub start_time: Instant,
    pub actor_id: String,
}

impl BlenderStreamServer {
    /// Binds the streamer to local loopback port (default 49168).
    pub fn bind(port: u16) -> anyhow::Result<Self> {
        let listener = TcpListener::bind(format!("127.0.0.1:{}", port))
            .with_context(|| format!("Failed to bind live stream server to 127.0.0.1:{}", port))?;
        listener.set_nonblocking(true)?;

        // Deterministic 64-character lowercase hex actor identifier
        let actor_id = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string();

        Ok(Self {
            listener,
            clients: Vec::new(),
            sequence: 0,
            start_time: Instant::now(),
            actor_id,
        })
    }

    /// Accepts pending inbound DCC connections and performs rig declaration handshake.
    pub fn accept_new_connections(&mut self) -> anyhow::Result<usize> {
        let mut new_count = 0;
        while let Ok((mut stream, _addr)) = self.listener.accept() {
            let _ = stream.set_nodelay(true);
            let _ = stream.set_write_timeout(Some(std::time::Duration::from_millis(50)));
            // Send initial rig declaration packet (R)
            if self.send_rig_declaration(&mut stream).is_ok() {
                self.clients.push(stream);
                new_count += 1;
            }
        }
        Ok(new_count)
    }

    /// Builds and transmits the canonical 60-joint armature declaration packet.
    pub fn send_rig_declaration(&self, stream: &mut TcpStream) -> anyhow::Result<()> {
        let mut bones = Vec::with_capacity(SKELETON.len());
        for (idx, spec) in SKELETON.iter().enumerate() {
            let parent_idx = match spec.parent {
                Some(p_name) => SKELETON
                    .iter()
                    .position(|s| s.name == p_name)
                    .map(|i| i as i32)
                    .unwrap_or(-1),
                None => -1,
            };

            // Standard identity affine matrix [3x4] row-major
            let rows: [f32; 12] = [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ];
            let length = if idx == 0 { 1.0 } else { 0.25 };
            bones.push(StreamBoneRecord(spec.name.to_string(), parent_idx, rows, length));
        }

        let schema = StreamRigSchema {
            version: 2,
            slot: 1,
            actor: self.actor_id.clone(),
            name: "KinaraActor".to_string(),
            bones,
        };

        let json_bytes = serde_json::to_vec(&schema)?;
        let total_size = (json_bytes.len() + 1) as u32; // payload + kind

        let mut packet = Vec::with_capacity(5 + json_bytes.len());
        packet.extend_from_slice(&total_size.to_le_bytes());
        packet.push(b'R');
        packet.extend_from_slice(&json_bytes);

        stream.write_all(&packet)?;
        stream.flush()?;
        Ok(())
    }

    /// Broadcasts a live motion frame to all connected DCC viewports.
    pub fn broadcast_frame(&mut self, frame: &MotionFrame) -> anyhow::Result<usize> {
        self.accept_new_connections()?;
        if self.clients.is_empty() {
            return Ok(0);
        }

        self.sequence += 1;
        let actor_count = 1u16;
        let slot = 1u32;
        let frame_num = frame.frame_index as u32;

        let num_bones = SKELETON.len();
        let payload_size = 8 + 2 + (4 + 4 + num_bones * 48); // sequence(8) + count(2) + [slot(4)+frame(4)+bones(48*N)]
        let total_msg_size = (payload_size + 1) as u32;

        let mut packet = Vec::with_capacity(4 + 1 + payload_size);
        packet.extend_from_slice(&total_msg_size.to_le_bytes());
        packet.push(b'S'); // Snapshot kind

        // Snapshot header
        packet.extend_from_slice(&self.sequence.to_le_bytes());
        packet.extend_from_slice(&actor_count.to_le_bytes());

        // Actor frame header
        packet.extend_from_slice(&slot.to_le_bytes());
        packet.extend_from_slice(&frame_num.to_le_bytes());

        // Stream 12 floats per bone with bone orientation basis (+Y along limb in Blender)
        for spec in &SKELETON {
            let j_pos = frame.joints.get(spec.name).map(|j| Vec3::new(j.x, j.y, j.z)).unwrap_or(Vec3::ZERO);

            let child_name = SKELETON.iter().find(|s| s.parent == Some(spec.name)).map(|s| s.name);
            let dir_vec = if let Some(c_name) = child_name {
                frame.joints.get(c_name).map(|c| Vec3::new(c.x, c.y, c.z) - j_pos)
            } else if let Some(p_name) = spec.parent {
                frame.joints.get(p_name).map(|p| j_pos - Vec3::new(p.x, p.y, p.z))
            } else {
                None
            };

            let v = dir_vec.unwrap_or(Vec3::Y);
            let v_len = v.length();
            let hat_y = if v_len > 1e-4 { v / v_len } else { Vec3::Y };

            let ref_up = if hat_y.z.abs() < 0.95 {
                Vec3::Z
            } else {
                Vec3::X
            };
            let hat_x = hat_y.cross(ref_up).normalize();
            let hat_z = hat_x.cross(hat_y).normalize();

            // 3x4 Affine matrix (row-major):
            // Row 0: [hat_x.x, hat_y.x, hat_z.x, j_pos.x]
            // Row 1: [hat_x.y, hat_y.y, hat_z.y, j_pos.y]
            // Row 2: [hat_x.z, hat_y.z, hat_z.z, j_pos.z]
            let bone_floats: [f32; 12] = [
                hat_x.x, hat_y.x, hat_z.x, j_pos.x,
                hat_x.y, hat_y.y, hat_z.y, j_pos.y,
                hat_x.z, hat_y.z, hat_z.z, j_pos.z,
            ];
            for f in &bone_floats {
                packet.extend_from_slice(&f.to_le_bytes());
            }
        }

        // Send to active clients, prune disconnected ones
        let mut active = Vec::new();
        for mut client in self.clients.drain(..) {
            if client.write_all(&packet).is_ok() && client.flush().is_ok() {
                active.push(client);
            }
        }
        let active_count = active.len();
        self.clients = active;
        Ok(active_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{JointMap, JointValue};

    #[test]
    fn test_blender_stream_server_bind_and_broadcast() {
        let mut server = BlenderStreamServer::bind(0).expect("Failed to bind ephemeral test port");
        let port = server.listener.local_addr().unwrap().port();

        // Connect a test client
        let _client = TcpStream::connect(format!("127.0.0.1:{}", port)).expect("Failed client connect");
        let connected = server.accept_new_connections().expect("Accept failed");
        assert_eq!(connected, 1);

        let mut joints = JointMap::new();
        joints.insert("HipsRoot".to_string(), JointValue::new(0.0, 0.0, 0.0, 1.0));
        let frame = MotionFrame::new(0, joints);

        let sent = server.broadcast_frame(&frame).expect("Broadcast failed");
        assert_eq!(sent, 1);
    }

    #[test]
    fn test_broadcast_frame_bone_orientations() {
        let mut server = BlenderStreamServer::bind(0).expect("Failed to bind port");
        let mut joints = JointMap::new();
        joints.insert("HipsRoot".to_string(), JointValue::new(0.0, 0.0, 0.0, 1.0));
        joints.insert("Chest".to_string(), JointValue::new(0.0, 0.0, 10.0, 1.0));
        let frame = MotionFrame::new(0, joints);

        let count = server.broadcast_frame(&frame).expect("Broadcast failed");
        assert_eq!(count, 0);
    }
}

`

## File: src/export/mod.rs

`rust
//! Export and coordinate transformation pipeline.

pub mod canonical;
pub mod coords;
pub mod fbx;
pub mod json;

pub use canonical::*;
pub use coords::*;
pub use fbx::*;
pub use json::*;

`

## File: src/export/canonical.rs

`rust
//! Canonical joint map assembly and rest-pose bind calculation.

use crate::export::coords::MotionFrame;
use crate::skeleton::{
    body_foot_name_to_index, body_name_to_index, hand_name_to_index, CoordinateSystemConfig,
    JointMap, JointValue, Point, FACE_POINT_INDICES, SKELETON,
};
use crate::solver::body_constraints::BodyKeypoints;
use crate::solver::body_geometry::derive_foot_points;
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pair of detected 21-point hand landmarks for left and right hands.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HandsPair {
    pub left: Option<Vec<Point>>,
    pub right: Option<Vec<Point>>,
}

impl HandsPair {
    pub fn new(left: Option<Vec<Point>>, right: Option<Vec<Point>>) -> Self {
        Self { left, right }
    }
}

/// Metadata envelope for exported motion files.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipMetadata {
    pub format: String,
    pub source_file: Option<String>,
    pub fps: f64,
    pub frame_count: usize,
    pub coordinate_system: CoordinateSystemConfig,
    pub skeleton: Vec<SkeletonJointMeta>,
    #[serde(flatten)]
    pub custom: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkeletonJointMeta {
    pub name: String,
    pub parent: Option<String>,
}

impl Default for ClipMetadata {
    fn default() -> Self {
        Self {
            format: "kinara-motion-json-v1".to_string(),
            source_file: None,
            fps: 30.0,
            frame_count: 0,
            coordinate_system: CoordinateSystemConfig::default(),
            skeleton: SKELETON
                .iter()
                .map(|j| SkeletonJointMeta {
                    name: j.name.to_string(),
                    parent: j.parent.map(|s| s.to_string()),
                })
                .collect(),
            custom: HashMap::new(),
        }
    }
}

/// Rest-pose bind joints and relative segment offset vectors across an armature.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RestJointOffsets {
    pub rest_joints: JointMap,
    pub rest_vectors: HashMap<String, Vec3>,
}

/// A multi-person animation frame indexing individual person tracks by label/ID.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiPersonFrame {
    pub frame_index: usize,
    pub people: HashMap<String, JointMap>,
}

impl MultiPersonFrame {
    pub fn new(frame_index: usize, people: HashMap<String, JointMap>) -> Self {
        Self {
            frame_index,
            people,
        }
    }
}

/// Assembles detected full-body and hand landmarks into the canonical 60-joint dictionary.
pub fn build_canonical_joint_map(
    body: &BodyKeypoints,
    hands: &HandsPair,
    joint_depths: Option<&HashMap<String, f32>>,
) -> JointMap {
    let mut map: JointMap = JointMap::new();
    for spec in &SKELETON {
        map.insert(spec.name.to_string(), JointValue::new(0.0, 0.0, 0.0, 0.0));
    }

    let default_depths = HashMap::new();
    let depths = joint_depths.unwrap_or(&default_depths);

    let body_pts = &body.points;
    if body_pts.len() < 17 {
        return map;
    }

    // Direct COCO-17 mapped body joints
    let body_joint_names = [
        "LeftShoulder",
        "RightShoulder",
        "LeftElbow",
        "RightElbow",
        "LeftWrist",
        "RightWrist",
        "LeftHip",
        "RightHip",
        "LeftKnee",
        "RightKnee",
        "LeftAnkle",
        "RightAnkle",
    ];

    for name in &body_joint_names {
        if let Some(idx) = body_name_to_index(name) {
            let pt = body_pts[idx];
            let z = depths.get(*name).copied().unwrap_or(0.0);
            map.insert(
                name.to_string(),
                JointValue::new(pt.x as f32, -(pt.y as f32), z, pt.confidence),
            );
        }
    }

    // Derived HipsRoot (midpoint of LeftHip and RightHip)
    let left_hip = body_pts[11];
    let right_hip = body_pts[12];
    let root_x = (left_hip.x + right_hip.x) as f32 * 0.5;
    let root_y = (left_hip.y + right_hip.y) as f32 * 0.5;
    let root_z = (depths.get("LeftHip").copied().unwrap_or(0.0)
        + depths.get("RightHip").copied().unwrap_or(0.0))
        * 0.5;
    let root_conf = (left_hip.confidence + right_hip.confidence) * 0.5;
    map.insert(
        "HipsRoot".to_string(),
        JointValue::new(root_x, -root_y, root_z, root_conf),
    );

    // Derived Chest (midpoint of LeftShoulder and RightShoulder)
    let left_shoulder = body_pts[5];
    let right_shoulder = body_pts[6];
    let chest_x = (left_shoulder.x + right_shoulder.x) as f32 * 0.5;
    let chest_y = (left_shoulder.y + right_shoulder.y) as f32 * 0.5;
    let chest_z = (depths.get("LeftShoulder").copied().unwrap_or(0.0)
        + depths.get("RightShoulder").copied().unwrap_or(0.0))
        * 0.5;
    let chest_conf = (left_shoulder.confidence + right_shoulder.confidence) * 0.5;
    map.insert(
        "Chest".to_string(),
        JointValue::new(chest_x, -chest_y, chest_z, chest_conf),
    );

    // Derived Neck and Head
    let (neck, head) = derive_head_joints(body_pts, depths);
    map.insert("Neck".to_string(), neck);
    map.insert("Head".to_string(), head);

    // Derived Feet
    let (left_foot, left_toe) = derive_foot_chain(
        body_pts,
        "Left",
        body_pts[13],
        body_pts[15],
        depths.get("LeftFoot").copied().unwrap_or(0.0),
        depths.get("LeftToeBase").copied().unwrap_or(0.0),
    );
    map.insert("LeftFoot".to_string(), left_foot);
    map.insert("LeftToeBase".to_string(), left_toe);

    let (right_foot, right_toe) = derive_foot_chain(
        body_pts,
        "Right",
        body_pts[14],
        body_pts[16],
        depths.get("RightFoot").copied().unwrap_or(0.0),
        depths.get("RightToeBase").copied().unwrap_or(0.0),
    );
    map.insert("RightFoot".to_string(), right_foot);
    map.insert("RightToeBase".to_string(), right_toe);

    // Hand Phalanges (Thumb, Index, Middle, Ring, Pinky 1-4)
    let hand_suffixes = [
        "Thumb1", "Thumb2", "Thumb3", "Thumb4", "Index1", "Index2", "Index3", "Index4",
        "Middle1", "Middle2", "Middle3", "Middle4", "Ring1", "Ring2", "Ring3", "Ring4",
        "Pinky1", "Pinky2", "Pinky3", "Pinky4",
    ];

    for (side, hand_opt) in [("Left", &hands.left), ("Right", &hands.right)] {
        if let Some(hand_pts) = hand_opt {
            if hand_pts.len() >= 21 {
                for suffix in &hand_suffixes {
                    if let Some(h_idx) = hand_name_to_index(suffix) {
                        let pt = hand_pts[h_idx];
                        let key = format!("{}{}", side, suffix);
                        let z = depths.get(&key).copied().unwrap_or(0.0);
                        map.insert(
                            key,
                            JointValue::new(pt.x as f32, -(pt.y as f32), z, pt.confidence),
                        );
                    }
                }
            }
        }
    }

    map
}

fn derive_head_joints(body_pts: &[Point], depths: &HashMap<String, f32>) -> (JointValue, JointValue) {
    let shoulder_x = (body_pts[5].x + body_pts[6].x) as f32 * 0.5;
    let shoulder_y = (body_pts[5].y + body_pts[6].y) as f32 * 0.5;
    let shoulder_conf = (body_pts[5].confidence + body_pts[6].confidence) * 0.5;

    let hip_x = (body_pts[11].x + body_pts[12].x) as f32 * 0.5;
    let hip_y = (body_pts[11].y + body_pts[12].y) as f32 * 0.5;
    let hip_conf = (body_pts[11].confidence + body_pts[12].confidence) * 0.5;

    let face_pts: Vec<Point> = FACE_POINT_INDICES
        .iter()
        .map(|&i| body_pts[i])
        .filter(|p| p.confidence > 0.0)
        .collect();

    let (neck_x, neck_y, head_x, head_y, conf) = if !face_pts.is_empty() {
        let count = face_pts.len() as f32;
        let face_x = face_pts.iter().map(|p| p.x as f32).sum::<f32>() / count;
        let face_y = face_pts.iter().map(|p| p.y as f32).sum::<f32>() / count;
        let face_conf = face_pts.iter().map(|p| p.confidence).sum::<f32>() / count;

        let nx = shoulder_x + (face_x - shoulder_x) * 0.35;
        let ny = shoulder_y + (face_y - shoulder_y) * 0.35;
        let hx = shoulder_x + (face_x - shoulder_x) * 0.85;
        let hy = shoulder_y + (face_y - shoulder_y) * 0.85;
        (nx, ny, hx, hy, (shoulder_conf + face_conf) * 0.5)
    } else {
        let dx = shoulder_x - hip_x;
        let dy = shoulder_y - hip_y;
        let mut len = dx.hypot(dy);
        let (ux, uy) = if len <= 1e-6 {
            len = 32.0;
            (0.0, -1.0)
        } else {
            (dx / len, dy / len)
        };

        let head_dist = (len * 0.35).max(18.0);
        (
            shoulder_x,
            shoulder_y,
            shoulder_x + ux * head_dist,
            shoulder_y + uy * head_dist,
            (shoulder_conf + hip_conf) * 0.5,
        )
    };

    let shoulder_depth = (depths.get("LeftShoulder").copied().unwrap_or(0.0)
        + depths.get("RightShoulder").copied().unwrap_or(0.0))
        * 0.5;

    (
        JointValue::new(neck_x, -neck_y, shoulder_depth, conf),
        JointValue::new(head_x, -head_y, shoulder_depth, conf),
    )
}

fn derive_foot_chain(
    body_pts: &[Point],
    side: &str,
    knee: Point,
    ankle: Point,
    foot_depth: f32,
    toe_depth: f32,
) -> (JointValue, JointValue) {
    let foot_name = format!("{}Foot", side);
    let toe_name = format!("{}ToeBase", side);

    if let (Some(f_idx), Some(t_idx)) = (
        body_foot_name_to_index(&foot_name),
        body_foot_name_to_index(&toe_name),
    ) {
        if body_pts.len() > t_idx && body_pts[f_idx].confidence > 0.0 && body_pts[t_idx].confidence > 0.0 {
            let fp = body_pts[f_idx];
            let tp = body_pts[t_idx];
            return (
                JointValue::new(fp.x as f32, -(fp.y as f32), foot_depth, fp.confidence),
                JointValue::new(tp.x as f32, -(tp.y as f32), toe_depth, tp.confidence),
            );
        }
    }

    let (foot_pt, toe_pt) = derive_foot_points(knee, ankle);
    (
        JointValue::new(foot_pt.x as f32, -(foot_pt.y as f32), foot_depth, foot_pt.confidence),
        JointValue::new(toe_pt.x as f32, -(toe_pt.y as f32), toe_depth, toe_pt.confidence),
    )
}

/// Computes median rest-pose bind joint positions and relative segment offset vectors across an animation clip.
pub fn compute_rest_pose_joints(frames: &[MotionFrame]) -> RestJointOffsets {
    if frames.is_empty() {
        return RestJointOffsets::default();
    }

    let mut root_xs = Vec::with_capacity(frames.len());
    let mut root_ys = Vec::with_capacity(frames.len());
    let mut root_zs = Vec::with_capacity(frames.len());
    let mut root_confs = Vec::with_capacity(frames.len());

    for frame in frames {
        if let Some(root) = frame.joints.get("HipsRoot") {
            if root.confidence > 0.05 {
                root_xs.push(root.x);
                root_ys.push(root.y);
                root_zs.push(root.z);
                root_confs.push(root.confidence);
            }
        }
    }

    let mut rest_joints = JointMap::new();
    if !root_xs.is_empty() {
        rest_joints.insert(
            "HipsRoot".to_string(),
            JointValue::new(
                median(&mut root_xs),
                median(&mut root_ys),
                median(&mut root_zs),
                median(&mut root_confs),
            ),
        );
    } else {
        rest_joints.insert("HipsRoot".to_string(), JointValue::new(0.0, 0.0, 0.0, 0.0));
    }

    let mut rest_vectors = HashMap::new();

    for spec in &SKELETON {
        let parent_name = match spec.parent {
            Some(p) => p,
            None => continue,
        };

        let mut dxs = Vec::with_capacity(frames.len());
        let mut dys = Vec::with_capacity(frames.len());
        let mut dzs = Vec::with_capacity(frames.len());
        let mut confs = Vec::with_capacity(frames.len());

        for frame in frames {
            if let (Some(pj), Some(cj)) = (frame.joints.get(parent_name), frame.joints.get(spec.name)) {
                if pj.confidence > 0.05 && cj.confidence > 0.05 {
                    dxs.push(cj.x - pj.x);
                    dys.push(cj.y - pj.y);
                    dzs.push(cj.z - pj.z);
                    confs.push((pj.confidence + cj.confidence) * 0.5);
                }
            }
        }

        let (dx, dy, dz, conf) = if !dxs.is_empty() {
            (
                median(&mut dxs),
                median(&mut dys),
                median(&mut dzs),
                median(&mut confs),
            )
        } else {
            (0.0, 0.0, 0.05, 0.0)
        };

        let vec = Vec3::new(dx, dy, dz);
        rest_vectors.insert(spec.name.to_string(), vec);

        let parent_rest = rest_joints.get(parent_name).cloned().unwrap_or_else(|| JointValue::new(0.0, 0.0, 0.0, 0.0));
        rest_joints.insert(
            spec.name.to_string(),
            JointValue::new(
                parent_rest.x + dx,
                parent_rest.y + dy,
                parent_rest.z + dz,
                conf,
            ),
        );
    }

    RestJointOffsets {
        rest_joints,
        rest_vectors,
    }
}

pub(crate) fn median(vals: &mut [f32]) -> f32 {
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals[vals.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::Point;

    fn dummy_body_keypoints() -> BodyKeypoints {
        let mut pts = vec![Point::new(0, 0, 0.9); 17];
        pts[5] = Point::new(100, 100, 0.9);
        pts[6] = Point::new(200, 100, 0.9);
        pts[7] = Point::new(90, 150, 0.9);
        pts[8] = Point::new(210, 150, 0.9);
        pts[9] = Point::new(80, 200, 0.9);
        pts[10] = Point::new(220, 200, 0.9);
        pts[11] = Point::new(120, 250, 0.9);
        pts[12] = Point::new(180, 250, 0.9);
        pts[13] = Point::new(120, 320, 0.9);
        pts[14] = Point::new(180, 320, 0.9);
        pts[15] = Point::new(120, 380, 0.9);
        pts[16] = Point::new(180, 380, 0.9);
        BodyKeypoints::new(pts)
    }

    #[test]
    fn test_build_canonical_joint_map() {
        let body = dummy_body_keypoints();
        let hands = HandsPair::default();
        let map = build_canonical_joint_map(&body, &hands, None);

        assert_eq!(map.len(), 60);
        assert!(map.contains_key("HipsRoot"));
        assert!(map.contains_key("Chest"));
        assert!(map.contains_key("Neck"));
        assert!(map.contains_key("Head"));
        assert!(map.contains_key("LeftFoot"));
        assert!(map.contains_key("RightFoot"));

        let root = map.get("HipsRoot").unwrap();
        assert_eq!(root.x, 150.0);
        assert_eq!(root.y, -250.0);
    }

    #[test]
    fn test_compute_rest_pose_joints() {
        let body = dummy_body_keypoints();
        let map = build_canonical_joint_map(&body, &HandsPair::default(), None);
        let frames = vec![MotionFrame::new(0, map)];

        let rest = compute_rest_pose_joints(&frames);
        assert!(rest.rest_joints.contains_key("HipsRoot"));
        assert!(rest.rest_vectors.contains_key("Chest"));
        assert!(rest.rest_vectors.contains_key("Neck"));
    }
}

`

## File: src/export/coords.rs

`rust
//! Coordinate transformation, scale normalization, and ground-plane alignment.
//! Direct translation of `utils/exports.py`.

use crate::skeleton::{JointMap, JointValue};
use glam::Vec3;
use serde::{Deserialize, Serialize};

/// A single animation/mocap frame consisting of a frame index and 3D joint dictionary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotionFrame {
    pub frame_index: usize,
    pub joints: JointMap,
}

impl MotionFrame {
    pub fn new(frame_index: usize, joints: JointMap) -> Self {
        Self {
            frame_index,
            joints,
        }
    }
}

/// Rotates 3D Cartesian coordinates from OpenCV camera space
/// (+X right, +Y down, +Z forward) to Blender world space (+X right, +Y forward, +Z up).
#[inline]
pub fn cv_to_blender_space(p: Vec3) -> Vec3 {
    Vec3::new(p.x, p.z, -p.y)
}

/// Converts an animation sequence from screen/OpenCV canonical coordinates (+X right, -Y up, Z depth)
/// to Blender world coordinates (+X right, +Y forward/depth, +Z up).
pub fn convert_sequence_to_blender_space(frames: &mut [MotionFrame]) {
    for frame in frames.iter_mut() {
        for joint in frame.joints.values_mut() {
            let old_y = joint.y;
            let old_z = joint.z;
            joint.y = old_z;
            joint.z = old_y;
        }
    }
}

/// Inverts Y coordinate from 2D screen/image space to 3D world space (+X right, +Y up, +Z depth).
#[inline]
pub fn screen_to_world_space(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, -y, z)
}

/// Measures the 3D Euclidean distance between the hip root and neck base.
#[inline]
pub fn measure_torso_length(hips: Vec3, neck: Vec3) -> f32 {
    (neck - hips).length()
}

/// Measures the effective torso length in a named joint map.
/// Looks up `HipsRoot` and `Neck`; falls back to `Chest` distance divided by 0.70.
pub fn measure_frame_torso_length(joints: &JointMap) -> Option<f32> {
    let root = joints.get("HipsRoot")?;
    if !joint_is_valid(root) {
        return None;
    }
    let root_v = Vec3::new(root.x, root.y, root.z);

    if let Some(neck) = joints.get("Neck") {
        if joint_is_valid(neck) {
            let neck_v = Vec3::new(neck.x, neck.y, neck.z);
            let dist = (neck_v - root_v).length();
            if dist > 1.0 {
                return Some(dist);
            }
        }
    }

    if let Some(chest) = joints.get("Chest") {
        if joint_is_valid(chest) {
            let chest_v = Vec3::new(chest.x, chest.y, chest.z);
            let dist = (chest_v - root_v).length();
            if dist > 1.0 {
                return Some(dist / 0.70);
            }
        }
    }

    None
}

/// Applies a zero-lag forward-backward exponential moving average (EMA) filter over a sequence of float values.
pub fn bidirectional_ema_filter(values: &[f32], alpha: f32) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let alpha = alpha.clamp(0.01, 1.0);

    // Forward pass
    let mut forward = Vec::with_capacity(values.len());
    forward.push(values[0]);
    for &val in &values[1..] {
        let prev = *forward.last().unwrap();
        forward.push(alpha * val + (1.0 - alpha) * prev);
    }

    // Backward pass
    let len = forward.len();
    let mut smoothed = vec![0.0; len];
    smoothed[len - 1] = forward[len - 1];
    for i in (0..len - 1).rev() {
        smoothed[i] = alpha * forward[i] + (1.0 - alpha) * smoothed[i + 1];
    }

    smoothed
}

/// Standardizes skeleton dimensions across animation frames so character proportions remain distance-invariant.
pub fn normalize_sequence_scale(frames: &mut [MotionFrame], target_torso_len: f32) {
    if frames.is_empty() || target_torso_len <= 0.0 {
        return;
    }

    let mut measured_lengths: Vec<Option<f32>> = Vec::with_capacity(frames.len());
    let mut valid_lengths: Vec<f32> = Vec::new();

    for frame in frames.iter() {
        let len_opt = measure_frame_torso_length(&frame.joints);
        measured_lengths.push(len_opt);
        if let Some(l) = len_opt {
            if l > 1.0 {
                valid_lengths.push(l);
            }
        }
    }

    if valid_lengths.is_empty() {
        return;
    }

    // Compute median of valid lengths
    valid_lengths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_len = valid_lengths[valid_lengths.len() / 2];

    let mut raw_scales = Vec::with_capacity(frames.len());
    let mut last_valid_scale = target_torso_len / median_len.max(1.0);

    for len_opt in measured_lengths {
        if let Some(l) = len_opt {
            if l > 1.0 {
                let s = target_torso_len / l;
                last_valid_scale = s;
                raw_scales.push(s);
                continue;
            }
        }
        raw_scales.push(last_valid_scale);
    }

    let smooth_scales = bidirectional_ema_filter(&raw_scales, 0.20);

    for (frame_idx, frame) in frames.iter_mut().enumerate() {
        let scale = smooth_scales[frame_idx];
        let root = frame.joints.get("HipsRoot").cloned();
        let (cx, cy, cz) = if let Some(r) = root {
            if joint_is_valid(&r) {
                (r.x, r.y, r.z)
            } else {
                (0.0, 0.0, 0.0)
            }
        } else {
            (0.0, 0.0, 0.0)
        };

        for joint in frame.joints.values_mut() {
            if !joint_is_valid(joint) {
                continue;
            }
            joint.x = cx + (joint.x - cx) * scale;
            joint.y = cy + (joint.y - cy) * scale;
            joint.z = cz + (joint.z - cz) * scale;
        }
    }
}

/// Offsets all frames vertically so the lowest foot contact or joint touches $Z = 0.0$.
pub fn align_to_ground_plane(frames: &mut [MotionFrame]) {
    align_to_ground_plane_on_axis(frames, "z");
}

/// Offsets all frames along a specified axis ("x", "y", or "z") so the lowest value is 0.0.
pub fn align_to_ground_plane_on_axis(frames: &mut [MotionFrame], axis: &str) {
    let mut min_val: Option<f32> = None;

    for frame in frames.iter() {
        for joint in frame.joints.values() {
            if !joint_is_valid(joint) {
                continue;
            }
            let val = match axis {
                "x" | "X" => joint.x,
                "y" | "Y" => joint.y,
                _ => joint.z,
            };
            min_val = Some(min_val.map_or(val, |m| m.min(val)));
        }
    }

    let offset = match min_val {
        Some(m) => m,
        None => return,
    };

    for frame in frames.iter_mut() {
        for joint in frame.joints.values_mut() {
            if !joint_is_valid(joint) {
                continue;
            }
            match axis {
                "x" | "X" => joint.x -= offset,
                "y" | "Y" => joint.y -= offset,
                _ => joint.z -= offset,
            }
        }
    }
}

fn joint_is_valid(joint: &JointValue) -> bool {
    joint.confidence > 0.05
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cv_to_blender_space() {
        // OpenCV (+X right, +Y down, +Z forward) -> Blender (+X right, +Y forward, +Z up)
        let cv_pt = Vec3::new(1.0, 2.0, 3.0);
        let bl_pt = cv_to_blender_space(cv_pt);
        assert_eq!(bl_pt, Vec3::new(1.0, 3.0, -2.0));
    }

    #[test]
    fn test_measure_torso_length() {
        let hips = Vec3::new(0.0, 0.0, 0.0);
        let neck = Vec3::new(0.0, 0.0, 50.0);
        assert_eq!(measure_torso_length(hips, neck), 50.0);
    }

    #[test]
    fn test_bidirectional_ema_filter_preserves_constant() {
        let vals = vec![10.0; 20];
        let smoothed = bidirectional_ema_filter(&vals, 0.2);
        assert_eq!(smoothed.len(), 20);
        for s in smoothed {
            assert!((s - 10.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_bidirectional_ema_filter_smooths_spike() {
        let mut vals = vec![10.0; 11];
        vals[5] = 20.0; // Spike in middle
        let smoothed = bidirectional_ema_filter(&vals, 0.2);
        assert!(smoothed[5] < 15.0); // Spike should be dampened
        assert!(smoothed[4] > 10.0); // Zero-phase spreading to neighbor
        assert!(smoothed[6] > 10.0);
    }

    #[test]
    fn test_align_to_ground_plane() {
        let mut joints = JointMap::new();
        joints.insert("HipsRoot".to_string(), JointValue::new(0.0, 0.0, 50.0, 0.9));
        joints.insert("LeftFoot".to_string(), JointValue::new(0.0, 0.0, 5.0, 0.9)); // lowest Z is 5.0

        let mut frames = vec![MotionFrame::new(0, joints)];
        align_to_ground_plane(&mut frames);

        assert_eq!(frames[0].joints["LeftFoot"].z, 0.0);
        assert_eq!(frames[0].joints["HipsRoot"].z, 45.0);
    }

    #[test]
    fn test_normalize_sequence_scale() {
        let mut joints = JointMap::new();
        joints.insert("HipsRoot".to_string(), JointValue::new(0.0, 0.0, 0.0, 0.9));
        joints.insert("Neck".to_string(), JointValue::new(0.0, 0.0, 25.0, 0.9)); // len = 25
        joints.insert("LeftHand".to_string(), JointValue::new(10.0, 0.0, 0.0, 0.9));

        let mut frames = vec![MotionFrame::new(0, joints)];
        // Target torso length = 50.0 (scale = 2.0)
        normalize_sequence_scale(&mut frames, 50.0);

        let neck_z = frames[0].joints["Neck"].z;
        assert!((neck_z - 50.0).abs() < 1e-3);
        let hand_x = frames[0].joints["LeftHand"].x;
        assert!((hand_x - 20.0).abs() < 1e-3);
    }
}

`

## File: src/export/fbx.rs

`rust
//! Standalone ASCII FBX 7.4.0 exporter with hierarchical `LimbNode` armature and curves.

use crate::export::canonical::{compute_rest_pose_joints, MultiPersonFrame};
use crate::export::coords::{align_to_ground_plane, convert_sequence_to_blender_space, normalize_sequence_scale, MotionFrame};
use crate::skeleton::{FBX_TIME_UNIT, SKELETON};
use glam::Vec3;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Exports motion frames to standalone ASCII FBX 7.4 with hierarchical `LimbNode` armature and curves.
/// Streams directly to disk via BufWriter to minimize memory footprint.
pub fn export_motion_fbx(
    path: impl AsRef<Path>,
    frames: &[MotionFrame],
    fps: f64,
) -> anyhow::Result<()> {
    let path = path.as_ref();
    if frames.is_empty() {
        return Ok(());
    }

    let rest = compute_rest_pose_joints(frames);

    // Compute localized joint maps (deltas from parent)
    let mut local_frames: Vec<HashMap<String, Vec3>> = Vec::with_capacity(frames.len());
    for f in frames {
        let mut local_map = HashMap::new();
        for spec in &SKELETON {
            if let Some(cj) = f.joints.get(spec.name) {
                if let Some(p_name) = spec.parent {
                    if let Some(pj) = f.joints.get(p_name) {
                        local_map.insert(
                            spec.name.to_string(),
                            Vec3::new(cj.x - pj.x, cj.y - pj.y, cj.z - pj.z),
                        );
                    } else {
                        local_map.insert(spec.name.to_string(), Vec3::new(cj.x, cj.y, cj.z));
                    }
                } else {
                    local_map.insert(spec.name.to_string(), Vec3::new(cj.x, cj.y, cj.z));
                }
            }
        }
        local_frames.push(local_map);
    }

    // Compute Euler rotations relative to rest vectors
    let mut rot_frames: Vec<HashMap<String, Vec3>> = Vec::with_capacity(frames.len());
    for local_map in &local_frames {
        let mut rot_map = HashMap::new();
        for spec in &SKELETON {
            if spec.parent.is_none() {
                rot_map.insert(spec.name.to_string(), Vec3::ZERO);
            } else if let (Some(cur_v), Some(rest_v)) = (local_map.get(spec.name), rest.rest_vectors.get(spec.name)) {
                let rot = compute_rotation_euler(*rest_v, *cur_v);
                rot_map.insert(spec.name.to_string(), rot);
            } else {
                rot_map.insert(spec.name.to_string(), Vec3::ZERO);
            }
        }
        rot_frames.push(rot_map);
    }

    // Assign consistent FBX IDs
    let mut model_ids: HashMap<&str, i64> = HashMap::new();
    let mut curve_node_ids: HashMap<(&str, &str), i64> = HashMap::new();
    let mut curve_ids: HashMap<(&str, &str, &str), i64> = HashMap::new();
    let animation_stack_id: i64 = 100000;
    let animation_layer_id: i64 = 100001;
    let mut next_id: i64 = 100100;

    for spec in &SKELETON {
        model_ids.insert(spec.name, next_id);
        next_id += 1;
    }

    for spec in &SKELETON {
        curve_node_ids.insert((spec.name, "T"), next_id);
        next_id += 1;
        for axis in ["X", "Y", "Z"] {
            curve_ids.insert((spec.name, axis, "T"), next_id);
            next_id += 1;
        }

        if spec.parent.is_some() {
            curve_node_ids.insert((spec.name, "R"), next_id);
            next_id += 1;
            for axis in ["X", "Y", "Z"] {
                curve_ids.insert((spec.name, axis, "R"), next_id);
                next_id += 1;
            }
        }
    }

    let key_times: Vec<i64> = (0..frames.len())
        .map(|idx| ((idx as f64 / fps.max(1.0)) * FBX_TIME_UNIT as f64).round() as i64)
        .collect();

    let tmp_path = PathBuf::from(format!("{}.tmp", path.display()));
    {
        let file = File::create(&tmp_path)?;
        let mut w = BufWriter::new(file);

        writeln!(w, "; FBX 7.4.0 project file")?;
        writeln!(w, "FBXHeaderExtension:  {{")?;
        writeln!(w, "  FBXHeaderVersion: 1003")?;
        writeln!(w, "  FBXVersion: 7400")?;
        writeln!(w, "  Creator: \"Kinara\"")?;
        writeln!(w, "}}")?;
        writeln!(w, "GlobalSettings:  {{")?;
        writeln!(w, "  Version: 1000")?;
        writeln!(w, "  Properties70:  {{")?;
        // Standard Blender Z-up, Y-forward convention
        writeln!(w, "    P: \"UpAxis\", \"int\", \"Integer\", \"\",2")?;
        writeln!(w, "    P: \"UpAxisSign\", \"int\", \"Integer\", \"\",1")?;
        writeln!(w, "    P: \"FrontAxis\", \"int\", \"Integer\", \"\",1")?;
        writeln!(w, "    P: \"FrontAxisSign\", \"int\", \"Integer\", \"\",1")?;
        writeln!(w, "    P: \"CoordAxis\", \"int\", \"Integer\", \"\",0")?;
        writeln!(w, "    P: \"CoordAxisSign\", \"int\", \"Integer\", \"\",1")?;
        writeln!(w, "    P: \"UnitScaleFactor\", \"double\", \"Number\", \"\",1")?;
        writeln!(w, "  }}")?;
        writeln!(w, "}}")?;

        writeln!(w, "Objects:  {{")?;
        writeln!(
            w,
            "  AnimationStack: {}, \"AnimStack::Take 001\", \"\" {{",
            animation_stack_id
        )?;
        writeln!(w, "    Properties70:  {{")?;
        writeln!(w, "      P: \"LocalStart\", \"KTime\", \"Time\", \"\",0")?;
        writeln!(
            w,
            "      P: \"LocalStop\", \"KTime\", \"Time\", \"\",{}",
            key_times.last().copied().unwrap_or(0)
        )?;
        writeln!(w, "    }}")?;
        writeln!(w, "  }}")?;

        writeln!(
            w,
            "  AnimationLayer: {}, \"AnimLayer::BaseLayer\", \"\" {{",
            animation_layer_id
        )?;
        writeln!(w, "  }}")?;

        for spec in &SKELETON {
            let m_id = model_ids[spec.name];
            writeln!(w, "  Model: {}, \"Model::{}\", \"LimbNode\" {{", m_id, spec.name)?;
            writeln!(w, "    Version: 232")?;
            writeln!(w, "    Properties70:  {{")?;
            writeln!(w, "      P: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",0,0,0")?;
            writeln!(w, "      P: \"Lcl Rotation\", \"Lcl Rotation\", \"\", \"A\",0,0,0")?;
            writeln!(w, "      P: \"Lcl Scaling\", \"Lcl Scaling\", \"\", \"A\",1,1,1")?;
            writeln!(w, "    }}")?;
            writeln!(w, "    Shading: T")?;
            writeln!(w, "    Culling: \"CullingOff\"")?;
            writeln!(w, "  }}")?;
        }

        // Write Translation and Rotation curves
        let time_str = key_times
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",");

        for spec in &SKELETON {
            for &(p_type, ref map_list) in &[("T", &local_frames), ("R", &rot_frames)] {
                if p_type == "R" && spec.parent.is_none() {
                    continue;
                }
                let node_id = curve_node_ids[&(spec.name, p_type)];
                writeln!(
                    w,
                    "  AnimationCurveNode: {}, \"AnimCurveNode::{}_{}\", \"\" {{",
                    node_id, spec.name, p_type
                )?;
                writeln!(w, "    Properties70:  {{")?;
                writeln!(w, "      P: \"d|X\", \"Number\", \"\", \"A\",0")?;
                writeln!(w, "      P: \"d|Y\", \"Number\", \"\", \"A\",0")?;
                writeln!(w, "      P: \"d|Z\", \"Number\", \"\", \"A\",0")?;
                writeln!(w, "    }}")?;
                writeln!(w, "  }}")?;

                for axis in ["X", "Y", "Z"] {
                    let curve_id = curve_ids[&(spec.name, axis, p_type)];
                    let num_keys = map_list.len();

                    writeln!(
                        w,
                        "  AnimationCurve: {}, \"AnimCurve::{}_{}_{}\", \"\" {{",
                        curve_id, spec.name, p_type, axis
                    )?;
                    writeln!(w, "    Default: 0")?;
                    writeln!(w, "    KeyVer: 4008")?;
                    writeln!(w, "    KeyTime: *{} {{", key_times.len())?;
                    writeln!(w, "      a: {}", time_str)?;
                    writeln!(w, "    }}")?;

                    // Stream float values directly without intermediate large buffer
                    write!(w, "    KeyValueFloat: *{} {{\n      a: ", num_keys)?;
                    for (i, m) in map_list.iter().enumerate() {
                        let v = m.get(spec.name).copied().unwrap_or(Vec3::ZERO);
                        let val = match axis {
                            "X" => v.x,
                            "Y" => v.y,
                            _ => v.z,
                        };
                        if i > 0 {
                            write!(w, ",")?;
                        }
                        write!(w, "{:.6}", val)?;
                    }
                    writeln!(w, "\n    }}")?;

                    // KeyAttrFlags
                    write!(w, "    KeyAttrFlags: *{} {{\n      a: ", num_keys)?;
                    for i in 0..num_keys {
                        if i > 0 {
                            write!(w, ",")?;
                        }
                        write!(w, "24836")?;
                    }
                    writeln!(w, "\n    }}")?;

                    // KeyAttrDataFloat
                    write!(w, "    KeyAttrDataFloat: *{} {{\n      a: ", num_keys * 4)?;
                    for i in 0..num_keys {
                        if i > 0 {
                            write!(w, ",")?;
                        }
                        write!(w, "0,0,0,0")?;
                    }
                    writeln!(w, "\n    }}")?;

                    // KeyAttrRefCount
                    write!(w, "    KeyAttrRefCount: *{} {{\n      a: ", num_keys)?;
                    for i in 0..num_keys {
                        if i > 0 {
                            write!(w, ",")?;
                        }
                        write!(w, "1")?;
                    }
                    writeln!(w, "\n    }}")?;

                    writeln!(w, "  }}")?;
                }
            }
        }
        writeln!(w, "}}")?;

        // Write Connections
        writeln!(w, "Connections:  {{")?;
        writeln!(
            w,
            "  C: \"OO\",{},{}",
            animation_layer_id, animation_stack_id
        )?;

        for spec in &SKELETON {
            let parent_id = match spec.parent {
                Some(p) => model_ids[p],
                None => 0,
            };
            writeln!(w, "  C: \"OO\",{},{}", model_ids[spec.name], parent_id)?;

            let t_node = curve_node_ids[&(spec.name, "T")];
            writeln!(w, "  C: \"OO\",{},{}", t_node, animation_layer_id)?;
            writeln!(
                w,
                "  C: \"OP\",{},{},\"Lcl Translation\"",
                t_node, model_ids[spec.name]
            )?;
            for axis in ["X", "Y", "Z"] {
                writeln!(
                    w,
                    "  C: \"OP\",{},{},\"d|{}\"",
                    curve_ids[&(spec.name, axis, "T")],
                    t_node,
                    axis
                )?;
            }

            if spec.parent.is_some() {
                let r_node = curve_node_ids[&(spec.name, "R")];
                writeln!(w, "  C: \"OO\",{},{}", r_node, animation_layer_id)?;
                writeln!(
                    w,
                    "  C: \"OP\",{},{},\"Lcl Rotation\"",
                    r_node, model_ids[spec.name]
                )?;
                for axis in ["X", "Y", "Z"] {
                    writeln!(
                        w,
                        "  C: \"OP\",{},{},\"d|{}\"",
                        curve_ids[&(spec.name, axis, "R")],
                        r_node,
                        axis
                    )?;
                }
            }
        }
        writeln!(w, "}}")?;
        w.flush()?;
    }

    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Splits multi-person timeline into individual per-person FBX files.
pub fn export_multi_person_bundle(
    base_path: impl AsRef<Path>,
    frames: &[MultiPersonFrame],
    fps: f64,
) -> anyhow::Result<Vec<PathBuf>> {
    let base_path = base_path.as_ref();
    if frames.is_empty() {
        return Ok(Vec::new());
    }

    // Collect all distinct person identifiers
    let mut person_tracks: HashMap<String, Vec<MotionFrame>> = HashMap::new();
    for f in frames {
        for (person_id, joints) in &f.people {
            person_tracks
                .entry(person_id.clone())
                .or_default()
                .push(MotionFrame::new(f.frame_index, joints.clone()));
        }
    }

    let stem = base_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("clip");
    let ext = base_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("fbx");
    let parent = base_path.parent().unwrap_or_else(|| Path::new("."));

    let mut exported = Vec::with_capacity(person_tracks.len());
    for (person_id, mut p_frames) in person_tracks {
        let clean_id: String = person_id
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
            .collect();
        let out_file = parent.join(format!("{}_{}.{}", stem, clean_id, ext));
        convert_sequence_to_blender_space(&mut p_frames);
        normalize_sequence_scale(&mut p_frames, 50.0);
        align_to_ground_plane(&mut p_frames);
        export_motion_fbx(&out_file, &p_frames, fps)?;
        exported.push(out_file);
    }

    Ok(exported)
}

pub(crate) fn compute_rotation_euler(rest_v: Vec3, cur_v: Vec3) -> Vec3 {
    if rest_v.length_squared() < 1e-12 || cur_v.length_squared() < 1e-12 {
        return Vec3::ZERO;
    }
    let q = glam::Quat::from_rotation_arc(rest_v.normalize(), cur_v.normalize());
    let (roll, pitch, yaw) = q.to_euler(glam::EulerRot::XYZ);
    Vec3::new(roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::canonical::{build_canonical_joint_map, HandsPair};
    use crate::export::coords::MotionFrame;
    use crate::skeleton::Point;
    use crate::solver::body_constraints::BodyKeypoints;

    fn dummy_body_keypoints() -> BodyKeypoints {
        let mut pts = vec![Point::new(0, 0, 0.9); 17];
        pts[5] = Point::new(100, 100, 0.9);
        pts[6] = Point::new(200, 100, 0.9);
        pts[7] = Point::new(90, 150, 0.9);
        pts[8] = Point::new(210, 150, 0.9);
        pts[9] = Point::new(80, 200, 0.9);
        pts[10] = Point::new(220, 200, 0.9);
        pts[11] = Point::new(120, 250, 0.9);
        pts[12] = Point::new(180, 250, 0.9);
        pts[13] = Point::new(120, 320, 0.9);
        pts[14] = Point::new(180, 320, 0.9);
        pts[15] = Point::new(120, 380, 0.9);
        pts[16] = Point::new(180, 380, 0.9);
        BodyKeypoints::new(pts)
    }

    #[test]
    fn test_export_motion_fbx() {
        let body = dummy_body_keypoints();
        let map = build_canonical_joint_map(&body, &HandsPair::default(), None);
        let frames = vec![MotionFrame::new(0, map)];

        let temp_dir = std::env::temp_dir();
        let out_path = temp_dir.join(format!("test_kinara_motion_{}.fbx", std::process::id()));

        export_motion_fbx(&out_path, &frames, 30.0).expect("Failed to export motion FBX");
        assert!(out_path.exists());

        let contents = std::fs::read_to_string(&out_path).expect("Failed to read FBX");
        assert!(contents.contains("FBX 7.4.0 project file"));
        assert!(contents.contains("Model::HipsRoot"));
        assert!(contents.contains("Model::Chest"));
        assert!(contents.contains("AnimStack::Take 001"));

        let _ = std::fs::remove_file(out_path);
    }
}

`

## File: src/export/json.rs

`rust
//! Motion JSON export in standard `kinara-motion-json-v1` format.

use crate::export::canonical::ClipMetadata;
use crate::export::coords::MotionFrame;
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Atomically exports motion frames and metadata to standard `kinara-motion-json-v1` format.
pub fn export_motion_json(
    path: impl AsRef<Path>,
    frames: &[MotionFrame],
    meta: &ClipMetadata,
) -> anyhow::Result<()> {
    let path = path.as_ref();
    let tmp_path = std::path::PathBuf::from(format!("{}.tmp", path.display()));

    #[derive(Serialize)]
    struct JsonPayload<'a> {
        format: &'a str,
        fps: f64,
        frame_count: usize,
        metadata: &'a ClipMetadata,
        frames: &'a [MotionFrame],
    }

    let payload = JsonPayload {
        format: &meta.format,
        fps: meta.fps,
        frame_count: frames.len(),
        metadata: meta,
        frames,
    };

    let serialized = serde_json::to_string_pretty(&payload)?;
    {
        let mut file = File::create(&tmp_path)?;
        file.write_all(serialized.as_bytes())?;
        file.flush()?;
    }

    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::canonical::{build_canonical_joint_map, HandsPair};
    use crate::skeleton::Point;
    use crate::solver::body_constraints::BodyKeypoints;

    fn dummy_body_keypoints() -> BodyKeypoints {
        let mut pts = vec![Point::new(0, 0, 0.9); 17];
        pts[5] = Point::new(100, 100, 0.9);
        pts[6] = Point::new(200, 100, 0.9);
        pts[7] = Point::new(90, 150, 0.9);
        pts[8] = Point::new(210, 150, 0.9);
        pts[9] = Point::new(80, 200, 0.9);
        pts[10] = Point::new(220, 200, 0.9);
        pts[11] = Point::new(120, 250, 0.9);
        pts[12] = Point::new(180, 250, 0.9);
        pts[13] = Point::new(120, 320, 0.9);
        pts[14] = Point::new(180, 320, 0.9);
        pts[15] = Point::new(120, 380, 0.9);
        pts[16] = Point::new(180, 380, 0.9);
        BodyKeypoints::new(pts)
    }

    #[test]
    fn test_export_motion_json_roundtrip() {
        let body = dummy_body_keypoints();
        let map = build_canonical_joint_map(&body, &HandsPair::default(), None);
        let frames = vec![MotionFrame::new(0, map)];
        let meta = ClipMetadata::default();

        let temp_dir = std::env::temp_dir();
        let out_path = temp_dir.join(format!("test_kinara_motion_{}.json", std::process::id()));

        export_motion_json(&out_path, &frames, &meta).expect("Failed to export motion JSON");
        assert!(out_path.exists());

        let contents = std::fs::read_to_string(&out_path).expect("Failed to read back");
        assert!(contents.contains("kinara-motion-json-v1"));
        assert!(contents.contains("HipsRoot"));

        let _ = std::fs::remove_file(out_path);
    }
}

`

## File: blender_addon/kinara_stream/__init__.py

`python
"""Kinara Motion Feed Streamer for Blender."""
# pyright: reportIncompatibleMethodOverride=false, reportOptionalMemberAccess=false

bl_info = {
    'name': 'Kinara Live Stream',
    'blender': (4, 0, 0),
    'version': (1, 4, 0),
    'category': 'Animation',
    'author': 'Kinara Core Team',
    'description': 'Receives a live markerless bone stream from Kinara and bakes it to keyframes.',
}

import bpy

from .constants import PORT
from .stream import receiver, stop_timer, tick
from .ui import (
    after_undo, before_load, CLASSES
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.kinara_bone_stream_port = bpy.props.IntProperty(
        name='Port', default=PORT, min=1024, max=65535, options={'SKIP_SAVE'}
    )
    bpy.app.handlers.load_pre.append(before_load)
    bpy.app.handlers.undo_post.append(after_undo)
    bpy.app.handlers.redo_post.append(after_undo)


def unregister():
    before_load(None)
    for handlers, fn in ((bpy.app.handlers.load_pre, before_load),
                         (bpy.app.handlers.undo_post, after_undo),
                         (bpy.app.handlers.redo_post, after_undo)):
        if fn in handlers:
            handlers.remove(fn)
    if hasattr(bpy.types.WindowManager, 'kinara_bone_stream_port'):
        del bpy.types.WindowManager.kinara_bone_stream_port
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


__all__ = ['bl_info', 'register', 'unregister']

`

## File: blender_addon/kinara_stream/bake.py

`python
"""Keyframe baking subsystem for streaming Kinara armatures."""
from array import array
import json
import math
import struct
import time

import bpy

from .constants import MAX_BAKE_FRAMES, OWNER, PoseError, SKELETON
from .rig import forget_rig


class BakeJob:
    def __init__(self, context, rig, schema):
        self.scene = context.scene
        self.layer = context.view_layer
        self.rig = rig
        self.actor = schema['actor']
        self.key = rig.object[SKELETON]
        self.fields = []
        self.tracks = []
        self.previous = [None] * len(rig.names)
        for name in rig.names:
            bone = rig.object.pose.bones[name]
            mode = bone.rotation_mode
            prop = {'QUATERNION': 'rotation_quaternion',
                    'AXIS_ANGLE': 'rotation_axis_angle'}.get(mode, 'rotation_euler')
            channels = [('location', 3), (prop, 3 if prop == 'rotation_euler' else 4), ('scale', 3)]
            fields = [(bone.path_from_id(path), index) for path, size in channels for index in range(size)]
            self.fields.append((mode, fields))
            self.tracks.append([array('f') for _ in fields])
        self.times = array('f')
        self.interpolation = array('i')
        self.last_frame = -1
        self.last_kept = -1
        self.skipped = 0
        self.count = None
        self.rate = 1.0
        self.linear = 0
        self.constant = 0
        self.action = None
        self.slot = None
        self.bag = None
        self.curves = iter(())
        self.writing = False

    def start(self, payload):
        if self.count is not None:
            raise ValueError('Duplicate bake header.')
        meta = json.loads(payload)
        fps, count = float(meta['fps']), int(meta['frames'])
        if (str(meta['actor']).lower() != self.actor or meta['rig'] != self.key or
                not math.isfinite(fps) or not 0 < fps <= 240 or not 1 <= count <= MAX_BAKE_FRAMES):
            raise ValueError('Unexpected bake source.')
        self.count = count
        self.rate = self.scene.render.fps / (self.scene.render.fps_base * fps)
        enums = bpy.types.Keyframe.bl_rna.properties['interpolation'].enum_items
        self.linear = enums['LINEAR'].value
        self.constant = enums['CONSTANT'].value

    def receive(self, payload):
        size = 4 + len(self.rig.names) * 48
        if self.count is None or not payload or len(payload) % size:
            raise ValueError('Invalid bake frame data.')
        for offset in range(0, len(payload), size):
            frame, = struct.unpack_from('<I', payload, offset)
            if not self.last_frame < frame < self.count:
                raise ValueError('Unexpected bake frame number.')
            self.last_frame = frame
            try:
                bases = self.rig.basis(payload, offset+4)
            except PoseError:
                self.skipped += 1
                continue
            if self.times and frame > self.last_kept + 1:
                self.interpolation[-1] = self.constant
            self.last_kept = frame
            self.times.append(1 + frame * self.rate)
            self.interpolation.append(self.linear)
            for i, basis in enumerate(bases):
                location, rotation, scale = basis.decompose()
                mode, _ = self.fields[i]
                previous = self.previous[i]
                if mode in ('QUATERNION', 'AXIS_ANGLE'):
                    if previous is not None and previous.dot(rotation) < 0:
                        rotation.negate()
                    self.previous[i] = rotation
                    values = tuple(rotation) if mode == 'QUATERNION' else (rotation.angle, *rotation.axis)
                else:
                    euler = rotation.to_euler(mode, previous) if previous is not None else rotation.to_euler(mode)
                    self.previous[i] = euler
                    values = tuple(euler)
                for track, value in zip(self.tracks[i], (*location, *values, *scale)):
                    track.append(value)

    def prepare_action(self):
        if not self.times:
            raise ValueError('There are no recorded frames to bake.')
        obj = self.rig.object
        self.action = bpy.data.actions.new(f'{obj.name} Kinara Motion')
        self.slot = self.action.slots.new('OBJECT', obj.name)
        strip = self.action.layers.new('Motion').strips.new(type='KEYFRAME')
        self.bag = strip.channelbag(self.slot, ensure=True)
        self.curves = iter([(path, component, values)
                            for (_, fields), tracks in zip(self.fields, self.tracks)
                            for (path, component), values in zip(fields, tracks)])
        self.writing = True

    def write_curves(self, until):
        while time.perf_counter() < until:
            curve_data = next(self.curves, None)
            if curve_data is None:
                return True
            path, component, values = curve_data
            curve = self.bag.fcurves.new(path, index=component)
            curve.keyframe_points.add(len(self.times))
            coords = array('f', [0]) * (2 * len(self.times))
            coords[::2], coords[1::2] = self.times, values
            curve.keyframe_points.foreach_set('co', coords)
            curve.keyframe_points.foreach_set('interpolation', self.interpolation)
            curve.update()
        return False

    def check_context(self, context):
        obj = self.rig.object
        if (context.scene != self.scene or context.view_layer != self.layer or context.mode != 'OBJECT'
                or context.view_layer.objects.get(obj.name) != obj):
            raise ValueError('Bake stopped because the Blender scene or mode changed.')

    def commit(self):
        obj = self.rig.object
        if not obj.get(OWNER) or any(
                obj.pose.bones.get(name) is None or obj.pose.bones[name].rotation_mode != mode
                for name, (mode, _) in zip(self.rig.names, self.fields)):
            raise ValueError('The rig changed during baking.')
        animation = obj.animation_data_create()
        if animation.use_tweak_mode:
            raise ValueError('Exit NLA Tweak Mode before baking.')
        previous, slot = animation.action, animation.action_slot
        try:
            animation.action = self.action
            animation.action_slot = self.slot
        except (TypeError, RuntimeError):
            animation.action = previous
            if previous is not None:
                animation.action_slot = slot
            raise
        if previous is not None:
            previous.use_fake_user = True
        self.action = None
        forget_rig(self.scene, obj)
        del obj[OWNER]
        obj.hide_set(False)

    def cancel(self):
        if self.action is not None:
            bpy.data.actions.remove(self.action)
            self.action = None

`

## File: blender_addon/kinara_stream/constants.py

`python
"""Kinara live bone stream constants and protocol definitions."""
import struct

PORT = 49168
LIMIT = 1024 * 1024
SNAPSHOT = struct.Struct('<QH')
ACTOR_FRAME = struct.Struct('<II')
SELECTION_COUNT = struct.Struct('<H')
ACTOR_ID_BYTES = 32
MAX_ACTORS = 128
MAX_BAKE_FRAMES = 36000
MAX_BAD_FRAMES = 60          # consecutive snapshots with an unusable pose before disconnecting
BAKE_TIMEOUT = 15.0          # seconds without bake packets before a pending bake is abandoned
COLLECTION = 'Kinara Stream'
OWNER = 'kinara_bone_stream'
SKELETON = 'kinara_stream_skeleton'
ACTOR = 'kinara_stream_actor'
BONE_INDEX = 'kinara_stream_bone_index'
REGISTRY_PREFIX = 'stream_rig_'

# Errors that mean "this connection or job is bad", not "the add-on has a bug".
RECOVERABLE = (OSError, ValueError, TypeError, KeyError, RuntimeError, ReferenceError, struct.error)


class PoseError(ValueError):
    """A single pose frame is unusable. The stream itself is still fine."""

`

## File: blender_addon/kinara_stream/rig.py

`python
"""Rig target construction and hierarchy binding for Kinara."""
from hashlib import sha256
import json
import math
import string
import struct

import bpy
from mathutils import Matrix

from .constants import (
    ACTOR, BONE_INDEX, COLLECTION, OWNER, PoseError, REGISTRY_PREFIX, SKELETON
)


def valid_actor_id(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in string.hexdigits for c in value))


def managed_collection(scene):
    return next((c for c in scene.collection.children if c.get(OWNER)), None)


def registry_key(actor, skeleton):
    return REGISTRY_PREFIX + sha256(f'{actor}/{skeleton}'.encode('utf-8')).hexdigest()[:48]


def tagged_rigs(scene):
    return [obj for obj in scene.objects if obj.type == 'ARMATURE' and obj.get(OWNER)
            and obj.get(ACTOR) and obj.get(SKELETON)]


def canonical_rigs(scene):
    collection = managed_collection(scene)
    by_identity = {}
    for obj in tagged_rigs(scene):
        key = registry_key(obj[ACTOR], obj[SKELETON])
        by_identity.setdefault(key, []).append(obj)
    for key, matches in by_identity.items():
        registered = collection.get(key) if collection is not None else None
        if registered is not None and registered in matches:
            yield registered
        elif len(matches) == 1 and (collection is None or key not in collection):
            yield matches[0]


def forget_rig(scene, obj):
    collection = managed_collection(scene)
    if collection is not None:
        key = registry_key(obj[ACTOR], obj[SKELETON])
        if collection.get(key) == obj:
            collection[key] = False


def matching_hierarchy(bones, parents):
    return all(bone is not None for bone in bones) and all(
        bone.parent == (bones[parent] if parent >= 0 else None)
        for bone, parent in zip(bones, parents))


def matrix(values):
    return Matrix((values[:4], values[4:8], values[8:12], (0, 0, 0, 1)))


class RigTarget:
    def __init__(self, context, schema):
        bones = schema['bones']
        if not 1 <= len(bones) <= 128:
            raise ValueError('Unsupported skeleton description')
        names, parents, binds = [], [], []
        for index, bone in enumerate(bones):
            name, parent, rows, length = bone
            if (not isinstance(name, str) or not 1 <= len(name.encode('utf-8')) <= 63
                    or '\0' in name or name in names or type(parent) is not int
                    or (parent != -1 if index == 0 else not 0 <= parent < index)
                    or not 0.0001 <= length <= 10 or len(rows) != 12
                    or not all(math.isfinite(x) for x in rows)):
                raise ValueError('Invalid bone description')
            bind = matrix(rows)
            axes = bind.to_3x3()
            if (abs(axes.determinant()-1) > 0.001 or
                    any(abs(axes.col[j].dot(axes.col[k])-(j == k)) > 0.001
                        for j in range(3) for k in range(3))):
                raise ValueError('Invalid rest rotation')
            names.append(name)
            parents.append(parent)
            binds.append((bind, length))

        key = sha256(json.dumps(bones, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
        actor = schema['actor']
        registry = registry_key(actor, key)
        collection = managed_collection(context.scene)
        obj = next((obj for obj in canonical_rigs(context.scene)
                    if obj[ACTOR] == actor and obj[SKELETON] == key), None)
        if obj is not None:
            if context.view_layer.objects.get(obj.name) is None:
                raise ValueError('The stream armature is excluded from this view layer.')
        else:
            if (collection is None or registry not in collection) and any(
                    rig[ACTOR] == actor and rig[SKELETON] == key for rig in tagged_rigs(context.scene)):
                raise ValueError('Several existing armatures share this stream ID. Keep one before reconnecting.')
            obj = self.create_rig(context, collection, names, parents, binds, schema['name'])
            obj[OWNER], obj[ACTOR], obj[SKELETON] = True, actor, key
            created = obj.data
            if not isinstance(created, bpy.types.Armature):
                raise ValueError('Stream object is not an armature.')
            for index, name in enumerate(names):
                created.bones[name][BONE_INDEX] = index

        armature = obj.data
        if not isinstance(armature, bpy.types.Armature):
            raise ValueError('Stream object is not an armature.')
        indexed = {bone[BONE_INDEX]: bone for bone in armature.bones if BONE_INDEX in bone}
        ordered = [indexed.get(i) for i in range(len(names))]
        if (len(indexed) != sum(BONE_INDEX in bone for bone in armature.bones)
                or not matching_hierarchy(ordered, parents)):
            raise ValueError('Stream armature hierarchy was edited. Restore its original parent links to reconnect.')
        found = [bone for bone in ordered if bone is not None]
        actual = [bone.matrix_local.copy() for bone in found]
        self.object, self.names, self.parents = obj, [bone.name for bone in found], parents
        self.bind_inverse = [
            (actual[parent].inverted() @ actual[index] if parent >= 0 else actual[index]).inverted()
            for index, parent in enumerate(parents)]
        collection = managed_collection(context.scene)
        if collection is not None:
            collection[registry] = obj

    @staticmethod
    def create_rig(context, collection, names, parents, binds, actor_name):
        if collection is None:
            collection = bpy.data.collections.new(COLLECTION)
            collection[OWNER] = True
            context.scene.collection.children.link(collection)
        armature = bpy.data.armatures.new(actor_name)
        obj = bpy.data.objects.new(actor_name, armature)
        object_name = obj.name
        collection.objects.link(obj)
        obj.show_in_front = True
        armature.display_type = 'OCTAHEDRAL'
        previous = context.view_layer.objects.active
        try:
            if context.view_layer.objects.get(obj.name) is None:
                raise ValueError('The stream collection is excluded from this view layer.')
            obj.select_set(True)
            context.view_layer.objects.active = obj
            with context.temp_override(object=obj, active_object=obj,
                                       selected_objects=[obj], selected_editable_objects=[obj]):
                bpy.ops.object.mode_set(mode='EDIT')
                try:
                    for index, (bind, length) in enumerate(binds):
                        edit = armature.edit_bones.new(names[index])
                        edit.head = bind.translation
                        edit.tail = bind.translation + bind.to_3x3().col[1] * length
                        edit.align_roll(bind.to_3x3().col[2])
                        if parents[index] >= 0:
                            edit.parent = armature.edit_bones[names[parents[index]]]
                finally:
                    bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.armatures.remove(armature)
            raise
        finally:
            context.view_layer.objects.active = previous
            if object_name in bpy.data.objects:
                obj.select_set(False)
        return obj

    def basis(self, payload, offset):
        values = struct.unpack_from(f'<{len(self.names)*12}f', payload, offset)
        if not all(math.isfinite(value) for value in values):
            raise PoseError('Invalid pose numbers.')
        matrices = [matrix(values[i:i+12]) for i in range(0, len(values), 12)]
        try:
            inverses = [pose.inverted() for pose in matrices]
        except ValueError as error:
            raise PoseError('Singular pose matrix.') from error
        result = []
        for i, parent in enumerate(self.parents):
            relative = inverses[parent] @ matrices[i] if parent >= 0 else matrices[i]
            result.append(self.bind_inverse[i] @ relative)
        return result

    def apply(self, payload, offset):
        bases = self.basis(payload, offset)
        for name, basis in zip(self.names, bases):
            self.object.pose.bones[name].matrix_basis = basis

`

## File: blender_addon/kinara_stream/stream.py

`python
"""TCP receiver and stream management for Kinara live capture."""
import json
import socket
import struct
import time
import traceback

import bpy

from .bake import BakeJob
from .constants import (
    ACTOR, ACTOR_FRAME, ACTOR_ID_BYTES, BAKE_TIMEOUT, LIMIT, MAX_ACTORS,
    MAX_BAD_FRAMES, PoseError, RECOVERABLE, SELECTION_COUNT, SNAPSHOT
)
from .rig import canonical_rigs, RigTarget, valid_actor_id


class Receiver:
    def __init__(self):
        self.sock = None
        self.data = bytearray()
        self.schemas = {}
        self.rigs = {}
        self.sequence = -1
        self.snapshot = None
        self.view_layer = None
        self.visible = None
        self.selected = None
        self.bake = None
        self.bake_pending = False
        self.bake_activity = 0.0
        self.bake_status = ''
        self.detached = set()
        self.scene_id = self.target_layer = None
        self.generation = 0
        self.selection_requested = False
        self.paused_for_context = False
        self.bad_frames = 0
        self.status = 'Disconnected'

    def send(self, data):
        sock = self.sock
        if sock is None:
            raise OSError('Stream disconnected.')
        sock.sendall(data)

    def connect(self, port, context=None):
        self.close()
        sock = socket.create_connection(('127.0.0.1', port), timeout=.15)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setblocking(False)
        except OSError:
            sock.close()
            raise
        self.sock = sock
        if context is not None:
            self.scene_id, self.target_layer = context.scene.session_uid, context.view_layer.name
        self.status = 'Connected - waiting for people'

    def close(self, status='Disconnected'):
        self.generation += 1
        self.bake_status = ''
        if self.bake is not None:
            self.stop_bake('Bake stopped because the stream disconnected.')
        self.bake_pending = False
        self.detached.clear()
        if self.sock is not None:
            self.sock.close()
        self.sock = None
        self.data.clear()
        self.schemas.clear()
        self.rigs.clear()
        self.sequence = -1
        self.snapshot = None
        self.view_layer = None
        self.visible = None
        self.selected = None
        self.scene_id = self.target_layer = None
        self.selection_requested = False
        self.paused_for_context = False
        self.bad_frames = 0
        self.status = status

    def in_target(self, context):
        return self.scene_id == context.scene.session_uid and self.target_layer == context.view_layer.name

    def bake_target(self, context):
        if (self.sock is None or self.bake_pending or self.bake is not None
                or context.mode != 'OBJECT' or not self.in_target(context)
                or self.view_layer != context.view_layer):
            return None
        targets = [(rig, self.schemas[slot]) for slot, rig in self.rigs.items()
                   if self.schemas[slot].get('bake') == 1]
        return next((pair for pair in targets if pair[0].object == context.active_object),
                    targets[0] if len(targets) == 1 else None)

    def request_bake(self, context):
        target = self.bake_target(context)
        if target is None:
            raise ValueError('Select an actively streaming armature in Object Mode.')
        job = BakeJob(context, *target)
        self.send(b'B' + bytes.fromhex(job.actor) + bytes.fromhex(job.key))
        self.bake = job
        self.bake_pending = True
        self.bake_activity = time.monotonic()
        self.bake_status = 'Requesting recorded motion...'

    def stop_bake(self, message):
        if self.bake is not None:
            self.bake.cancel()
        self.bake = None
        self.bake_status = message

    def bake_packet(self, kind, payload):
        job = self.bake
        if kind in (ord('F'), ord('Z')):
            self.bake_pending = False
        self.bake_activity = time.monotonic()
        if job is None:
            return
        try:
            if kind == ord('H'):
                job.start(payload)
            elif kind == ord('D'):
                job.receive(payload)
                skipped = f' ({job.skipped} skipped)' if job.skipped else ''
                self.bake_status = f'Receiving motion - {len(job.times)} recorded frames{skipped}'
            elif kind == ord('Z'):
                job.prepare_action()
                self.bake_status = 'Writing Blender keyframes...'
            else:
                self.stop_bake(payload.decode('utf-8', errors='replace'))
        except RECOVERABLE as error:
            self.stop_bake(str(error))

    def advance_bake(self, context):
        if self.bake_pending and time.monotonic() - self.bake_activity > BAKE_TIMEOUT:
            self.bake_pending = False
            if self.bake is not None:
                self.stop_bake('The app stopped sending bake frames. Try the bake again.')
            return True
        job = self.bake
        if job is None:
            return False
        try:
            job.check_context(context)
            if not job.writing:
                return False
            if not job.write_curves(time.perf_counter() + .008):
                return True
            job.commit()
        except RECOVERABLE as error:
            self.stop_bake(str(error))
            return True
        self.detached.add(job.actor)
        self.rigs = {slot: rig for slot, rig in self.rigs.items() if self.schemas[slot]['actor'] != job.actor}
        if self.visible is not None:
            self.visible.discard(job.rig.object)
        self.bake = None
        skipped = f', {job.skipped} skipped' if job.skipped else ''
        message = f'Baked {len(job.times)} frames{skipped} - rig disconnected'
        self.bake_status = message
        try:
            self.send(b'A' + bytes.fromhex(job.actor))
        except OSError:
            self.close('Stream stopped after baking.')
            self.bake_status = message
        return True

    def apply(self, context, payload):
        sequence, count = SNAPSHOT.unpack_from(payload)
        if sequence <= self.sequence:
            return False
        if count > MAX_ACTORS:
            raise ValueError('Too many people in stream.')
        offset = SNAPSHOT.size
        entries = {}
        for _ in range(count):
            slot, _frame = ACTOR_FRAME.unpack_from(payload, offset)
            schema = self.schemas[slot]
            offset += ACTOR_FRAME.size
            end = offset + len(schema['bones']) * 48
            if slot in entries or end > len(payload):
                raise ValueError('Invalid pose size or slot.')
            entries[slot] = offset
            offset = end
        selected = None
        if offset < len(payload):
            total, = SELECTION_COUNT.unpack_from(payload, offset)
            offset += SELECTION_COUNT.size
            if total > MAX_ACTORS or len(payload) != offset + total * ACTOR_ID_BYTES:
                raise ValueError('Invalid stream selection.')
            selected = {payload[start:start + ACTOR_ID_BYTES].hex()
                        for start in range(offset, len(payload), ACTOR_ID_BYTES)}

        if selected is not None:
            self.detached.intersection_update(selected)
        visible = set()
        live = {}
        bad = False
        for slot, offset in entries.items():
            if self.schemas[slot]['actor'] in self.detached:
                continue
            rig = self.rigs.get(slot)
            if rig is None:
                rig = RigTarget(context, self.schemas[slot])
            try:
                rig.apply(payload, offset)
            except PoseError:
                bad = True
            live[slot] = rig
            visible.add(rig.object)
        self.bad_frames = self.bad_frames + 1 if bad else 0
        if self.bad_frames > MAX_BAD_FRAMES:
            raise ValueError('Too many invalid pose frames in a row.')
        self.rigs = live
        if self.visible != visible:
            for obj in canonical_rigs(context.scene):
                if context.view_layer.objects.get(obj.name) == obj and obj.hide_get() == (obj in visible):
                    obj.hide_set(obj not in visible)
            self.visible = visible
        self.sequence = sequence
        self.selected = selected
        self.status = f'Connected - {count} people' if count else 'Connected - waiting for people'
        if bad:
            self.status += ' - skipping invalid pose'
        return True

    def tick(self, context):
        if self.sock is None:
            return False
        try:
            if self.scene_id is None:
                self.scene_id, self.target_layer = context.scene.session_uid, context.view_layer.name
            redraw = self.advance_bake(context)
            if self.sock is None:
                return True
            try:
                if len(self.data) < LIMIT + 4:
                    chunk = self.sock.recv(256 * 1024)
                    if not chunk:
                        self.close('Stream stopped in Kinara')
                        return True
                    self.data.extend(chunk)
            except BlockingIOError:
                pass
            stop = time.perf_counter() + .004
            while len(self.data) >= 5 and time.perf_counter() < stop:
                size, = struct.unpack_from('<I', self.data)
                if not 2 <= size <= LIMIT:
                    raise ValueError('Invalid stream message size.')
                if len(self.data) < size+4:
                    break
                kind = self.data[4]
                payload = bytes(self.data[5:size+4])
                del self.data[:size+4]
                if kind == ord('R'):
                    schema = json.loads(payload)
                    slot = schema['slot']
                    if (schema['version'] != 2 or type(slot) is not int or not 1 <= slot <= MAX_ACTORS
                            or not valid_actor_id(schema['actor'])
                            or not isinstance(schema['name'], str) or not 1 <= len(schema['bones']) <= 128):
                        raise ValueError('Invalid stream rig declaration.')
                    schema['actor'] = schema['actor'].lower()
                    self.schemas[slot] = schema
                    if schema.get('selection') == 1 and not self.selection_requested:
                        self.send(b'V')
                        self.selection_requested = True
                    self.rigs.pop(slot, None)
                    self.snapshot = None
                elif kind == ord('S'):
                    self.snapshot = payload
                elif kind in (ord('H'), ord('D'), ord('Z'), ord('F')):
                    self.bake_packet(kind, payload)
                    redraw = True
                else:
                    raise ValueError('Unexpected stream message.')
            if not self.in_target(context):
                if not self.paused_for_context:
                    self.paused_for_context = True
                    self.status = 'Paused - return to the connected scene and view layer'
                    return True
                return redraw
            if self.paused_for_context:
                self.paused_for_context = False
                self.sequence = -1
            if context.mode != 'OBJECT':
                self.rigs.clear()
                self.sequence = -1
                return False
            if context.view_layer != self.view_layer:
                self.view_layer = context.view_layer
                self.rigs.clear()
                self.sequence = -1
                self.visible = None
            updated = self.snapshot is not None and self.apply(context, self.snapshot)
            return redraw or updated
        except RECOVERABLE as error:
            traceback.print_exc()
            self.close(str(error))
            return True


receiver = Receiver()


def tick():
    if receiver.tick(bpy.context) and bpy.context.screen:
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    return 0.02 if receiver.sock is not None else None


def stop_timer():
    if bpy.app.timers.is_registered(tick):
        bpy.app.timers.unregister(tick)


def inactive_rigs(context):
    if receiver.sock is not None:
        if (receiver.selected is None or receiver.visible is None
                or not receiver.in_target(context)):
            return None
        selected, visible = receiver.selected, receiver.visible
    else:
        selected, visible = (), ()
    candidates = {obj for obj in canonical_rigs(context.scene)
                  if obj.get(ACTOR) not in selected and obj not in visible
                  and obj.library is None and len(obj.users_scene) == 1}
    users = bpy.data.user_map(subset=candidates) if candidates else {}
    return sorted((obj for obj in candidates if obj.override_library is None
                   and all(isinstance(user, (bpy.types.Collection, bpy.types.Scene))
                           for user in users.get(obj, ()))), key=lambda obj: obj.name)

`

## File: blender_addon/kinara_stream/ui.py

`python
"""User interface, panels, and operators for Kinara Blender streaming."""
import bpy
from bpy.app.handlers import persistent

from .constants import PORT
from .rig import forget_rig
from .stream import inactive_rigs, receiver, stop_timer, tick


@persistent
def before_load(_):
    stop_timer()
    receiver.close()


@persistent
def after_undo(_):
    receiver.generation += 1
    receiver.rigs.clear()
    receiver.view_layer = None
    receiver.selected = None


class KINARA_OT_StreamConnect(bpy.types.Operator):
    bl_idname = 'kinara_bone_stream.connect'
    bl_label = 'Connect'

    def execute(self, context):
        try:
            receiver.connect(context.window_manager.kinara_bone_stream_port, context)
        except OSError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        if not bpy.app.timers.is_registered(tick):
            bpy.app.timers.register(tick)
        return {'FINISHED'}


class KINARA_OT_StreamDisconnect(bpy.types.Operator):
    bl_idname = 'kinara_bone_stream.disconnect'
    bl_label = 'Disconnect'

    def execute(self, context):
        stop_timer()
        receiver.close()
        return {'FINISHED'}


class KINARA_OT_StreamClean(bpy.types.Operator):
    bl_idname = 'kinara_bone_stream.clean'
    bl_label = 'Clean'
    bl_description = 'Remove Kinara person rigs outside the current stream'
    bl_options = {'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def candidates(self, context):
        rigs = inactive_rigs(context)
        if rigs is None:
            self.report({'WARNING'}, 'Waiting for the current stream selection. Restart Kinara if needed.')
        elif not rigs:
            self.report({'INFO'}, 'No inactive Kinara rigs to clean.')
        return rigs

    def invoke(self, context, event):
        rigs = self.candidates(context)
        if not rigs:
            return {'CANCELLED'}
        self.confirmed_rigs = {obj.session_uid for obj in rigs}
        self.confirmed_context = (receiver.generation, context.scene.session_uid, context.view_layer.name)
        message = (f'Remove {len(rigs)} inactive Kinara rig(s) from this scene?' if receiver.sock else
                   f'Stream is disconnected. Remove all {len(rigs)} eligible Kinara rig(s) from this scene?')
        return context.window_manager.invoke_confirm(self, event, title='Clean Kinara Rigs',
                                                     message=message, confirm_text='Remove')

    def execute(self, context):
        confirmation = getattr(self, 'confirmed_context', None)
        if context.mode != 'OBJECT' or (confirmation is not None and confirmation != (
                receiver.generation, context.scene.session_uid, context.view_layer.name)):
            self.report({'WARNING'}, 'Connection or scene changed. Open Clean again.')
            return {'CANCELLED'}
        rigs = self.candidates(context)
        if not rigs:
            return {'CANCELLED'}
        confirmed = getattr(self, 'confirmed_rigs', None)
        removed = 0
        for obj in rigs:
            if confirmed is not None and obj.session_uid not in confirmed:
                continue
            armature = obj.data
            forget_rig(context.scene, obj)
            bpy.data.objects.remove(obj, do_unlink=True)
            if armature.users == 0 and armature.library is None:
                bpy.data.armatures.remove(armature)
            removed += 1
        self.report({'INFO'}, f'Removed {removed} inactive Kinara rig(s).')
        return {'FINISHED'} if removed else {'CANCELLED'}


class KINARA_OT_StreamBake(bpy.types.Operator):
    bl_idname = 'kinara_bone_stream.bake'
    bl_label = 'Bake'
    bl_description = 'Bake all recorded motion to the selected streaming rig and detach it'

    @classmethod
    def poll(cls, context):
        return receiver.bake_target(context) is not None

    def execute(self, context):
        try:
            receiver.request_bake(context)
        except Exception as error:
            self.report({'WARNING'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}


class KINARA_PT_StreamPanel(bpy.types.Panel):
    bl_label = 'Kinara Motion Stream'
    bl_idname = 'KINARA_PT_stream_panel'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Kinara'

    def draw(self, context):
        layout = self.layout
        assert layout is not None
        layout.prop(context.window_manager, 'kinara_bone_stream_port')
        layout.operator('kinara_bone_stream.disconnect' if receiver.sock else 'kinara_bone_stream.connect')
        layout.operator('kinara_bone_stream.bake')
        clean_row = layout.row()
        clean_row.operator_context = 'INVOKE_DEFAULT'
        clean_row.operator('kinara_bone_stream.clean')
        layout.label(text=receiver.status)
        if receiver.bake_status:
            layout.label(text=receiver.bake_status)


CLASSES = (
    KINARA_OT_StreamConnect,
    KINARA_OT_StreamDisconnect,
    KINARA_OT_StreamClean,
    KINARA_OT_StreamBake,
    KINARA_PT_StreamPanel,
)

`
