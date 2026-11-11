# Kinara Rust Rewrite: Implementation Progress & Function Ledger

This document tracks the systematic porting of Kinara from Python to native Rust. Every function, struct, and algorithmic component requiring implementation is indexed below with its target Rust signature, responsibilities, and completion status.

---

## Progress Overview

| Subsystem | Module Path | Total Functions | Completed | Status |
| :--- | :--- | :---: | :---: | :--- |
| **1. Video Ingestion & IO** | `src/video/` | 6 | 0 | 🟡 Pending Start |
| **2. Inference Engine (ONNX)** | `src/inference/` | 8 | 0 | 🟡 Pending Start |
| **3. Biomechanical IK Solver** | `src/solver/` | 9 | 0 | 🟡 Pending Start |
| **4. Temporal Smoothing & Filter**| `src/solver/` | 5 | 0 | 🟡 Pending Start |
| **5. Multi-Person Tracking** | `src/tracking/` | 7 | 0 | 🟡 Pending Start |
| **6. Multi-Camera Triangulation**| `src/fusion/` | 5 | 0 | 🟡 Pending Start |
| **7. Coordinate & Scale Normalizer**| `src/export/` | 5 | 0 | 🟡 Pending Start |
| **8. Export Engine (JSON / FBX)**| `src/export/` | 5 | 0 | 🟡 Pending Start |
| **9. Configuration & Native GUI**| `src/` | 6 | 0 | 🟡 Pending Start |
| **Total** | | **56** | **0** | **0% Completed** |

---

## 1. Video Ingestion & IO (`src/video/`)

- [ ] `VideoInputSource::open(path_or_dev: &str, fps: Option<f64>) -> Result<Self>`
  - *Ref*: `camera/capture.py:VideoInputSource.__init__`
  - *Description*: Opens camera device or video file using hardware decoders (DirectShow / V4L2); validates FPS and dimensions.
  - *Signature*: `pub fn open(src: &str, fps: Option<f64>) -> anyhow::Result<Self>`

- [ ] `VideoInputSource::read_frame(&mut self) -> Result<Option<RgbImage>>`
  - *Ref*: `camera/capture.py:VideoInputSource.read`
  - *Description*: Decodes next frame into contiguous raw sRGB pixel buffer; reuses heap buffers across iterations.
  - *Signature*: `pub fn read_frame(&mut self) -> anyhow::Result<Option<image::RgbImage>>`

- [ ] `VideoInputSource::seek_frame(&mut self, frame_index: usize) -> Result<()>`
  - *Ref*: `camera/capture.py`
  - *Description*: Seeks to a specific frame timestamp for parallel chunk decoding.
  - *Signature*: `pub fn seek_frame(&mut self, index: usize) -> anyhow::Result<()>`

- [ ] `VideoOutputWriter::new(path: &Path, w: u32, h: u32, fps: f64) -> Result<Self>`
  - *Ref*: `camera/capture.py:VideoOutputWriter.__init__`
  - *Description*: Initializes H.264/NVENC video encoder stream for MP4 output.
  - *Signature*: `pub fn new(path: &Path, w: u32, h: u32, fps: f64) -> anyhow::Result<Self>`

- [ ] `VideoOutputWriter::write_frame(&mut self, frame: &RgbImage) -> Result<()>`
  - *Ref*: `camera/capture.py:VideoOutputWriter.write`
  - *Description*: Encodes and appends one annotated video frame to destination MP4.
  - *Signature*: `pub fn write_frame(&mut self, frame: &image::RgbImage) -> anyhow::Result<()>`

- [ ] `VideoOutputWriter::finish(self) -> Result<()>`
  - *Ref*: `camera/capture.py:VideoOutputWriter.close`
  - *Description*: Flushes remaining packets and writes MP4 file trailer.
  - *Signature*: `pub fn finish(self) -> anyhow::Result<()>`

---

## 2. Pose & Hand Inference Engine (`src/inference/`)

- [ ] `BodyPoseEstimator::new(model_path: &Path, backend: Backend) -> Result<Self>`
  - *Ref*: `inference/rtmpose.py:ONNXPoseHandRunner._init_rtmpose_body`
  - *Description*: Instantiates ONNX Runtime session (`ort`) with CUDA / DirectML provider and performs warmup.
  - *Signature*: `pub fn new(path: &Path, backend: ExecutionBackend) -> anyhow::Result<Self>`

- [ ] `BodyPoseEstimator::detect_body(&mut self, frame: &RgbImage) -> Result<BodyKeypointSet>`
  - *Ref*: `inference/rtmpose.py:ONNXPoseHandRunner.detect_body`
  - *Description*: Runs full-body pose inference on downscaled proxy; decodes heatmaps into 17 COCO keypoints.
  - *Signature*: `pub fn detect_body(&mut self, frame: &image::RgbImage) -> anyhow::Result<BodyKeypoints>`

- [ ] `compute_forearm_vector(elbow: Vec2, wrist: Vec2) -> (Vec2, f32)`
  - *Ref*: `utils/normalize.py:build_hand_box`
  - *Description*: Calculates normalized direction vector and Euclidean segment length between elbow and wrist.
  - *Signature*: `pub fn compute_forearm_vector(elbow: glam::Vec2, wrist: glam::Vec2) -> (glam::Vec2, f32)`

- [ ] `extract_hand_crop(frame: &RgbImage, wrist: Vec2, dir: Vec2, len: f32) -> Option<HandCrop>`
  - *Ref*: `utils/normalize.py:build_hand_box`
  - *Description*: Extracts native-resolution square pixel crop aligned along forearm vector with padding.
  - *Signature*: `pub fn extract_hand_crop(frame: &image::RgbImage, wrist: glam::Vec2, dir: glam::Vec2, len: f32) -> Option<HandCrop>`

- [ ] `HandMeshEstimator::new(model_path: &Path, backend: Backend) -> Result<Self>`
  - *Ref*: `inference/rtmpose.py:ONNXPoseHandRunner._init_onnx_hand`
  - *Description*: Loads 3D hand mesh recovery / landmark ONNX model with hardware acceleration.
  - *Signature*: `pub fn new(path: &Path, backend: ExecutionBackend) -> anyhow::Result<Self>`

- [ ] `HandMeshEstimator::detect_hand(&mut self, crop: &HandCrop) -> Result<HandKeypointSet>`
  - *Ref*: `inference/rtmpose.py:ONNXPoseHandRunner.detect_hand`
  - *Description*: Infers 21 3D hand landmarks; projects crop coordinates back to full-frame image space.
  - *Signature*: `pub fn detect_hand(&mut self, crop: &HandCrop) -> anyhow::Result<HandKeypoints>`

- [ ] `anchor_hand_to_body_wrist(hand: &mut HandKeypoints, body_wrist: Vec3)`
  - *Ref*: `utils/hand_fallback.py:anchor_hand_to_wrist`
  - *Description*: Rigidly translates hand keypoint set so the hand root coincides with tracked body wrist.
  - *Signature*: `pub fn anchor_hand_to_body_wrist(hand: &mut HandKeypoints, body_wrist: glam::Vec3)`

- [ ] `generate_fallback_hand(wrist: Vec2, dir: Vec2, len: f32) -> HandKeypoints`
  - *Ref*: `utils/hand_fallback.py:generate_default_hand`
  - *Description*: Generates anatomically proportioned synthetic rest-pose hand when detection is occluded or rejected.
  - *Signature*: `pub fn generate_fallback_hand(wrist: glam::Vec2, dir: glam::Vec2, len: f32) -> HandKeypoints`

---

## 3. Biomechanical IK & Constraint Solver (`src/solver/`)

- [ ] `HandIKSolver::enforce_constraints(raw: &HandKeypoints, forearm_dir: Vec2) -> HandKeypoints`
  - *Ref*: `utils/hand_constraints.py:enforce_hand_constraints`
  - *Description*: Orchestrates full hand kinematic pass: tendon coupling, fist clamp, hinge limits, and bone lengths.
  - *Signature*: `pub fn enforce_constraints(raw: &HandKeypoints, forearm_dir: glam::Vec2) -> HandKeypoints`

- [ ] `apply_dip_pip_tendon_coupling(chain: &mut FingerChain)`
  - *Ref*: `utils/hand_constraints.py:_enforce_chain_bend`
  - *Description*: Locks distal joint flexion to proximal joint: $\theta_{\text{DIP}} = 0.67 \times \theta_{\text{PIP}}$.
  - *Signature*: `pub fn apply_dip_pip_tendon_coupling(chain: &mut FingerChain)`

- [ ] `apply_dynamic_fist_abduction_clamp(fingers: &mut [FingerChain; 4])`
  - *Ref*: `utils/hand_constraints.py:_enforce_finger_lanes`
  - *Description*: Multiplies allowed lateral finger spread by $(1.0 - \theta_{\text{MCP}} / 90^\circ)$, preventing fist collision.
  - *Signature*: `pub fn apply_dynamic_fist_abduction_clamp(fingers: &mut [FingerChain; 4])`

- [ ] `clamp_planar_hinge(parent: Vec3, hinge: Vec3, child: Vec3, min_deg: f32, max_deg: f32) -> Vec3`
  - *Ref*: `utils/hand_constraints.py:_enforce_chain_bend`
  - *Description*: Constrains 1-DOF knuckle/joint to planar flexion, enforcing strict zero hyperextension.
  - *Signature*: `pub fn clamp_planar_hinge(p: glam::Vec3, h: glam::Vec3, c: glam::Vec3, min_deg: f32, max_deg: f32) -> glam::Vec3`

- [ ] `enforce_radial_limits(wrist: Vec2, points: &mut [Vec2], max_multipliers: &[f32])`
  - *Ref*: `utils/hand_constraints.py:_enforce_radial_limits`
  - *Description*: Prevents distal phalanges from drifting beyond anatomical radial bounds from the wrist.
  - *Signature*: `pub fn enforce_radial_limits(wrist: glam::Vec2, points: &mut [glam::Vec2], mults: &[f32])`

- [ ] `enforce_bone_lengths(parent: Vec2, child: Vec2, min_len: f32, max_len: f32) -> Vec2`
  - *Ref*: `utils/hand_constraints.py:_enforce_bone_lengths`
  - *Description*: Clamps segment distance to stay within physiological length proportions.
  - *Signature*: `pub fn enforce_bone_lengths(p: glam::Vec2, c: glam::Vec2, min_len: f32, max_len: f32) -> glam::Vec2`

- [ ] `BodyIKSolver::new(config: &SolverConfig) -> Self`
  - *Ref*: `utils/body_constraints.py:BodyKinematicConstraints.__init__`
  - *Description*: Initializes body constraint state and learned target segment length cache.
  - *Signature*: `pub fn new(config: &SolverConfig) -> Self`

- [ ] `BodyIKSolver::enforce_limb_lengths(&mut self, body: &mut BodyKeypoints)`
  - *Ref*: `utils/body_constraints.py:BodyKinematicConstraints.apply`
  - *Description*: Prevents limb stretching by re-projecting joints along learned invariant segment lengths.
  - *Signature*: `pub fn enforce_limb_lengths(&mut self, body: &mut BodyKeypoints)`

- [ ] `derive_foot_points(knee: Vec2, ankle: Vec2) -> (Vec2, Vec2)`
  - *Ref*: `utils/body_geometry.py:derive_foot_points`
  - *Description*: Extrapolates foot and toe base points from knee-to-ankle vector direction and shin length.
  - *Signature*: `pub fn derive_foot_points(knee: glam::Vec2, ankle: glam::Vec2) -> (glam::Vec2, glam::Vec2)`

---

## 4. Temporal Smoothing & Motion Filtering (`src/solver/`)

- [ ] `OneEuroFilter::new(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self`
  - *Ref*: `utils/smoothing.py:LandmarkSmoother.__init__`
  - *Description*: Initializes speed-adaptive low-pass filter for eliminating jitter without latency.
  - *Signature*: `pub fn new(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self`

- [ ] `OneEuroFilter::filter_point(&mut self, point: Vec3, dt: f32) -> Vec3`
  - *Ref*: `utils/smoothing.py:LandmarkSmoother._smooth_points`
  - *Description*: Evaluates landmark derivative speed to dynamically tune smoothing cutoff frequency.
  - *Signature*: `pub fn filter_point(&mut self, p: glam::Vec3, dt: f32) -> glam::Vec3`

- [ ] `QuaternionFilter::slerp(&self, q_prev: Quat, q_curr: Quat, alpha: f32) -> Quat`
  - *Ref*: `utils/smoothing.py`
  - *Description*: Performs spherical linear interpolation on rotation quaternions along shortest arc on $S^3$.
  - *Signature*: `pub fn slerp(&self, q0: glam::Quat, q1: glam::Quat, alpha: f32) -> glam::Quat`

- [ ] `predict_missing_landmarks(prev: &[Vec3], vel: &[Vec3], dt: f32, decay: f32) -> Vec<Vec3>`
  - *Ref*: `utils/prediction.py:predict_points`
  - *Description*: Extrapolates occluded points along constant velocity vector with confidence decay.
  - *Signature*: `pub fn predict_missing_landmarks(pts: &[glam::Vec3], vel: &[glam::Vec3], dt: f32, decay: f32) -> Vec<glam::Vec3>`

- [ ] `FootLockStabilizer::lock_planted_feet(&mut self, feet: &mut FootContacts, dt: f32)`
  - *Ref*: `utils/motion_cleanup.py:cleanup_motion_frames`
  - *Description*: Detects foot floor impact; clamps $(x, y)$ coordinates to zero velocity while planted.
  - *Signature*: `pub fn lock_planted_feet(&mut self, feet: &mut FootContacts, dt: f32)`

---

## 5. Multi-Person Tracking & Association (`src/tracking/`)

- [ ] `compute_bbox_iou(a: BoundingBox, b: BoundingBox) -> f32`
  - *Ref*: `utils/multi_person.py:_iou`
  - *Description*: Calculates intersection-over-union metric between two bounding boxes.
  - *Signature*: `pub fn compute_bbox_iou(a: BoundingBox, b: BoundingBox) -> f32`

- [ ] `TorsoColorProfiler::extract_histogram(frame: &RgbImage, torso: BoundingBox) -> ColorHistogram`
  - *Ref*: `utils/multi_person.py:_color_scores`
  - *Description*: Builds 16-bin normalized HSV color histogram of actor clothing.
  - *Signature*: `pub fn extract_histogram(frame: &image::RgbImage, torso: BoundingBox) -> ColorHistogram`

- [ ] `compare_color_histograms(a: &ColorHistogram, b: &ColorHistogram) -> f32`
  - *Ref*: `utils/multi_person.py:color_profile_similarity`
  - *Description*: Measures histogram intersection score for clothing re-identification.
  - *Signature*: `pub fn compare_color_histograms(a: &ColorHistogram, b: &ColorHistogram) -> f32`

- [ ] `MultiPersonTracker::new(config: TrackerConfig) -> Self`
  - *Ref*: `utils/multi_person.py:MultiPersonTracker.__init__`
  - *Description*: Initializes multi-person tracking manager with ByteTrack Kalman filters and track history.
  - *Signature*: `pub fn new(config: TrackerConfig) -> Self`

- [ ] `MultiPersonTracker::update(&mut self, dets: &[BodyDetection], frame: &RgbImage) -> Vec<PersonTrack>`
  - *Ref*: `utils/multi_person.py:MultiPersonTracker.update`
  - *Description*: Matches detections to existing tracks using IoU, velocity prediction, and clothing color.
  - *Signature*: `pub fn update(&mut self, dets: &[BodyDetection], frame: &image::RgbImage) -> Vec<PersonTrack>`

- [ ] `score_detection_to_track(det: &BodyDetection, track: &PersonTrack) -> f32`
  - *Ref*: `utils/multi_person.py:MultiPersonTracker._match_score`
  - *Description*: Evaluates composite association score weighting spatial overlap, motion, and color.
  - *Signature*: `pub fn score_detection_to_track(det: &BodyDetection, track: &PersonTrack) -> f32`

- [ ] `resolve_cross_person_hand_ownership(tracks: &mut [PersonTrack])`
  - *Ref*: `utils/multi_person.py:MultiPersonTracker._resolve_cross_person_hands`
  - *Description*: Disambiguates hand ownership during actor crossovers, reassigning stolen hands.
  - *Signature*: `pub fn resolve_cross_person_hand_ownership(tracks: &mut [PersonTrack])`

---

## 6. Multi-Camera Calibration & Triangulation (`src/fusion/`)

- [ ] `detect_charuco_corners(frame: &RgbImage, board: &CharucoBoard) -> Option<CharucoCorners>`
  - *Ref*: `utils/calibration.py:calibrate_cameras`
  - *Description*: Detects ChArUco checkerboard corners with sub-pixel saddle point refinement.
  - *Signature*: `pub fn detect_charuco_corners(frame: &image::RgbImage, board: &CharucoBoard) -> Option<CharucoCorners>`

- [ ] `CameraCalibrator::solve_camera_poses(&self, all_corners: &[CameraCorners]) -> Result<CameraRig>`
  - *Ref*: `utils/calibration.py:calibrate_cameras`
  - *Description*: Computes camera intrinsic matrices $\mathbf{K}$ and relative extrinsics $[\mathbf{R} \mid \mathbf{t}]$.
  - *Signature*: `pub fn solve_camera_poses(&self, corners: &[CameraCorners]) -> anyhow::Result<CameraRig>`

- [ ] `triangulate_dlt_point(points: &[(Vec2, f32)], projections: &[Mat3x4]) -> Option<Vec3>`
  - *Ref*: `utils/triangulation.py:triangulate_observation_frames`
  - *Description*: Computes 3D Cartesian coordinates via Direct Linear Transformation (DLT) with SVD.
  - *Signature*: `pub fn triangulate_dlt_point(pts: &[(glam::Vec2, f32)], mats: &[glam::Mat4]) -> Option<glam::Vec3>`

- [ ] `fuse_multiview_landmarks(views: &[CameraObservation]) -> FusedJointSet`
  - *Ref*: `utils/fusion.py:fuse_body_views`
  - *Description*: Fuses multi-view observations using confidence-weighted reprojection.
  - *Signature*: `pub fn fuse_multiview_landmarks(views: &[CameraObservation]) -> FusedJointSet`

- [ ] `compute_reprojection_error(p3d: Vec3, p2d: Vec2, proj: Mat3x4) -> f32`
  - *Ref*: `utils/triangulation.py:triangulation_metadata`
  - *Description*: Measures pixel distance between projected 3D point and observed 2D detector keypoint.
  - *Signature*: `pub fn compute_reprojection_error(p3d: glam::Vec3, p2d: glam::Vec2, proj: glam::Mat4) -> f32`

---

## 7. Coordinate Transformation & Scale Normalization (`src/export/`)

- [ ] `cv_to_blender_space(point: Vec3) -> Vec3`
  - *Ref*: `utils/exports.py:_to_world`
  - *Description*: Rotates from OpenCV camera space $(+X \text{ right}, +Y \text{ down}, +Z \text{ forward})$ to Blender $(+X \text{ right}, +Y \text{ forward}, +Z \text{ up})$.
  - *Signature*: `pub fn cv_to_blender_space(p: glam::Vec3) -> glam::Vec3`

- [ ] `measure_torso_length(hips: Vec3, neck: Vec3) -> f32`
  - *Ref*: `utils/exports.py:_measure_frame_torso_length`
  - *Description*: Measures 3D Euclidean distance between hip root and neck base.
  - *Signature*: `pub fn measure_torso_length(hips: glam::Vec3, neck: glam::Vec3) -> f32`

- [ ] `bidirectional_ema_filter(values: &[f32], alpha: f32) -> Vec<f32>`
  - *Ref*: `utils/exports.py:_smooth_scale_factors`
  - *Description*: Applies zero-lag forward-backward exponential moving average over sequence.
  - *Signature*: `pub fn bidirectional_ema_filter(vals: &[f32], alpha: f32) -> Vec<f32>`

- [ ] `normalize_sequence_scale(frames: &mut [MotionFrame], target_torso_len: f32)`
  - *Ref*: `utils/exports.py:_scale_normalize_frames`
  - *Description*: Standardizes skeleton dimensions so character proportions remain distance-invariant.
  - *Signature*: `pub fn normalize_sequence_scale(frames: &mut [MotionFrame], target_len: f32)`

- [ ] `align_to_ground_plane(frames: &mut [MotionFrame])`
  - *Ref*: `utils/exports.py:_ground_z_axis_frames`
  - *Description*: Offsets all frames vertically so lowest foot contact touches $Z = 0.0$.
  - *Signature*: `pub fn align_to_ground_plane(frames: &mut [MotionFrame])`

---

## 8. Export Pipeline (`src/export/`)

- [ ] `build_canonical_joint_map(body: &BodyKeypoints, hands: &HandKeypointsPair) -> JointMap`
  - *Ref*: `utils/exports.py:build_joint_map`
  - *Description*: Assembles body and finger landmarks into the standard 78-joint named skeleton dictionary.
  - *Signature*: `pub fn build_canonical_joint_map(body: &BodyKeypoints, hands: &HandsPair) -> JointMap`

- [ ] `compute_rest_pose_joints(frames: &[MotionFrame]) -> RestJointOffsets`
  - *Ref*: `utils/exports.py:_compute_rest_joints`
  - *Description*: Computes median segment offset vectors across the clip to generate a stable bind pose.
  - *Signature*: `pub fn compute_rest_pose_joints(frames: &[MotionFrame]) -> RestJointOffsets`

- [ ] `export_motion_json(path: &Path, frames: &[MotionFrame], meta: &ClipMetadata) -> Result<()>`
  - *Ref*: `utils/exports.py:export_motion_json`
  - *Description*: Emits atomic `kinara-motion-json-v1` file containing frames, joint trajectories, and metadata.
  - *Signature*: `pub fn export_motion_json(path: &Path, frames: &[MotionFrame], meta: &ClipMetadata) -> anyhow::Result<()>`

- [ ] `export_motion_fbx(path: &Path, frames: &[MotionFrame], fps: f64) -> Result<()>`
  - *Ref*: `utils/exports.py:export_motion_fbx`
  - *Description*: Writes standalone ASCII FBX 7.4 document with hierarchical `LimbNode` armature and curves.
  - *Signature*: `pub fn export_motion_fbx(path: &Path, frames: &[MotionFrame], fps: f64) -> anyhow::Result<()>`

- [ ] `export_multi_person_bundle(base_path: &Path, frames: &[MultiPersonFrame], fps: f64) -> Result<()>`
  - *Ref*: `utils/exports.py:export_multi_person_fbx_bundle`
  - *Description*: Splits multi-person timeline into individual per-person FBX files.
  - *Signature*: `pub fn export_multi_person_bundle(path: &Path, frames: &[MultiPersonFrame], fps: f64) -> anyhow::Result<()>`

---

## 9. Configuration, Desktop GUI & CLI (`src/`)

- [ ] `PipelineConfig::load_or_default(path: &Path) -> Self`
  - *Ref*: `core/config.py:PipelineConfig`
  - *Description*: Loads settings from `kinara.config.json` with fallback to high-performance defaults.
  - *Signature*: `pub fn load_or_default(path: &Path) -> Self`

- [ ] `PipelineConfig::save(&self, path: &Path) -> Result<()>`
  - *Ref*: `core/config.py:PipelineConfig`
  - *Description*: Persists active configuration, video paths, and UI preferences to disk.
  - *Signature*: `pub fn save(&self, path: &Path) -> anyhow::Result<()>`

- [ ] `parse_cli_arguments() -> CliArgs`
  - *Ref*: `core/cli.py:build_parser`
  - *Description*: Parses command line arguments via `clap` (derive macro), overriding config file values.
  - *Signature*: `pub fn parse_cli_arguments() -> CliArgs`

- [ ] `KinaraApp::new(cc: &eframe::CreationContext, config: PipelineConfig) -> Self`
  - *Ref*: `app/gui.rs`
  - *Description*: Initializes native desktop GUI application with egui themes and texture allocators.
  - *Signature*: `pub fn new(cc: &eframe::CreationContext, cfg: PipelineConfig) -> Self`

- [ ] `KinaraApp::update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame)`
  - *Ref*: `app/gui.rs`
  - *Description*: Renders desktop studio deck: video file picker, camera selection, live preview, and progress bar.
  - *Signature*: `impl eframe::App for KinaraApp { fn update(...) }`

- [ ] `run_cli_pipeline(config: PipelineConfig) -> Result<()>`
  - *Ref*: `app/main.py:main`
  - *Description*: Headless batch runner executing the full mocap pipeline without graphical desktop dependencies.
  - *Signature*: `pub fn run_cli_pipeline(cfg: PipelineConfig) -> anyhow::Result<()>`
