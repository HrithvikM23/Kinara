# Kinara Function Architecture & Specification

This document provides a comprehensive technical index of all functions within the Kinara motion capture engine, their responsibilities, inputs, outputs, and internal algorithms for implementation in native Rust.

---

## 1. Video Ingestion & IO Subsystem (`src/video/`)

### `VideoInputSource::open(path_or_device: &str, target_fps: Option<f64>) -> Result<Self>`
* **Responsibility**: Opens a camera feed (DirectShow / V4L2) or pre-recorded video file (MP4/MKV) using hardware-accelerated video decoders.
* **Inputs**: Path string or camera device index, optional FPS override.
* **Outputs**: Initialized `VideoInputSource` with frame width, height, detected FPS, and total frame count.
* **Internal Logic**: Probes hardware capabilities; initializes decoder context; falls back to 30.0/60.0 FPS if metadata is corrupt.

### `VideoInputSource::read_frame(&mut self) -> Result<Option<RgbImage>>`
* **Responsibility**: Decodes the next available video frame into an uncompressed, contiguous sRGB pixel buffer.
* **Inputs**: Mutable reference to `VideoInputSource`.
* **Outputs**: `Some(RgbImage)` containing 24-bit raw RGB pixels, or `None` on end-of-stream.
* **Internal Logic**: Manages frame buffer recycling to avoid dynamic heap reallocations.

### `VideoOutputWriter::new(path: &Path, width: u32, height: u32, fps: f64) -> Result<Self>`
* **Responsibility**: Initializes an MP4 video writer for preview/rendered output with H.264/NVENC encoding.
* **Inputs**: Destination file path, frame dimensions, framerate.
* **Outputs**: Initialized `VideoOutputWriter`.
* **Internal Logic**: Sets FourCC codec; validates divisible-by-2 dimensions for MPEG compliance.

### `VideoOutputWriter::write_frame(&mut self, frame: &RgbImage) -> Result<()>`
* **Responsibility**: Encodes and appends one rendered visual frame with pose overlays to the output video stream.
* **Inputs**: Reference to the annotated `RgbImage`.
* **Outputs**: Unit `Result<()>`.

---

## 2. Deep Inference & ROI Processing Subsystem (`src/inference/`)

### `BodyPoseEstimator::new(model_path: &Path, backend: ExecutionBackend) -> Result<Self>`
* **Responsibility**: Loads the full-body pose estimation model (DWPose / WholeBody ONNX) via ONNX Runtime (`ort`).
* **Inputs**: Path to ONNX model weights, target hardware backend (CUDA, DirectML, TensorRT, or CPU).
* **Outputs**: Initialized `BodyPoseEstimator` session.
* **Internal Logic**: Allocates GPU execution providers, establishes memory arenas, and runs a zero-tensor warmup pass.

### `BodyPoseEstimator::detect_body(&mut self, frame: &RgbImage) -> Result<BodyKeypointSet>`
* **Responsibility**: Detects 17 canonical COCO keypoints (plus optional whole-body face/foot points) from a downscaled frame.
* **Inputs**: Full or downscaled RGB image frame.
* **Outputs**: `BodyKeypointSet` containing $(x, y, \text{confidence})$ for all body joints.
* **Internal Logic**: Letterbox padding to $640 \times 640$; tensor normalization; heatmap argmax decoding with sub-pixel Gaussian peak refinement.

### `extract_hand_roi(body_points: &BodyKeypointSet, side: HandSide, image_dims: (u32, u32)) -> Option<BoundingBox>`
* **Responsibility**: Computes an adaptive, forearm-oriented bounding crop around the wrist at native 1:1 camera resolution.
* **Inputs**: Detected body keypoints, target side (`Left` or `Right`), full-resolution image dimensions.
* **Outputs**: Square pixel `BoundingBox` enclosing the hand with context padding.
* **Internal Logic**: Computes vector from elbow to wrist ($\vec{v} = \mathbf{p}_{\text{wrist}} - \mathbf{p}_{\text{elbow}}$); establishes box side length proportional to forearm length ($1.1 \times \|\vec{v}\|$); shifts center forward along $\vec{v}$ to encapsulate fingertips; clamps to image boundary.

### `HandMeshEstimator::new(model_path: &Path, backend: ExecutionBackend) -> Result<Self>`
* **Responsibility**: Loads the 3D hand mesh recovery model (WiLoR / Hand Landmark ONNX) for sub-pixel finger tracking.
* **Inputs**: Path to hand model weights, execution backend.
* **Outputs**: Initialized `HandMeshEstimator` session.

### `HandMeshEstimator::detect_hand(&mut self, crop: &RgbImage) -> Result<HandKeypointSet>`
* **Responsibility**: Computes 21 3D hand keypoints (wrist, MCP, PIP, DIP, tip per finger) from the native-resolution crop.
* **Inputs**: Native-resolution square hand crop.
* **Outputs**: `HandKeypointSet` containing 21 joint coordinates with confidence and local depth.
* **Internal Logic**: Normalizes crop to $256 \times 256$; runs inference; transforms local crop coordinates back into full-frame image space.

---

## 3. Biomechanical IK & Constraint Solver (`src/solver/`)

### `HandIKSolver::enforce_constraints(raw_hand: &HandKeypointSet, forearm_vec: Vec2) -> HandKeypointSet`
* **Responsibility**: Applies anatomical kinematic rules to raw hand detections to guarantee physically valid finger poses.
* **Inputs**: Raw 21-point hand detection, forearm direction vector.
* **Outputs**: Cleaned, anatomically compliant `HandKeypointSet`.
* **Internal Logic**: Sequentially executes tendon coupling, dynamic fist clamping, hinge angle limiting, segment length clamping, and lateral finger lane checks.

### `apply_dip_pip_tendon_coupling(finger: &mut FingerChain)`
* **Responsibility**: Enforces the physiological coupling between the distal (DIP) and proximal (PIP) knuckles.
* **Inputs**: Mutable reference to a finger joint chain.
* **Outputs**: Mutated chain with DIP angle set to $\theta_{\text{DIP}} = 0.67 \times \theta_{\text{PIP}}$.
* **Internal Logic**: Measures flexion angle at PIP joint; computes coupled DIP target; rotates distal phalanx around PIP hinge axis.

### `apply_dynamic_fist_abduction_clamp(fingers: &mut [FingerChain; 4])`
* **Responsibility**: Suppresses lateral finger splay (abduction) as fingers curl into a closed fist.
* **Inputs**: Mutable slice of index, middle, ring, and pinky finger chains.
* **Outputs**: Mutated finger chains with lateral angles clamped according to knuckle flexion.
* **Internal Logic**: Evaluates average MCP flexion $\bar{\theta}_{\text{MCP}}$; scales maximum allowable lateral spread by $(1.0 - \bar{\theta}_{\text{MCP}} / 90^\circ)$; clamps finger positions within lateral boundary envelopes.

### `clamp_planar_hinge(parent: Vec3, middle: Vec3, child: Vec3, limits: (f32, f32)) -> Vec3`
* **Responsibility**: Constrains a 1-DOF joint (elbow, knee, PIP, DIP) to pure planar flexion, preventing lateral bend and backward hyperextension.
* **Inputs**: Parent joint position, middle hinge position, proposed child position, (min_flexion, max_flexion) angles.
* **Outputs**: Corrected child joint position.
* **Internal Logic**: Projects child vector onto the primary hinge plane; clamps angle between $(\text{middle} - \text{parent})$ and $(\text{child} - \text{middle})$ to $[0^\circ, \theta_{\max}]$; re-scales to invariant bone length.

### `BodyIKSolver::enforce_segment_lengths(&mut self, body: &mut BodyKeypointSet)`
* **Responsibility**: Prevents bones (femur, tibia, humerus, radius) from shrinking or stretching during perspective changes.
* **Inputs**: Mutable reference to full-body keypoints.
* **Outputs**: Mutated body keypoints conforming to learned invariant segment lengths.
* **Internal Logic**: Maintains EMA of confident bone lengths over time; re-projects child joints along unit direction vectors from parent joints matching invariant lengths.

---

## 4. Temporal Smoothing & Motion Filtering (`src/solver/`)

### `LandmarkFilter::smooth_points(&mut self, current: &[Point3D], dt: f32) -> Vec<Point3D>`
* **Responsibility**: Eliminates high-frequency sensor noise and landmark jitter without introducing phase lag.
* **Inputs**: Slice of observed 3D points, delta time since last frame.
* **Outputs**: Smoothed vector of 3D points.
* **Internal Logic**: Implements an adaptive OneEuro filter with dynamic cutoff frequency based on landmark velocity: $\beta \cdot |\dot{x}| + f_{\text{min}}$.

### `QuaternionFilter::slerp_rotations(&mut self, prev: Quat, current: Quat, alpha: f32) -> Quat`
* **Responsibility**: Interpolates joint orientations smoothly on the unit quaternion hypersphere $S^3$.
* **Inputs**: Previous joint quaternion, current raw joint quaternion, smoothing factor $\alpha$.
* **Outputs**: Spherical linearly interpolated quaternion $\mathbf{q}_{\text{smooth}}$.
* **Internal Logic**: Computes dot product $\mathbf{q}_0 \cdot \mathbf{q}_1$; negates target if negative to choose shortest geodesic path; calculates $\sin(\Omega)$ interpolation weights.

### `FootLockStabilizer::update(&mut self, ankles: (Vec3, Vec3), toes: (Vec3, Vec3), dt: f32) -> (Vec3, Vec3, Vec3, Vec3)`
* **Responsibility**: Detects ground contact phases and locks foot positions to prevent foot sliding / skating artifacts.
* **Inputs**: Left/right ankle and toe coordinates, frame delta time.
* **Outputs**: Stabilized left/right ankle and toe positions.
* **Internal Logic**: Evaluates downward velocity $v_z$ and vertical altitude $z$; if $|v| < v_{\text{threshold}}$ and $z < z_{\text{contact}}$, engages foot plant lock, holding horizontal $(x, y)$ coordinates fixed until upward acceleration triggers release.

---

## 5. Multi-Person Tracking & Association (`src/tracking/`)

### `MultiPersonTracker::update(&mut self, detections: &[BodyDetection], frame: &RgbImage) -> Vec<PersonTrack>`
* **Responsibility**: Associates per-frame multi-person detections into persistent, temporally coherent identity tracks.
* **Inputs**: Detections across all visible individuals, source video frame.
* **Outputs**: Active tracked people with historical trajectories.
* **Internal Logic**: Computes bipartite matching cost matrix using ByteTrack IoU overlap, constant-velocity motion extrapolation, and torso HSV color histogram similarity.

### `TorsoColorProfiler::compute_signature(frame: &RgbImage, torso_box: BoundingBox) -> ColorHistogram`
* **Responsibility**: Extracts an invariant clothing color signature from a person's chest and torso region.
* **Inputs**: Frame image, torso bounding box.
* **Outputs**: Normalized 16-bin HSV color histogram.
* **Internal Logic**: Crops torso region; converts to HSV color space; computes histogram over non-edge hue/saturation bins; smooths signature temporally.

### `resolve_cross_person_hand_ownership(tracks: &mut [PersonTrack])`
* **Responsibility**: Disambiguates hand ownership when actors stand close or cross hands in multi-person scenes.
* **Inputs**: Mutable slice of active person tracks.
* **Outputs**: Resolved hand assignments preventing "stolen" hand detections.
* **Internal Logic**: Measures Euclidean distance from hand wrist to candidate actor wrists; applies anatomical bounding-box penalty; substitutes synthetic fallback hands if another actor is the closer owner.

---

## 6. Multi-Camera Calibration & 3D Triangulation (`src/fusion/`)

### `CameraCalibrator::calibrate_charuco(frames_by_cam: &[Vec<RgbImage>]) -> Result<CameraIntrinsicsExtrinsics>`
* **Responsibility**: Computes intrinsic calibration matrices and extrinsic relative poses using synchronized ChArUco board footage.
* **Inputs**: Slices of synchronized frames per camera view.
* **Outputs**: Calibrated projection matrices $\mathbf{P}_k = \mathbf{K}_k [\mathbf{R}_k \mid \mathbf{t}_k]$ and lens distortion coefficients.
* **Internal Logic**: Detects ChArUco corners with sub-pixel refinement; solves Levenberg-Marquardt bundle adjustment; validates mean reprojection error $< 0.8\text{ px}$.

### `triangulate_3d_joint(projections: &[(Vec2, f32, &Mat3x4)]) -> Option<Vec3>`
* **Responsibility**: Triangulates true metric 3D joint coordinates from synchronized 2D observations across multiple calibrated cameras.
* **Inputs**: List of 2D screen points, confidence scores, and camera projection matrices $\mathbf{P}$.
* **Outputs**: Triangulated 3D Cartesian position $(X, Y, Z)$.
* **Internal Logic**: Formulates Direct Linear Transformation (DLT) linear system $\mathbf{A} \mathbf{X} = 0$; performs Singular Value Decomposition (SVD); extracts eigenvector corresponding to smallest singular value; checks reprojection residuals.

---

## 7. Coordinate Transformation & Scale Normalization (`src/export/`)

### `transform_opencv_to_blender_space(point: Vec3) -> Vec3`
* **Responsibility**: Converts coordinates from OpenCV camera conventions to Blender world space conventions.
* **Inputs**: 3D point in OpenCV camera space $(+X \text{ right}, +Y \text{ down}, +Z \text{ forward})$.
* **Outputs**: 3D point in Blender world space $(+X \text{ right}, +Y \text{ forward}, +Z \text{ up})$.
* **Internal Logic**: Evaluates $(x_{\text{blender}}, y_{\text{blender}}, z_{\text{blender}}) = (x_{\text{cv}}, z_{\text{cv}}, -y_{\text{cv}})$.

### `TorsoScaleNormalizer::normalize_sequence(&mut self, frames: &mut [MotionFrame], target_torso_length: f32)`
* **Responsibility**: Standardizes character skeletal scale across an entire motion sequence regardless of camera distance or lens zoom.
* **Inputs**: Mutable slice of timeline motion frames, target reference torso length (e.g., 0.50m).
* **Outputs**: Scaled motion frames with distance-invariant bone proportions.
* **Internal Logic**: Evaluates per-frame distance $L(t) = \|\mathbf{p}_{\text{Neck}} - \mathbf{p}_{\text{Hips}}\|$; runs bidirectional zero-lag EMA smoothing over $L(t)$; scales all joint offsets from `HipsRoot` by $S(t) = L_{\text{target}} / L_{\text{smoothed}}(t)$.

### `align_ground_plane(frames: &mut [MotionFrame])`
* **Responsibility**: Offsets the animation sequence vertically so the lowest foot contact aligns precisely with $Z = 0.0$.
* **Inputs**: Mutable slice of normalized motion frames.
* **Outputs**: Grounded motion frames.
* **Internal Logic**: Finds global minimum $Z_{\text{min}} = \min_{t, j \in \text{Feet}} Z_j(t)$; subtracts $Z_{\text{min}}$ from all $Z$ coordinates across all frames.

---

## 8. Export Pipeline (`src/export/`)

### `export_motion_json(path: &Path, frames: &[MotionFrame], skeleton: &SkeletonHierarchy, fps: f64) -> Result<()>`
* **Responsibility**: Serializes the normalized animation into standard `kinara-motion-json-v1` format.
* **Inputs**: Output JSON file path, sequence of motion frames, skeleton hierarchy specification, clip FPS.
* **Outputs**: Atomic unit `Result<()>`.
* **Internal Logic**: Writes JSON envelope with schema version, coordinate metadata, rest-pose offsets, and per-joint Cartesian trajectory dictionaries.

### `export_motion_fbx(path: &Path, frames: &[MotionFrame], skeleton: &SkeletonHierarchy, fps: f64) -> Result<()>`
* **Responsibility**: Generates a self-contained ASCII FBX 7.4 file containing an armature hierarchy and baked translation/rotation curves.
* **Inputs**: Destination FBX path, motion frames, skeleton spec, framerate.
* **Outputs**: Atomic unit `Result<()>`.
* **Internal Logic**: Emits FBX 7400 header; creates `Model::LimbNode` objects for all 78 joints; binds parent-child relations; writes time-sampled keyframe curves using FBX ticks ($46,186,158,000 \text{ ticks/sec}$).

---

## 9. Application Lifecycle & UI (`src/`)

### `run_pipeline(config: PipelineConfig) -> Result<RunSummary>`
* **Responsibility**: Orchestrates the top-level execution loop across frame capture, inference, IK solving, smoothing, and export.
* **Inputs**: Resolved runtime configuration struct.
* **Outputs**: Execution metrics summary (processed frames, average FPS, dropped frames, output paths).

### `KinaraApp::update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame)`
* **Responsibility**: Renders the immediate-mode native desktop studio GUI using `eframe` / `egui`.
* **Inputs**: egui render context, desktop window frame.
* **Outputs**: Interactive user interface updating model selection, source file pickers, real-time pose preview viewports, and processing progress bars.
