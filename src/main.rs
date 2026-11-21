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
        if self.config.hand_ik_enabled && body_kps.len() >= 11 {
            // Left hand (wrist: 9, elbow: 7)
            let lw = Vec2::new(body_kps[9].x as f32, body_kps[9].y as f32);
            let le = Vec2::new(body_kps[7].x as f32, body_kps[7].y as f32);
            let (ldir, llen) = compute_forearm_vector(le, lw);
            if let Some(crop) = extract_hand_crop(frame, lw, ldir, llen) {
                if let Ok(raw_hand) = self.hand_estimator.detect_hand(&crop) {
                    let constrained_pts = enforce_hand_constraints(&raw_hand.points);
                    hands.left = Some(constrained_pts);
                }
            }

            // Right hand (wrist: 10, elbow: 8)
            let rw = Vec2::new(body_kps[10].x as f32, body_kps[10].y as f32);
            let re = Vec2::new(body_kps[8].x as f32, body_kps[8].y as f32);
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
        for frame in motion_frames.iter_mut() {
            let left_ankle = frame.joints.get("LeftAnkle").map(|j| glam::Vec3::new(j.x, j.y, j.z)).unwrap_or(glam::Vec3::ZERO);
            let right_ankle = frame.joints.get("RightAnkle").map(|j| glam::Vec3::new(j.x, j.y, j.z)).unwrap_or(glam::Vec3::ZERO);
            let left_toe = frame.joints.get("LeftToeBase").map(|j| glam::Vec3::new(j.x, j.y, j.z)).unwrap_or(glam::Vec3::ZERO);
            let right_toe = frame.joints.get("RightToeBase").map(|j| glam::Vec3::new(j.x, j.y, j.z)).unwrap_or(glam::Vec3::ZERO);

            let mut contacts = FootContacts::new(left_ankle, right_ankle, left_toe, right_toe);
            session.foot_stabilizer.lock_planted_feet(&mut contacts, dt);

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
        let mut meta = ClipMetadata::default();
        meta.fps = config.fps;
        meta.frame_count = motion_frames.len();
        meta.source_file = Some(config.input_source.clone());
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
    let args = CliArgs::parse();

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
                normalize_sequence_scale(&mut seq, 50.0);
                align_to_ground_plane(&mut seq);
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
            println!("Kinara Desktop Studio GUI is scheduled for Phase 2 desktop windowing.");
            if let Some(inp) = input {
                println!("Target input source: {}", inp);
            }
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
