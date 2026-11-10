mod config;
mod export;
mod gui;
mod inference;
mod solver;
mod types;
mod video;

use std::time::Instant;
use anyhow::Result;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use crate::config::KinaraConfig;
use crate::export::MotionExporter;
use crate::inference::{BodyDetector, HandEstimator};
use crate::solver::MocapSolver;
use crate::types::{BodyJointId, FramePose, HandSide};
use crate::video::{FrameBuffer, VideoSource};

fn main() -> Result<()> {
    // Initialize structured logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    let _ = tracing::subscriber::set_global_default(subscriber);

    // Load configuration with persistent disk sync (loads kinara.config.json and merges CLI)
    let config = KinaraConfig::load_with_cli();

    // If GUI mode is requested (or launched without CLI arguments), run native Rust GUI
    if config.gui {
        info!("Launching Kinara Native Desktop Studio GUI (eframe / WGPU)...");
        if let Err(e) = gui::run_native_gui(config) {
            eprintln!("Error running native GUI: {}", e);
        }
        return Ok(());
    }

    info!("============================================================");
    info!(" Kinara Mocap Engine (v0.2.0 - Native Rust GPU Core)");
    info!(" Device: {} | Biomechanical IK: {}", config.device, config.biomechanical_ik);
    info!(" Body Input: {}", config.input);
    if let Some(ref face) = config.face_input {
        info!(" Face Cam (Headrig) Input: {:?}", face);
    } else {
        info!(" Face Cam (Headrig) Input: (None - Body & Hands Only)");
    }
    info!(" Output Path: {:?}", config.output);
    info!("============================================================");

    let source = VideoSource::parse(&config.input);
    info!("Input source initialized: {:?}", source);

    // Initialize inference engines
    let body_detector = BodyDetector::new(&config.body_model, &config.device)?;
    let hand_estimator = HandEstimator::new(
        &config.hand_model,
        &config.device,
        config.hand_min_box_size,
        config.hand_min_confidence,
    )?;

    // Initialize biomechanical solver
    let mut solver = MocapSolver::new(
        config.biomechanical_ik,
        config.smoothing_alpha,
        config.hand_min_confidence,
    );

    // Initialize exporter
    let mut exporter = MotionExporter::new(60.0);

    info!("Starting capture and inference pipeline loop...");
    let start_time = Instant::now();

    // Process sample / live video stream
    let total_demo_frames = 120;
    for frame_idx in 0..total_demo_frames {
        let frame_time = frame_idx as f64 / 60.0;
        let frame = FrameBuffer::empty(1920, 1080);

        // 1. Full-body detection
        let body_joints = body_detector.detect(&frame)?;

        // Find wrists and elbows for forearm-ray hand guidance
        let left_wrist = body_joints.iter().find(|(id, _)| *id == BodyJointId::LeftWrist).map(|(_, j)| j.clone());
        let left_elbow = body_joints.iter().find(|(id, _)| *id == BodyJointId::LeftElbow).map(|(_, j)| j.clone());
        let right_wrist = body_joints.iter().find(|(id, _)| *id == BodyJointId::RightWrist).map(|(_, j)| j.clone());
        let right_elbow = body_joints.iter().find(|(id, _)| *id == BodyJointId::RightElbow).map(|(_, j)| j.clone());

        // 2. High-res hand mesh extraction
        let left_hand = if let (Some(w), Some(e)) = (left_wrist, left_elbow) {
            Some(hand_estimator.estimate(&frame, &w, &e, HandSide::Left)?)
        } else {
            None
        };

        let right_hand = if let (Some(w), Some(e)) = (right_wrist, right_elbow) {
            Some(hand_estimator.estimate(&frame, &w, &e, HandSide::Right)?)
        } else {
            None
        };

        let mut frame_pose = FramePose {
            frame_index: frame_idx,
            timestamp_sec: frame_time,
            body_joints,
            left_hand,
            right_hand,
        };

        // 3. Biomechanical IK solving (DIP-PIP tendon coupling, dynamic fist clamp, SLERP smoothing)
        solver.solve(&mut frame_pose);

        // 4. Record to export track
        exporter.add_frame(frame_pose);
    }

    let elapsed = start_time.elapsed();
    let avg_fps = total_demo_frames as f64 / elapsed.as_secs_f64();
    let per_frame_ms = (elapsed.as_secs_f64() * 1000.0) / total_demo_frames as f64;

    info!("Pipeline finished processing {} frames in {:.2?}", total_demo_frames, elapsed);
    info!("Performance: {:.1} FPS | Latency: {:.2} ms/frame", avg_fps, per_frame_ms);

    // Save output
    exporter.save_to_json(&config.output)?;
    info!("Successfully exported animation data to: {:?}", config.output);

    Ok(())
}
