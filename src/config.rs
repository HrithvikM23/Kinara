use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tracing::info;

pub const DEFAULT_CONFIG_PATH: &str = "kinara.config.json";

/// Engine runtime configuration and CLI arguments.
#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
#[command(
    name = "kinara",
    author = "HrithvikM",
    version = "0.2.0",
    about = "High-performance GPU-accelerated motion capture engine with finger-accurate 3D hand tracking."
)]
pub struct KinaraConfig {
    /// Launch the native Rust desktop GUI interface.
    #[arg(long, default_value_t = false)]
    #[serde(default)]
    pub gui: bool,

    /// Video input source: webcam index (e.g. "0") or video file path (e.g. "dance.mp4").
    #[arg(short, long, default_value = "0")]
    pub input: String,

    /// Optional synchronized face cam / headrig video file (e.g. "face_cam.mp4").
    #[arg(long)]
    #[serde(default)]
    pub face_input: Option<PathBuf>,

    /// Path to the body pose ONNX model.
    #[arg(long, default_value = "models/dwpose_l.onnx")]
    pub body_model: PathBuf,

    /// Path to the 3D hand mesh / pose ONNX model.
    #[arg(long, default_value = "models/wilor_hand.onnx")]
    pub hand_model: PathBuf,

    /// Output animation file path (.json, .fbx, or .bvh).
    #[arg(short, long, default_value = "outputs/motion.json")]
    pub output: PathBuf,

    /// Target execution device: "cuda", "directml", or "cpu".
    #[arg(long, default_value = "directml")]
    pub device: String,

    /// Enable strict biomechanical IK limits (DIP-PIP tendon coupling, dynamic fist clamping).
    #[arg(long, default_value_t = true)]
    pub biomechanical_ik: bool,

    /// Quaternion SLERP smoothing alpha (0.0 = completely frozen, 1.0 = raw unsmoothed).
    #[arg(long, default_value_t = 0.65)]
    pub smoothing_alpha: f32,

    /// Minimum confidence threshold for hand detection before falling back to neutral rest pose.
    #[arg(long, default_value_t = 0.40)]
    pub hand_min_confidence: f32,

    /// Minimum hand crop box size (pixels) before blending into natural rest pose.
    #[arg(long, default_value_t = 24.0)]
    pub hand_min_box_size: f32,

    /// Enable parallel multi-frame pipelining across worker threads.
    #[arg(long, default_value_t = true)]
    pub pipelined: bool,
}

impl Default for KinaraConfig {
    fn default() -> Self {
        Self {
            gui: false,
            input: "0".to_string(),
            face_input: None,
            body_model: PathBuf::from("models/dwpose_l.onnx"),
            hand_model: PathBuf::from("models/wilor_hand.onnx"),
            output: PathBuf::from("outputs/motion.json"),
            device: "directml".to_string(),
            biomechanical_ik: true,
            smoothing_alpha: 0.65,
            hand_min_confidence: 0.40,
            hand_min_box_size: 24.0,
            pipelined: true,
        }
    }
}

impl KinaraConfig {
    /// Loads saved persistent configuration from disk if available, otherwise returns default.
    pub fn load_or_default(path: &Path) -> Self {
        if path.exists() {
            if let Ok(file) = File::open(path) {
                let reader = BufReader::new(file);
                if let Ok(config) = serde_json::from_reader(reader) {
                    info!("Loaded persistent configuration from {:?}", path);
                    return config;
                }
            }
        }
        Self::default()
    }

    /// Saves the current configuration to disk to ensure user choices are permanently preserved.
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).context("Failed to create config directory")?;
            }
        }
        let file = File::create(path).context("Failed to create config file")?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self).context("Failed to serialize config JSON")?;
        info!("Saved persistent configuration to {:?}", path);
        Ok(())
    }

    /// Loads saved configuration and merges with any explicit CLI overrides.
    pub fn load_with_cli() -> Self {
        let default_config_path = PathBuf::from(DEFAULT_CONFIG_PATH);
        let cli_args = std::env::args().collect::<Vec<_>>();

        // If no arguments passed at all, or only program name passed, launch GUI with saved config
        if cli_args.len() <= 1 {
            let mut config = Self::load_or_default(&default_config_path);
            config.gui = true;
            return config;
        }

        let parsed = Self::parse();

        // Load base config from disk if it exists
        let mut base = Self::load_or_default(&default_config_path);

        // Merge CLI values over saved base
        base.gui = parsed.gui;
        if parsed.input != "0" || base.input.is_empty() {
            base.input = parsed.input;
        }
        if parsed.face_input.is_some() {
            base.face_input = parsed.face_input;
        }
        base.device = parsed.device;
        base.body_model = parsed.body_model;
        base.hand_model = parsed.hand_model;
        base.output = parsed.output;
        base.biomechanical_ik = parsed.biomechanical_ik;
        base.smoothing_alpha = parsed.smoothing_alpha;
        base.hand_min_confidence = parsed.hand_min_confidence;
        base.hand_min_box_size = parsed.hand_min_box_size;
        base.pipelined = parsed.pipelined;

        // Persist the merged configuration
        let _ = base.save_to_file(&default_config_path);

        base
    }
}
