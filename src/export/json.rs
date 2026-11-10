use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use crate::types::FramePose;

#[derive(Debug, Serialize, Deserialize)]
pub struct MotionExportData {
    pub engine: String,
    pub version: String,
    pub total_frames: usize,
    pub fps: f32,
    pub frames: Vec<FramePose>,
}

pub struct MotionExporter {
    frames: Vec<FramePose>,
    fps: f32,
}

impl MotionExporter {
    pub fn new(fps: f32) -> Self {
        Self {
            frames: Vec::new(),
            fps,
        }
    }

    pub fn add_frame(&mut self, pose: FramePose) {
        self.frames.push(pose);
    }

    pub fn save_to_json(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create export output directory")?;
        }

        let file = File::create(path).context("Failed to create motion export JSON file")?;
        let writer = BufWriter::new(file);

        let export_data = MotionExportData {
            engine: "Kinara-RS".to_string(),
            version: "0.2.0".to_string(),
            total_frames: self.frames.len(),
            fps: self.fps,
            frames: self.frames.clone(),
        };

        serde_json::to_writer_pretty(writer, &export_data)
            .context("Failed to serialize motion JSON")?;

        Ok(())
    }
}
