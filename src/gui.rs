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
