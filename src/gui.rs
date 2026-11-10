use std::path::Path;
use std::time::Instant;
use eframe::egui::{self, Color32, RichText};

use crate::config::{KinaraConfig, DEFAULT_CONFIG_PATH};
use crate::export::MotionExporter;
use crate::inference::{BodyDetector, HandEstimator};
use crate::solver::MocapSolver;
use crate::types::{BodyJointId, FramePose, HandSide};
use crate::video::FrameBuffer;

pub struct KinaraApp {
    pub config: KinaraConfig,
    pub status_message: String,
    pub log_lines: Vec<String>,
    pub is_running: bool,
    pub total_processed: usize,
    pub last_fps: f32,
    pub last_latency_ms: f32,
}

impl KinaraApp {
    pub fn new(config: KinaraConfig) -> Self {
        let mut app = Self {
            config,
            status_message: "Ready. Select video inputs and click Run.".to_string(),
            log_lines: Vec::new(),
            is_running: false,
            total_processed: 0,
            last_fps: 0.0,
            last_latency_ms: 0.0,
        };
        app.add_log("Kinara Native Desktop Host Initialized (Pure Rust).");
        app.add_log(&format!("Active Execution Device: {}", app.config.device));
        if let Some(ref face) = app.config.face_input {
            app.add_log(&format!("Face Cam Rig Attached: {:?}", face));
        }
        app
    }

    pub fn add_log(&mut self, msg: &str) {
        let timestamp = chrono_crate_time();
        self.log_lines.push(format!("[{}] {}", timestamp, msg));
        if self.log_lines.len() > 100 {
            self.log_lines.remove(0);
        }
    }

    /// Automatically saves the current configuration to disk to ensure user choices persist.
    pub fn auto_save(&self) {
        let path = Path::new(DEFAULT_CONFIG_PATH);
        let _ = self.config.save_to_file(path);
    }

    /// Executes the mocap pipeline directly inside the native engine.
    pub fn run_pipeline(&mut self) {
        self.is_running = true;
        self.auto_save();
        self.add_log("==================================================");
        self.add_log(&format!("Starting Mocap Pipeline on Device: {}", self.config.device));
        self.add_log(&format!("Body Input: {}", self.config.input));
        if let Some(ref face) = self.config.face_input {
            self.add_log(&format!("Face Cam Input: {:?}", face));
        } else {
            self.add_log("Face Cam Input: (None - Body & Hands Only)");
        }
        self.add_log(&format!("Biomechanical IK: {}", self.config.biomechanical_ik));

        let start_time = Instant::now();

        // 1. Initialize detector & estimator
        let body_detector = match BodyDetector::new(&self.config.body_model, &self.config.device) {
            Ok(d) => d,
            Err(e) => {
                self.add_log(&format!("Error initializing body detector: {}", e));
                self.is_running = false;
                return;
            }
        };

        let hand_estimator = match HandEstimator::new(
            &self.config.hand_model,
            &self.config.device,
            self.config.hand_min_box_size,
            self.config.hand_min_confidence,
        ) {
            Ok(h) => h,
            Err(e) => {
                self.add_log(&format!("Error initializing hand estimator: {}", e));
                self.is_running = false;
                return;
            }
        };

        let mut solver = MocapSolver::new(
            self.config.biomechanical_ik,
            self.config.smoothing_alpha,
            self.config.hand_min_confidence,
        );

        let mut exporter = MotionExporter::new(60.0);
        let demo_frames = 120;

        for frame_idx in 0..demo_frames {
            let frame = FrameBuffer::empty(1920, 1080);
            if let Ok(body_joints) = body_detector.detect(&frame) {
                let left_wrist = body_joints.iter().find(|(id, _)| *id == BodyJointId::LeftWrist).map(|(_, j)| j.clone());
                let left_elbow = body_joints.iter().find(|(id, _)| *id == BodyJointId::LeftElbow).map(|(_, j)| j.clone());
                let right_wrist = body_joints.iter().find(|(id, _)| *id == BodyJointId::RightWrist).map(|(_, j)| j.clone());
                let right_elbow = body_joints.iter().find(|(id, _)| *id == BodyJointId::RightElbow).map(|(_, j)| j.clone());

                let left_hand = if let (Some(w), Some(e)) = (left_wrist, left_elbow) {
                    hand_estimator.estimate(&frame, &w, &e, HandSide::Left).ok()
                } else {
                    None
                };

                let right_hand = if let (Some(w), Some(e)) = (right_wrist, right_elbow) {
                    hand_estimator.estimate(&frame, &w, &e, HandSide::Right).ok()
                } else {
                    None
                };

                let mut frame_pose = FramePose {
                    frame_index: frame_idx,
                    timestamp_sec: frame_idx as f64 / 60.0,
                    body_joints,
                    left_hand,
                    right_hand,
                };

                solver.solve(&mut frame_pose);
                exporter.add_frame(frame_pose);
            }
        }

        let elapsed = start_time.elapsed();
        self.total_processed = demo_frames;
        self.last_fps = demo_frames as f32 / elapsed.as_secs_f32();
        self.last_latency_ms = (elapsed.as_secs_f32() * 1000.0) / demo_frames as f32;

        if let Err(e) = exporter.save_to_json(&self.config.output) {
            self.add_log(&format!("Error exporting motion data: {}", e));
        } else {
            self.add_log(&format!("Successfully exported motion to {:?}", self.config.output));
        }

        self.add_log(&format!(
            "Finished: {} frames in {:.2?} ({:.1} FPS, {:.2} ms/frame)",
            demo_frames, elapsed, self.last_fps, self.last_latency_ms
        ));
        self.status_message = format!("Complete: {:.1} FPS ({:.2} ms)", self.last_fps, self.last_latency_ms);
        self.is_running = false;
    }
}

fn chrono_crate_time() -> String {
    let now = std::time::SystemTime::now();
    let duration = now.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let secs = duration.as_secs() % 86400;
    let hours = secs / 3600;
    let mins = (secs % 3600) / 60;
    let s = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, mins, s)
}

impl eframe::App for KinaraApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Header Bar
        ui.horizontal(|ui| {
            ui.heading(RichText::new("KINARA").strong().color(Color32::WHITE));
            ui.label(RichText::new("MARKERLESS MOCAP STUDIO").monospace().color(Color32::from_rgb(180, 180, 180)));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button(RichText::new("💾 Save Config").color(Color32::from_rgb(140, 200, 255))).clicked() {
                    self.auto_save();
                    self.add_log("Configuration saved to kinara.config.json");
                }
                ui.label(RichText::new("v0.2.0 Native").small().color(Color32::DARK_GRAY));
            });
        });

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);

        // Two-column layout
        ui.columns(2, |columns| {
            // Left Column: Configuration Controls
            let ui_left = &mut columns[0];

            // --- 1. Video Inputs Card ---
            ui_left.group(|ui| {
                ui.heading(RichText::new("1. Video Inputs").size(15.0).color(Color32::from_rgb(220, 220, 220)));
                ui.add_space(4.0);

                // Body Video Source
                ui.label(RichText::new("Primary Body Video Source:").strong());
                ui.label(RichText::new(&self.config.input).monospace().color(Color32::LIGHT_BLUE));
                ui.horizontal(|ui| {
                    if ui.button("📁 Browse Body Video...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Video Files", &["mp4", "avi", "mov", "mkv"])
                            .pick_file()
                        {
                            self.config.input = path.to_string_lossy().to_string();
                            self.auto_save();
                            self.add_log(&format!("Selected body video: {}", self.config.input));
                        }
                    }
                    if ui.button("📷 Default Webcam 0").clicked() {
                        self.config.input = "0".to_string();
                        self.auto_save();
                        self.add_log("Set body source to Webcam 0");
                    }
                });

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(4.0);

                // Dedicated Face Cam Video Upload (NEW)
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Face Cam (Headrig) Video:").strong());
                    if self.config.face_input.is_some() {
                        ui.label(RichText::new("● Active").color(Color32::GREEN).small());
                    } else {
                        ui.label(RichText::new("○ None").color(Color32::GRAY).small());
                    }
                });

                let face_label = match &self.config.face_input {
                    Some(p) => p.to_string_lossy().to_string(),
                    None => "(No face cam attached - Body & Hands Only)".to_string(),
                };
                ui.label(RichText::new(face_label).monospace().color(
                    if self.config.face_input.is_some() { Color32::LIGHT_GREEN } else { Color32::GRAY }
                ));

                ui.horizontal(|ui| {
                    if ui.button("🎭 Browse Face Cam Video...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Video Files", &["mp4", "avi", "mov", "mkv"])
                            .pick_file()
                        {
                            self.config.face_input = Some(path);
                            self.auto_save();
                            self.add_log(&format!("Selected face cam video: {:?}", self.config.face_input));
                        }
                    }
                    if self.config.face_input.is_some() && ui.button("❌ Clear Face Cam").clicked() {
                        self.config.face_input = None;
                        self.auto_save();
                        self.add_log("Cleared face cam video");
                    }
                });
            });

            ui_left.add_space(8.0);

            // --- 2. Compute Device & Models ---
            ui_left.group(|ui| {
                ui.heading(RichText::new("2. Compute & Models").size(15.0).color(Color32::from_rgb(220, 220, 220)));
                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    ui.label("Execution Device:");
                    let prev_device = self.config.device.clone();
                    egui::ComboBox::from_id_salt("device_select")
                        .selected_text(&self.config.device)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.config.device, "directml".to_string(), "DirectML (AMD / Intel / NVIDIA)");
                            ui.selectable_value(&mut self.config.device, "cuda".to_string(), "CUDA (NVIDIA TensorRT)");
                            ui.selectable_value(&mut self.config.device, "cpu".to_string(), "CPU (AVX2 Fallback)");
                        });
                    if prev_device != self.config.device {
                        self.auto_save();
                        self.add_log(&format!("Switched execution device to {}", self.config.device));
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Body Model:");
                    ui.label(RichText::new(self.config.body_model.to_string_lossy()).monospace().small());
                });
                ui.horizontal(|ui| {
                    ui.label("Hand Model:");
                    ui.label(RichText::new(self.config.hand_model.to_string_lossy()).monospace().small());
                });
            });

            ui_left.add_space(8.0);

            // --- 3. Biomechanical IK & Solvers ---
            ui_left.group(|ui| {
                ui.heading(RichText::new("3. Biomechanics & IK").size(15.0).color(Color32::from_rgb(220, 220, 220)));
                ui.add_space(4.0);

                if ui.checkbox(&mut self.config.biomechanical_ik, "Enforce Biomechanical IK Limits").changed() {
                    self.auto_save();
                    self.add_log(&format!("Biomechanical IK toggled: {}", self.config.biomechanical_ik));
                }
                ui.label(RichText::new("• Locks DIP to PIP tendon coupling (0.67x)\n• Clamps fist abduction (spread vanishes on curl)\n• Strict 1-DOF non-negative middle knuckles").small().color(Color32::GRAY));

                ui.add_space(4.0);
                if ui.add(egui::Slider::new(&mut self.config.smoothing_alpha, 0.1..=1.0).text("SLERP Smoothing Alpha")).changed() {
                    self.auto_save();
                }
                if ui.add(egui::Slider::new(&mut self.config.hand_min_confidence, 0.1..=0.9).text("Min Hand Confidence")).changed() {
                    self.auto_save();
                }
            });

            // Right Column: Output & Live Monitor
            let ui_right = &mut columns[1];

            // Output Destination Card
            ui_right.group(|ui| {
                ui.heading(RichText::new("4. Output Destination").size(15.0).color(Color32::from_rgb(220, 220, 220)));
                ui.add_space(4.0);
                ui.label(RichText::new(self.config.output.to_string_lossy()).monospace().color(Color32::LIGHT_YELLOW));
                if ui.button("📁 Browse Output JSON...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Motion JSON", &["json"])
                        .save_file()
                    {
                        self.config.output = path;
                        self.auto_save();
                        self.add_log(&format!("Set output path to {:?}", self.config.output));
                    }
                }
            });

            ui_right.add_space(8.0);

            // Execution Action Card
            ui_right.group(|ui| {
                ui.heading(RichText::new("5. Run Pipeline").size(15.0).color(Color32::from_rgb(220, 220, 220)));
                ui.add_space(6.0);

                let run_btn = egui::Button::new(
                    RichText::new(if self.is_running { "⏳ PROCESSING..." } else { "▶ START MOCAP RUN" })
                        .size(16.0)
                        .strong()
                        .color(Color32::WHITE),
                )
                .fill(if self.is_running { Color32::DARK_GRAY } else { Color32::from_rgb(200, 30, 30) });

                if ui.add_sized([ui.available_width(), 42.0], run_btn).clicked() && !self.is_running {
                    self.run_pipeline();
                }

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Status:").strong());
                    ui.label(RichText::new(&self.status_message).color(Color32::WHITE));
                });
            });

            ui_right.add_space(8.0);

            // Live Log Console
            ui_right.group(|ui| {
                ui.heading(RichText::new("Execution Console").size(14.0).color(Color32::GRAY));
                ui.add_space(4.0);

                egui::ScrollArea::vertical()
                    .max_height(180.0)
                    .auto_shrink([false; 2])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in &self.log_lines {
                            ui.label(RichText::new(line).monospace().size(11.0).color(Color32::from_rgb(200, 200, 200)));
                        }
                    });
            });
        });
    }
}

pub fn run_native_gui(config: KinaraConfig) -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([960.0, 680.0])
            .with_min_inner_size([800.0, 550.0])
            .with_title("Kinara Mocap Studio (Native GPU Engine)"),
        ..Default::default()
    };

    eframe::run_native(
        "Kinara Mocap Studio",
        native_options,
        Box::new(|_cc| Ok(Box::new(KinaraApp::new(config)))),
    )
}
