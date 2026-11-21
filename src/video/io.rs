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
                let _ = child.kill();
                let _ = child.wait();
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
