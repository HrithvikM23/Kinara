use std::path::PathBuf;
use crate::types::BoundingBox;

/// Input video source.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum VideoSource {
    Camera(u32),
    File(PathBuf),
}

impl VideoSource {
    pub fn parse(input: &str) -> Self {
        if let Ok(index) = input.parse::<u32>() {
            VideoSource::Camera(index)
        } else {
            VideoSource::File(PathBuf::from(input))
        }
    }
}

/// Uncompressed native-resolution frame buffer.
/// Preserves full 1080p / 4K sensor data without destructive downsampling.
#[derive(Debug, Clone)]
pub struct FrameBuffer {
    pub width: u32,
    pub height: u32,
    /// Packed RGB8 pixel bytes (len = width * height * 3).
    pub data: Vec<u8>,
    pub frame_index: usize,
    pub timestamp_sec: f64,
}

impl FrameBuffer {
    pub fn new(width: u32, height: u32, data: Vec<u8>, frame_index: usize, timestamp_sec: f64) -> Self {
        Self {
            width,
            height,
            data,
            frame_index,
            timestamp_sec,
        }
    }

    /// Creates an empty frame buffer (e.g. for testing / placeholders).
    pub fn empty(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0u8; (width * height * 3) as usize],
            frame_index: 0,
            timestamp_sec: 0.0,
        }
    }

    /// Crops a sub-region from the native frame at full 1:1 pixel resolution.
    /// This ensures distant hands retain sub-pixel finger detail from 1080p/4K footage.
    pub fn crop(&self, bbox: &BoundingBox) -> Option<FrameBuffer> {
        let x1 = bbox.x.max(0.0).min(self.width as f32) as u32;
        let y1 = bbox.y.max(0.0).min(self.height as f32) as u32;
        let x2 = (bbox.x + bbox.width).max(0.0).min(self.width as f32) as u32;
        let y2 = (bbox.y + bbox.height).max(0.0).min(self.height as f32) as u32;

        if x2 <= x1 || y2 <= y1 {
            return None;
        }

        let crop_w = x2 - x1;
        let crop_h = y2 - y1;
        let mut crop_data = Vec::with_capacity((crop_w * crop_h * 3) as usize);

        for row in y1..y2 {
            let row_start = ((row * self.width + x1) * 3) as usize;
            let row_end = row_start + (crop_w * 3) as usize;
            crop_data.extend_from_slice(&self.data[row_start..row_end]);
        }

        Some(FrameBuffer::new(
            crop_w,
            crop_h,
            crop_data,
            self.frame_index,
            self.timestamp_sec,
        ))
    }

    /// Generates a fast downscaled copy for the full-body detector (e.g. 640x360),
    /// leaving the native unscaled buffer untouched for subsequent hand crops.
    pub fn downscale_for_detector(&self, target_w: u32, target_h: u32) -> Vec<u8> {
        let mut resized = Vec::with_capacity((target_w * target_h * 3) as usize);
        let x_ratio = self.width as f32 / target_w as f32;
        let y_ratio = self.height as f32 / target_h as f32;

        for ty in 0..target_h {
            let sy = (ty as f32 * y_ratio) as u32;
            let src_row = (sy.min(self.height - 1) * self.width) as usize;
            for tx in 0..target_w {
                let sx = (tx as f32 * x_ratio) as u32;
                let src_idx = (src_row + sx.min(self.width - 1) as usize) * 3;
                resized.push(self.data[src_idx]);
                resized.push(self.data[src_idx + 1]);
                resized.push(self.data[src_idx + 2]);
            }
        }
        resized
    }
}
