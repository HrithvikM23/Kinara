//! Motion JSON export in standard `kinara-motion-json-v1` format.

use crate::export::canonical::ClipMetadata;
use crate::export::coords::MotionFrame;
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Atomically exports motion frames and metadata to standard `kinara-motion-json-v1` format.
pub fn export_motion_json(
    path: impl AsRef<Path>,
    frames: &[MotionFrame],
    meta: &ClipMetadata,
) -> anyhow::Result<()> {
    let path = path.as_ref();
    let tmp_path = std::path::PathBuf::from(format!("{}.tmp", path.display()));

    #[derive(Serialize)]
    struct JsonPayload<'a> {
        format: &'a str,
        fps: f64,
        frame_count: usize,
        metadata: &'a ClipMetadata,
        frames: &'a [MotionFrame],
    }

    let payload = JsonPayload {
        format: &meta.format,
        fps: meta.fps,
        frame_count: frames.len(),
        metadata: meta,
        frames,
    };

    let serialized = serde_json::to_string_pretty(&payload)?;
    {
        let mut file = File::create(&tmp_path)?;
        file.write_all(serialized.as_bytes())?;
        file.flush()?;
    }

    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::canonical::{build_canonical_joint_map, HandsPair};
    use crate::skeleton::Point;
    use crate::solver::body_constraints::BodyKeypoints;

    fn dummy_body_keypoints() -> BodyKeypoints {
        let mut pts = vec![Point::new(0, 0, 0.9); 17];
        pts[5] = Point::new(100, 100, 0.9);
        pts[6] = Point::new(200, 100, 0.9);
        pts[7] = Point::new(90, 150, 0.9);
        pts[8] = Point::new(210, 150, 0.9);
        pts[9] = Point::new(80, 200, 0.9);
        pts[10] = Point::new(220, 200, 0.9);
        pts[11] = Point::new(120, 250, 0.9);
        pts[12] = Point::new(180, 250, 0.9);
        pts[13] = Point::new(120, 320, 0.9);
        pts[14] = Point::new(180, 320, 0.9);
        pts[15] = Point::new(120, 380, 0.9);
        pts[16] = Point::new(180, 380, 0.9);
        BodyKeypoints::new(pts)
    }

    #[test]
    fn test_export_motion_json_roundtrip() {
        let body = dummy_body_keypoints();
        let map = build_canonical_joint_map(&body, &HandsPair::default(), None);
        let frames = vec![MotionFrame::new(0, map)];
        let meta = ClipMetadata::default();

        let temp_dir = std::env::temp_dir();
        let out_path = temp_dir.join(format!("test_kinara_motion_{}.json", std::process::id()));

        export_motion_json(&out_path, &frames, &meta).expect("Failed to export motion JSON");
        assert!(out_path.exists());

        let contents = std::fs::read_to_string(&out_path).expect("Failed to read back");
        assert!(contents.contains("kinara-motion-json-v1"));
        assert!(contents.contains("HipsRoot"));

        let _ = std::fs::remove_file(out_path);
    }
}
