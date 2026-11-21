//! Non-linear video timeline data structures, clip cutting, trimming, and multi-clip merging.
//! Foundation for the native desktop timeline editor.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Represents a single video clip placed on the timeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelineClip {
    pub id: u64,
    pub file_path: PathBuf,
    pub source_fps: f64,
    pub total_source_frames: usize,
    /// Trim in-point (inclusive start frame in source video)
    pub in_frame: usize,
    /// Trim out-point (exclusive end frame in source video)
    pub out_frame: usize,
    /// Track lane index (0 = primary video track, 1+ = auxiliary/camera views)
    pub track_index: usize,
}

impl TimelineClip {
    pub fn new(id: u64, file_path: PathBuf, source_fps: f64, total_frames: usize) -> Self {
        Self {
            id,
            file_path,
            source_fps,
            total_source_frames: total_frames,
            in_frame: 0,
            out_frame: total_frames,
            track_index: 0,
        }
    }

    /// Effective duration of the clip in frames after trimming.
    #[inline]
    pub fn duration_frames(&self) -> usize {
        self.out_frame.saturating_sub(self.in_frame)
    }

    /// Effective duration of the clip in seconds.
    #[inline]
    pub fn duration_seconds(&self) -> f64 {
        if self.source_fps <= 0.0 {
            0.0
        } else {
            self.duration_frames() as f64 / self.source_fps
        }
    }

    /// Sets non-destructive trim in/out bounds.
    pub fn set_trim(&mut self, in_frame: usize, out_frame: usize) -> Result<(), &'static str> {
        if in_frame >= out_frame {
            return Err("In-point must be strictly less than out-point");
        }
        if out_frame > self.total_source_frames {
            return Err("Out-point exceeds source duration");
        }
        self.in_frame = in_frame;
        self.out_frame = out_frame;
        Ok(())
    }

    /// Splits (cuts) this clip into two clips at the specified frame offset relative to in_frame.
    /// Returns the second clip starting at the cut point.
    pub fn split_at(&mut self, cut_offset: usize, new_clip_id: u64) -> Result<Self, &'static str> {
        let absolute_cut = self.in_frame + cut_offset;
        if absolute_cut <= self.in_frame || absolute_cut >= self.out_frame {
            return Err("Cut position must fall strictly inside the active trim range");
        }

        let mut second_clip = self.clone();
        second_clip.id = new_clip_id;
        second_clip.in_frame = absolute_cut;

        self.out_frame = absolute_cut;
        Ok(second_clip)
    }
}

/// Non-linear multi-track video timeline editor state.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TimelineEditor {
    pub clips: Vec<TimelineClip>,
    pub playhead_frame: usize,
    pub next_clip_id: u64,
}

impl TimelineEditor {
    pub fn new() -> Self {
        Self {
            clips: Vec::new(),
            playhead_frame: 0,
            next_clip_id: 1,
        }
    }

    /// Ingests a video clip dropped or loaded into the timeline.
    pub fn add_clip(&mut self, path: impl AsRef<Path>, fps: f64, total_frames: usize) -> u64 {
        let id = self.next_clip_id;
        self.next_clip_id += 1;
        self.clips.push(TimelineClip::new(id, path.as_ref().to_path_buf(), fps, total_frames));
        id
    }

    /// Removes a clip by its unique ID.
    pub fn remove_clip(&mut self, clip_id: u64) -> bool {
        if let Some(pos) = self.clips.iter().position(|c| c.id == clip_id) {
            self.clips.remove(pos);
            true
        } else {
            false
        }
    }

    /// Cuts (splits) a clip at a given frame offset, appending the newly created second half.
    pub fn cut_clip(&mut self, clip_id: u64, cut_offset: usize) -> Result<u64, &'static str> {
        let new_id = self.next_clip_id;
        let pos = self
            .clips
            .iter()
            .position(|c| c.id == clip_id)
            .ok_or("Clip not found")?;

        let second_half = self.clips[pos].split_at(cut_offset, new_id)?;
        self.next_clip_id += 1;
        self.clips.insert(pos + 1, second_half);
        Ok(new_id)
    }

    /// Merges two contiguous clips that share the same source file into one unified clip.
    pub fn merge_clips(&mut self, first_id: u64, second_id: u64) -> Result<(), &'static str> {
        let pos1 = self.clips.iter().position(|c| c.id == first_id).ok_or("First clip not found")?;
        let pos2 = self.clips.iter().position(|c| c.id == second_id).ok_or("Second clip not found")?;

        if self.clips[pos1].file_path != self.clips[pos2].file_path {
            return Err("Cannot merge clips with different source video files");
        }
        if self.clips[pos1].out_frame != self.clips[pos2].in_frame {
            return Err("Clips are not contiguous in source time");
        }

        self.clips[pos1].out_frame = self.clips[pos2].out_frame;
        self.clips.remove(pos2);
        Ok(())
    }

    /// Total timeline frame count across all sequential clips on primary track 0.
    pub fn total_timeline_frames(&self) -> usize {
        self.clips
            .iter()
            .filter(|c| c.track_index == 0)
            .map(|c| c.duration_frames())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeline_clip_duration_and_trim() {
        let mut clip = TimelineClip::new(1, PathBuf::from("video.mp4"), 30.0, 300);
        assert_eq!(clip.duration_frames(), 300);
        assert!((clip.duration_seconds() - 10.0).abs() < 1e-4);

        clip.set_trim(50, 200).unwrap();
        assert_eq!(clip.duration_frames(), 150);
        assert!((clip.duration_seconds() - 5.0).abs() < 1e-4);

        assert!(clip.set_trim(200, 100).is_err());
        assert!(clip.set_trim(0, 400).is_err());
    }

    #[test]
    fn test_timeline_editor_cut_and_merge() {
        let mut editor = TimelineEditor::new();
        let clip_id = editor.add_clip("action.mp4", 60.0, 600);
        assert_eq!(editor.total_timeline_frames(), 600);

        // Cut clip at frame offset 200
        let new_clip_id = editor.cut_clip(clip_id, 200).unwrap();
        assert_eq!(editor.clips.len(), 2);
        assert_eq!(editor.clips[0].duration_frames(), 200);
        assert_eq!(editor.clips[1].duration_frames(), 400);
        assert_eq!(editor.total_timeline_frames(), 600);

        // Merge the two contiguous clips back together
        editor.merge_clips(clip_id, new_clip_id).unwrap();
        assert_eq!(editor.clips.len(), 1);
        assert_eq!(editor.clips[0].duration_frames(), 600);
    }

    #[test]
    fn test_merge_discontiguous_fails() {
        let mut editor = TimelineEditor::new();
        let c1 = editor.add_clip("part1.mp4", 30.0, 100);
        let c2 = editor.add_clip("part2.mp4", 30.0, 100);

        // Different file paths -> merge fails
        assert!(editor.merge_clips(c1, c2).is_err());
    }
}
