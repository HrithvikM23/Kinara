//! Standalone ASCII FBX 7.4.0 exporter with hierarchical `LimbNode` armature and curves.

use crate::export::canonical::{compute_rest_pose_joints, MultiPersonFrame};
use crate::export::coords::{align_to_ground_plane, convert_sequence_to_blender_space, normalize_sequence_scale, MotionFrame};
use crate::skeleton::{FBX_TIME_UNIT, SKELETON};
use glam::Vec3;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Exports motion frames to standalone ASCII FBX 7.4 with hierarchical `LimbNode` armature and curves.
/// Streams directly to disk via BufWriter to minimize memory footprint.
pub fn export_motion_fbx(
    path: impl AsRef<Path>,
    frames: &[MotionFrame],
    fps: f64,
) -> anyhow::Result<()> {
    let path = path.as_ref();
    if frames.is_empty() {
        return Ok(());
    }

    let rest = compute_rest_pose_joints(frames);

    // Compute localized joint maps (deltas from parent)
    let mut local_frames: Vec<HashMap<String, Vec3>> = Vec::with_capacity(frames.len());
    for f in frames {
        let mut local_map = HashMap::new();
        for spec in &SKELETON {
            if let Some(cj) = f.joints.get(spec.name) {
                if let Some(p_name) = spec.parent {
                    if let Some(pj) = f.joints.get(p_name) {
                        local_map.insert(
                            spec.name.to_string(),
                            Vec3::new(cj.x - pj.x, cj.y - pj.y, cj.z - pj.z),
                        );
                    } else {
                        local_map.insert(spec.name.to_string(), Vec3::new(cj.x, cj.y, cj.z));
                    }
                } else {
                    local_map.insert(spec.name.to_string(), Vec3::new(cj.x, cj.y, cj.z));
                }
            }
        }
        local_frames.push(local_map);
    }

    // Compute Euler rotations relative to rest vectors
    let mut rot_frames: Vec<HashMap<String, Vec3>> = Vec::with_capacity(frames.len());
    for local_map in &local_frames {
        let mut rot_map = HashMap::new();
        for spec in &SKELETON {
            if spec.parent.is_none() {
                rot_map.insert(spec.name.to_string(), Vec3::ZERO);
            } else if let (Some(cur_v), Some(rest_v)) = (local_map.get(spec.name), rest.rest_vectors.get(spec.name)) {
                let rot = compute_rotation_euler(*rest_v, *cur_v);
                rot_map.insert(spec.name.to_string(), rot);
            } else {
                rot_map.insert(spec.name.to_string(), Vec3::ZERO);
            }
        }
        rot_frames.push(rot_map);
    }

    // Assign consistent FBX IDs
    let mut model_ids: HashMap<&str, i64> = HashMap::new();
    let mut curve_node_ids: HashMap<(&str, &str), i64> = HashMap::new();
    let mut curve_ids: HashMap<(&str, &str, &str), i64> = HashMap::new();
    let animation_stack_id: i64 = 100000;
    let animation_layer_id: i64 = 100001;
    let mut next_id: i64 = 100100;

    for spec in &SKELETON {
        model_ids.insert(spec.name, next_id);
        next_id += 1;
    }

    for spec in &SKELETON {
        curve_node_ids.insert((spec.name, "T"), next_id);
        next_id += 1;
        for axis in ["X", "Y", "Z"] {
            curve_ids.insert((spec.name, axis, "T"), next_id);
            next_id += 1;
        }

        if spec.parent.is_some() {
            curve_node_ids.insert((spec.name, "R"), next_id);
            next_id += 1;
            for axis in ["X", "Y", "Z"] {
                curve_ids.insert((spec.name, axis, "R"), next_id);
                next_id += 1;
            }
        }
    }

    let key_times: Vec<i64> = (0..frames.len())
        .map(|idx| ((idx as f64 / fps.max(1.0)) * FBX_TIME_UNIT as f64).round() as i64)
        .collect();

    let tmp_path = PathBuf::from(format!("{}.tmp", path.display()));
    {
        let file = File::create(&tmp_path)?;
        let mut w = BufWriter::new(file);

        writeln!(w, "; FBX 7.4.0 project file")?;
        writeln!(w, "FBXHeaderExtension:  {{")?;
        writeln!(w, "  FBXHeaderVersion: 1003")?;
        writeln!(w, "  FBXVersion: 7400")?;
        writeln!(w, "  Creator: \"Kinara\"")?;
        writeln!(w, "}}")?;
        writeln!(w, "GlobalSettings:  {{")?;
        writeln!(w, "  Version: 1000")?;
        writeln!(w, "  Properties70:  {{")?;
        // Standard Blender Z-up, Y-forward convention
        writeln!(w, "    P: \"UpAxis\", \"int\", \"Integer\", \"\",2")?;
        writeln!(w, "    P: \"UpAxisSign\", \"int\", \"Integer\", \"\",1")?;
        writeln!(w, "    P: \"FrontAxis\", \"int\", \"Integer\", \"\",1")?;
        writeln!(w, "    P: \"FrontAxisSign\", \"int\", \"Integer\", \"\",1")?;
        writeln!(w, "    P: \"CoordAxis\", \"int\", \"Integer\", \"\",0")?;
        writeln!(w, "    P: \"CoordAxisSign\", \"int\", \"Integer\", \"\",1")?;
        writeln!(w, "    P: \"UnitScaleFactor\", \"double\", \"Number\", \"\",1")?;
        writeln!(w, "  }}")?;
        writeln!(w, "}}")?;

        writeln!(w, "Objects:  {{")?;
        writeln!(
            w,
            "  AnimationStack: {}, \"AnimStack::Take 001\", \"\" {{",
            animation_stack_id
        )?;
        writeln!(w, "    Properties70:  {{")?;
        writeln!(w, "      P: \"LocalStart\", \"KTime\", \"Time\", \"\",0")?;
        writeln!(
            w,
            "      P: \"LocalStop\", \"KTime\", \"Time\", \"\",{}",
            key_times.last().copied().unwrap_or(0)
        )?;
        writeln!(w, "    }}")?;
        writeln!(w, "  }}")?;

        writeln!(
            w,
            "  AnimationLayer: {}, \"AnimLayer::BaseLayer\", \"\" {{",
            animation_layer_id
        )?;
        writeln!(w, "  }}")?;

        for spec in &SKELETON {
            let m_id = model_ids[spec.name];
            writeln!(w, "  Model: {}, \"Model::{}\", \"LimbNode\" {{", m_id, spec.name)?;
            writeln!(w, "    Version: 232")?;
            writeln!(w, "    Properties70:  {{")?;
            writeln!(w, "      P: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",0,0,0")?;
            writeln!(w, "      P: \"Lcl Rotation\", \"Lcl Rotation\", \"\", \"A\",0,0,0")?;
            writeln!(w, "      P: \"Lcl Scaling\", \"Lcl Scaling\", \"\", \"A\",1,1,1")?;
            writeln!(w, "    }}")?;
            writeln!(w, "    Shading: T")?;
            writeln!(w, "    Culling: \"CullingOff\"")?;
            writeln!(w, "  }}")?;
        }

        // Write Translation and Rotation curves
        let time_str = key_times
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",");

        for spec in &SKELETON {
            for &(p_type, ref map_list) in &[("T", &local_frames), ("R", &rot_frames)] {
                if p_type == "R" && spec.parent.is_none() {
                    continue;
                }
                let node_id = curve_node_ids[&(spec.name, p_type)];
                writeln!(
                    w,
                    "  AnimationCurveNode: {}, \"AnimCurveNode::{}_{}\", \"\" {{",
                    node_id, spec.name, p_type
                )?;
                writeln!(w, "    Properties70:  {{")?;
                writeln!(w, "      P: \"d|X\", \"Number\", \"\", \"A\",0")?;
                writeln!(w, "      P: \"d|Y\", \"Number\", \"\", \"A\",0")?;
                writeln!(w, "      P: \"d|Z\", \"Number\", \"\", \"A\",0")?;
                writeln!(w, "    }}")?;
                writeln!(w, "  }}")?;

                for axis in ["X", "Y", "Z"] {
                    let curve_id = curve_ids[&(spec.name, axis, p_type)];
                    let num_keys = map_list.len();

                    writeln!(
                        w,
                        "  AnimationCurve: {}, \"AnimCurve::{}_{}_{}\", \"\" {{",
                        curve_id, spec.name, p_type, axis
                    )?;
                    writeln!(w, "    Default: 0")?;
                    writeln!(w, "    KeyVer: 4008")?;
                    writeln!(w, "    KeyTime: *{} {{", key_times.len())?;
                    writeln!(w, "      a: {}", time_str)?;
                    writeln!(w, "    }}")?;

                    // Stream float values directly without intermediate large buffer
                    write!(w, "    KeyValueFloat: *{} {{\n      a: ", num_keys)?;
                    for (i, m) in map_list.iter().enumerate() {
                        let v = m.get(spec.name).copied().unwrap_or(Vec3::ZERO);
                        let val = match axis {
                            "X" => v.x,
                            "Y" => v.y,
                            _ => v.z,
                        };
                        if i > 0 {
                            write!(w, ",")?;
                        }
                        write!(w, "{:.6}", val)?;
                    }
                    writeln!(w, "\n    }}")?;

                    // KeyAttrFlags
                    write!(w, "    KeyAttrFlags: *{} {{\n      a: ", num_keys)?;
                    for i in 0..num_keys {
                        if i > 0 {
                            write!(w, ",")?;
                        }
                        write!(w, "24836")?;
                    }
                    writeln!(w, "\n    }}")?;

                    // KeyAttrDataFloat
                    write!(w, "    KeyAttrDataFloat: *{} {{\n      a: ", num_keys * 4)?;
                    for i in 0..num_keys {
                        if i > 0 {
                            write!(w, ",")?;
                        }
                        write!(w, "0,0,0,0")?;
                    }
                    writeln!(w, "\n    }}")?;

                    // KeyAttrRefCount
                    write!(w, "    KeyAttrRefCount: *{} {{\n      a: ", num_keys)?;
                    for i in 0..num_keys {
                        if i > 0 {
                            write!(w, ",")?;
                        }
                        write!(w, "1")?;
                    }
                    writeln!(w, "\n    }}")?;

                    writeln!(w, "  }}")?;
                }
            }
        }
        writeln!(w, "}}")?;

        // Write Connections
        writeln!(w, "Connections:  {{")?;
        writeln!(
            w,
            "  C: \"OO\",{},{}",
            animation_layer_id, animation_stack_id
        )?;

        for spec in &SKELETON {
            let parent_id = match spec.parent {
                Some(p) => model_ids[p],
                None => 0,
            };
            writeln!(w, "  C: \"OO\",{},{}", model_ids[spec.name], parent_id)?;

            let t_node = curve_node_ids[&(spec.name, "T")];
            writeln!(w, "  C: \"OO\",{},{}", t_node, animation_layer_id)?;
            writeln!(
                w,
                "  C: \"OP\",{},{},\"Lcl Translation\"",
                t_node, model_ids[spec.name]
            )?;
            for axis in ["X", "Y", "Z"] {
                writeln!(
                    w,
                    "  C: \"OP\",{},{},\"d|{}\"",
                    curve_ids[&(spec.name, axis, "T")],
                    t_node,
                    axis
                )?;
            }

            if spec.parent.is_some() {
                let r_node = curve_node_ids[&(spec.name, "R")];
                writeln!(w, "  C: \"OO\",{},{}", r_node, animation_layer_id)?;
                writeln!(
                    w,
                    "  C: \"OP\",{},{},\"Lcl Rotation\"",
                    r_node, model_ids[spec.name]
                )?;
                for axis in ["X", "Y", "Z"] {
                    writeln!(
                        w,
                        "  C: \"OP\",{},{},\"d|{}\"",
                        curve_ids[&(spec.name, axis, "R")],
                        r_node,
                        axis
                    )?;
                }
            }
        }
        writeln!(w, "}}")?;
        w.flush()?;
    }

    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Splits multi-person timeline into individual per-person FBX files.
pub fn export_multi_person_bundle(
    base_path: impl AsRef<Path>,
    frames: &[MultiPersonFrame],
    fps: f64,
) -> anyhow::Result<Vec<PathBuf>> {
    let base_path = base_path.as_ref();
    if frames.is_empty() {
        return Ok(Vec::new());
    }

    // Collect all distinct person identifiers
    let mut person_tracks: HashMap<String, Vec<MotionFrame>> = HashMap::new();
    for f in frames {
        for (person_id, joints) in &f.people {
            person_tracks
                .entry(person_id.clone())
                .or_default()
                .push(MotionFrame::new(f.frame_index, joints.clone()));
        }
    }

    let stem = base_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("clip");
    let ext = base_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("fbx");
    let parent = base_path.parent().unwrap_or_else(|| Path::new("."));

    let mut exported = Vec::with_capacity(person_tracks.len());
    for (person_id, mut p_frames) in person_tracks {
        let clean_id: String = person_id
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
            .collect();
        let out_file = parent.join(format!("{}_{}.{}", stem, clean_id, ext));
        convert_sequence_to_blender_space(&mut p_frames);
        normalize_sequence_scale(&mut p_frames, 50.0);
        align_to_ground_plane(&mut p_frames);
        export_motion_fbx(&out_file, &p_frames, fps)?;
        exported.push(out_file);
    }

    Ok(exported)
}

pub(crate) fn compute_rotation_euler(rest_v: Vec3, cur_v: Vec3) -> Vec3 {
    if rest_v.length_squared() < 1e-12 || cur_v.length_squared() < 1e-12 {
        return Vec3::ZERO;
    }
    let q = glam::Quat::from_rotation_arc(rest_v.normalize(), cur_v.normalize());
    let (roll, pitch, yaw) = q.to_euler(glam::EulerRot::XYZ);
    Vec3::new(roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::canonical::{build_canonical_joint_map, HandsPair};
    use crate::export::coords::MotionFrame;
    use crate::skeleton::Point;

    fn dummy_body_keypoints() -> Vec<Point> {
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
        pts
    }

    #[test]
    fn test_export_motion_fbx() {
        let body = dummy_body_keypoints();
        let map = build_canonical_joint_map(&body, &HandsPair::default(), None);
        let frames = vec![MotionFrame::new(0, map)];

        let temp_dir = std::env::temp_dir();
        let out_path = temp_dir.join(format!("test_kinara_motion_{}.fbx", std::process::id()));

        export_motion_fbx(&out_path, &frames, 30.0).expect("Failed to export motion FBX");
        assert!(out_path.exists());

        let contents = std::fs::read_to_string(&out_path).expect("Failed to read FBX");
        assert!(contents.contains("FBX 7.4.0 project file"));
        assert!(contents.contains("Model::HipsRoot"));
        assert!(contents.contains("Model::Chest"));
        assert!(contents.contains("AnimStack::Take 001"));

        let _ = std::fs::remove_file(out_path);
    }
}
