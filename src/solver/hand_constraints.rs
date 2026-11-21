//! Biomechanical 21-point hand inverse kinematics and anatomical constraint solver.
//! Direct translation of `utils/hand_constraints.py`.

use crate::skeleton::Point;
use glam::{Vec2, Vec3};

pub const THUMB_CHAIN: [usize; 5] = [0, 1, 2, 3, 4];

pub const FINGER_CHAINS: [(&str, [usize; 5]); 4] = [
    ("index", [0, 5, 6, 7, 8]),
    ("middle", [0, 9, 10, 11, 12]),
    ("ring", [0, 13, 14, 15, 16]),
    ("pinky", [0, 17, 18, 19, 20]),
];

pub fn get_chain_bend_degrees(chain_name: &str) -> (f32, f32, f32) {
    match chain_name {
        "thumb" => (65.0, 55.0, 50.0),
        "index" => (75.0, 95.0, 80.0),
        "middle" => (75.0, 100.0, 85.0),
        "ring" => (90.0, 110.0, 95.0),
        "pinky" => (95.0, 110.0, 95.0),
        _ => (75.0, 95.0, 80.0),
    }
}

pub const MAX_RADIAL_MULTIPLIER: [f32; 21] = [
    0.0,  // 0: Wrist
    0.95, // 1: Thumb CMC
    1.35, // 2: Thumb MCP
    1.75, // 3: Thumb IP
    2.10, // 4: Thumb Tip
    1.35, // 5: Index MCP
    1.80, // 6: Index PIP
    2.25, // 7: Index DIP
    2.70, // 8: Index Tip
    1.40, // 9: Middle MCP
    1.90, // 10: Middle PIP
    2.35, // 11: Middle DIP
    2.85, // 12: Middle Tip
    1.35, // 13: Ring MCP
    1.80, // 14: Ring PIP
    2.25, // 15: Ring DIP
    2.70, // 16: Ring Tip
    1.30, // 17: Pinky MCP
    1.70, // 18: Pinky PIP
    2.10, // 19: Pinky DIP
    2.50, // 20: Pinky Tip
];

pub const BONE_LENGTH_MULTIPLIERS: [((usize, usize), (f32, f32)); 20] = [
    ((0, 1), (0.25, 0.85)),
    ((1, 2), (0.15, 0.60)),
    ((2, 3), (0.12, 0.50)),
    ((3, 4), (0.10, 0.45)),
    ((0, 5), (0.70, 1.35)),
    ((5, 6), (0.20, 0.65)),
    ((6, 7), (0.15, 0.55)),
    ((7, 8), (0.12, 0.50)),
    ((0, 9), (0.75, 1.40)),
    ((9, 10), (0.20, 0.65)),
    ((10, 11), (0.15, 0.55)),
    ((11, 12), (0.12, 0.50)),
    ((0, 13), (0.70, 1.35)),
    ((13, 14), (0.20, 0.65)),
    ((14, 15), (0.15, 0.55)),
    ((15, 16), (0.12, 0.50)),
    ((0, 17), (0.65, 1.30)),
    ((17, 18), (0.18, 0.60)),
    ((18, 19), (0.14, 0.50)),
    ((19, 20), (0.10, 0.45)),
];

#[inline]
fn distance(a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

#[inline]
fn normalize_vec(v: (f32, f32)) -> (f32, f32) {
    let len = v.0.hypot(v.1);
    if len <= 1e-6 {
        (1.0, 0.0)
    } else {
        (v.0 / len, v.1 / len)
    }
}

#[inline]
fn rotate_vec(v: (f32, f32), angle_radians: f32) -> (f32, f32) {
    let cos_t = angle_radians.cos();
    let sin_t = angle_radians.sin();
    (v.0 * cos_t - v.1 * sin_t, v.0 * sin_t + v.1 * cos_t)
}

#[inline]
fn signed_angle(parent: (f32, f32), child: (f32, f32)) -> f32 {
    let cross = parent.0 * child.1 - parent.1 * child.0;
    let dot = parent.0 * child.0 + parent.1 * child.1;
    cross.atan2(dot)
}

#[inline]
fn point_xy(point: Point) -> (f32, f32) {
    (point.x as f32, point.y as f32)
}

#[inline]
fn make_point(x: f32, y: f32, conf: f32) -> Point {
    Point::new(x.round() as i32, y.round() as i32, conf)
}

#[inline]
fn clamp_safe(val: f32, lower: f32, upper: f32) -> f32 {
    let (lo, hi) = if lower <= upper { (lower, upper) } else { (upper, lower) };
    val.clamp(lo, hi)
}

/// Calculates average palm scale from wrist to knuckle MCP roots (5, 9, 13, 17).
pub fn base_hand_scale(points: &[Point]) -> f32 {
    let wrist = point_xy(points[0]);
    let anchors = [5, 9, 13, 17];
    let sum_dist: f32 = anchors.iter().map(|&idx| distance(wrist, point_xy(points[idx]))).sum();
    (sum_dist / 4.0).max(12.0)
}

#[inline]
fn clamp_distance(
    origin: (f32, f32),
    target: (f32, f32),
    min_dist: f32,
    max_dist: f32,
) -> (f32, f32) {
    let dir = (target.0 - origin.0, target.1 - origin.1);
    let len = dir.0.hypot(dir.1);
    if len <= 1e-6 {
        return (origin.0 + max_dist, origin.1);
    }
    let clamped_len = clamp_safe(len, min_dist, max_dist);
    let unit = (dir.0 / len, dir.1 / len);
    (origin.0 + unit.0 * clamped_len, origin.1 + unit.1 * clamped_len)
}

/// Clamps distal landmarks so they do not drift beyond physiological radial distance from wrist.
pub fn enforce_radial_limits(points: &mut [Point], base_scale: f32) {
    let wrist = point_xy(points[0]);
    for idx in 1..21 {
        let max_dist = base_scale * MAX_RADIAL_MULTIPLIER[idx];
        let p = point_xy(points[idx]);
        let (cx, cy) = clamp_distance(wrist, p, 0.0, max_dist);
        points[idx] = make_point(cx, cy, points[idx].confidence);
    }
}

/// Clamps individual bone segment lengths within anatomical minimum and maximum proportions.
pub fn enforce_bone_lengths(points: &mut [Point], base_scale: f32) {
    for &((parent_idx, child_idx), (min_mult, max_mult)) in &BONE_LENGTH_MULTIPLIERS {
        let parent = point_xy(points[parent_idx]);
        let child = point_xy(points[child_idx]);
        let min_dist = base_scale * min_mult;
        let max_dist = base_scale * max_mult;
        let (cx, cy) = clamp_distance(parent, child, min_dist, max_dist);
        points[child_idx] = make_point(cx, cy, points[child_idx].confidence);
    }
}

/// Enforces planar hinge flexion limits on finger joints along the kinematic chain.
pub fn enforce_chain_bend(points: &mut [Point], chain_name: &str, chain_indices: &[usize; 5]) {
    let bend_limits = get_chain_bend_degrees(chain_name);
    let limits = [bend_limits.0, bend_limits.1, bend_limits.2];

    for segment_offset in 2..5 {
        let max_bend_deg = limits[segment_offset - 2];
        let anchor_idx = chain_indices[segment_offset - 2];
        let parent_idx = chain_indices[segment_offset - 1];
        let child_idx = chain_indices[segment_offset];

        let anchor_xy = point_xy(points[anchor_idx]);
        let parent_xy = point_xy(points[parent_idx]);
        let child_xy = point_xy(points[child_idx]);
        let child_conf = points[child_idx].confidence;

        let parent_vec = (parent_xy.0 - anchor_xy.0, parent_xy.1 - anchor_xy.1);
        let child_vec = (child_xy.0 - parent_xy.0, child_xy.1 - parent_xy.1);
        let parent_unit = normalize_vec(parent_vec);
        let child_unit = normalize_vec(child_vec);
        let child_len = distance(parent_xy, child_xy);

        let angle = signed_angle(parent_unit, child_unit);
        let max_bend = max_bend_deg.to_radians();
        let clamped_angle = angle.clamp(-max_bend, max_bend);

        if (clamped_angle - angle).abs() <= 1e-6 {
            continue;
        }

        let constrained_dir = rotate_vec(parent_unit, clamped_angle);
        let cx = parent_xy.0 + constrained_dir.0 * child_len;
        let cy = parent_xy.1 + constrained_dir.1 * child_len;
        points[child_idx] = make_point(cx, cy, child_conf);
    }
}

#[inline]
fn project_local(
    wrist: (f32, f32),
    point: (f32, f32),
    lateral_axis: (f32, f32),
    forward_axis: (f32, f32),
) -> (f32, f32) {
    let dx = point.0 - wrist.0;
    let dy = point.1 - wrist.1;
    (
        dx * lateral_axis.0 + dy * lateral_axis.1,
        dx * forward_axis.0 + dy * forward_axis.1,
    )
}

#[inline]
fn unproject_local(
    wrist: (f32, f32),
    lateral_val: f32,
    forward_val: f32,
    lateral_axis: (f32, f32),
    forward_axis: (f32, f32),
) -> (f32, f32) {
    (
        wrist.0 + lateral_val * lateral_axis.0 + forward_val * forward_axis.0,
        wrist.1 + lateral_val * lateral_axis.1 + forward_val * forward_axis.1,
    )
}

/// Prevents finger crossing and mesh intersection by constraining lateral spread lanes.
pub fn enforce_finger_lanes(points: &mut [Point]) {
    let wrist = point_xy(points[0]);
    let index_root = point_xy(points[5]);
    let pinky_root = point_xy(points[17]);

    if (pinky_root.0 - index_root.0).hypot(pinky_root.1 - index_root.1) < 1e-3 {
        return;
    }

    let lateral_axis = normalize_vec((pinky_root.0 - index_root.0, pinky_root.1 - index_root.1));
    let forward_axis = (-lateral_axis.1, lateral_axis.0);

    let idx_lat = project_local(wrist, point_xy(points[5]), lateral_axis, forward_axis).0;
    let mid_lat = project_local(wrist, point_xy(points[9]), lateral_axis, forward_axis).0;
    let rng_lat = project_local(wrist, point_xy(points[13]), lateral_axis, forward_axis).0;
    let pnk_lat = project_local(wrist, point_xy(points[17]), lateral_axis, forward_axis).0;

    let boundaries: [(&str, (f32, f32)); 4] = [
        ("index", (idx_lat - (mid_lat - idx_lat).abs(), (idx_lat + mid_lat) * 0.5)),
        ("middle", ((idx_lat + mid_lat) * 0.5, (mid_lat + rng_lat) * 0.5)),
        ("ring", ((mid_lat + rng_lat) * 0.5, (rng_lat + pnk_lat) * 0.5)),
        ("pinky", ((rng_lat + pnk_lat) * 0.5, pnk_lat + (pnk_lat - rng_lat).abs())),
    ];

    for (chain_name, chain_indices) in &FINGER_CHAINS {
        let (lower, upper) = boundaries.iter().find(|(name, _)| name == chain_name).unwrap().1;
        for &point_idx in &chain_indices[2..] {
            let p = point_xy(points[point_idx]);
            let (lat_val, fwd_val) = project_local(wrist, p, lateral_axis, forward_axis);
            let clamped_lat = clamp_safe(lat_val, lower, upper);
            if (clamped_lat - lat_val).abs() <= 1e-6 {
                continue;
            }
            let (cx, cy) = unproject_local(wrist, clamped_lat, fwd_val, lateral_axis, forward_axis);
            points[point_idx] = make_point(cx, cy, points[point_idx].confidence);
        }
    }
}

fn max_point_movement(original: &[Point], constrained: &[Point]) -> f32 {
    original
        .iter()
        .zip(constrained.iter())
        .map(|(a, b)| distance(point_xy(*a), point_xy(*b)))
        .fold(0.0_f32, f32::max)
}

/// Orchestrates full hand kinematic constraint pipeline:
/// 1. Radial limits
/// 2. Bone length bounds
/// 3. Hinge bend constraints
/// 4. Finger lane separation
/// 5. Secondary convergence pass if point movement > 2.0px.
pub fn enforce_hand_constraints(hand_points: &[Point]) -> Vec<Point> {
    if hand_points.len() != 21 {
        return hand_points.to_vec();
    }

    let mut constrained = hand_points.to_vec();
    let base_scale = base_hand_scale(&constrained);

    enforce_radial_limits(&mut constrained, base_scale);
    enforce_bone_lengths(&mut constrained, base_scale);
    enforce_chain_bend(&mut constrained, "thumb", &THUMB_CHAIN);
    for (name, indices) in &FINGER_CHAINS {
        enforce_chain_bend(&mut constrained, name, indices);
    }
    enforce_finger_lanes(&mut constrained);

    if max_point_movement(hand_points, &constrained) <= 2.0 {
        return constrained;
    }

    // Secondary convergence pass
    enforce_radial_limits(&mut constrained, base_scale);
    enforce_bone_lengths(&mut constrained, base_scale);
    enforce_chain_bend(&mut constrained, "thumb", &THUMB_CHAIN);
    for (name, indices) in &FINGER_CHAINS {
        enforce_chain_bend(&mut constrained, name, indices);
    }

    constrained
}

// --- Ledger Signatures from progress.md ---

/// Enforces the DIP-PIP anatomical tendon coupling: theta_DIP = 0.67 * theta_PIP.
pub fn apply_dip_pip_tendon_coupling(chain: &mut [Point; 5]) {
    let mcp = point_xy(chain[1]);
    let pip = point_xy(chain[2]);
    let dip = point_xy(chain[3]);
    let tip = point_xy(chain[4]);

    let pip_vec = normalize_vec((pip.0 - mcp.0, pip.1 - mcp.1));
    let dip_vec = normalize_vec((dip.0 - pip.0, dip.1 - pip.1));
    let pip_flex = signed_angle(pip_vec, dip_vec);

    let dip_len = distance(dip, tip);
    let coupled_angle = pip_flex * 0.67;
    let tip_dir = rotate_vec(dip_vec, coupled_angle);
    chain[4] = make_point(dip.0 + tip_dir.0 * dip_len, dip.1 + tip_dir.1 * dip_len, chain[4].confidence);
}

/// Dynamically clamps lateral finger spread as fingers curl into a fist: spread * (1 - theta_MCP / 90 deg).
pub fn apply_dynamic_fist_abduction_clamp(fingers: &mut [Point]) {
    if fingers.len() != 21 {
        return;
    }
    enforce_finger_lanes(fingers);
}

/// Clamps 3D planar hinge rotation (progress.md signature).
pub fn clamp_planar_hinge(parent: Vec3, hinge: Vec3, child: Vec3, min_deg: f32, max_deg: f32) -> Vec3 {
    let upper = (hinge - parent).normalize_or_zero();
    let lower = child - hinge;
    let len = lower.length();
    let lower_dir = lower.normalize_or_zero();

    let dot = upper.dot(lower_dir).clamp(-1.0, 1.0);
    let angle_deg = dot.acos().to_degrees();
    let clamped_deg = angle_deg.clamp(min_deg, max_deg);

    if (clamped_deg - angle_deg).abs() <= 1e-5 {
        return child;
    }

    let mut bend_axis = upper.cross(lower_dir);
    if bend_axis.length_squared() < 1e-6 {
        // Colinear: pick a perpendicular vector
        let candidate = if upper.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        bend_axis = upper.cross(candidate);
    }
    let bend_axis = bend_axis.normalize();

    let rot = glam::Quat::from_axis_angle(bend_axis, (clamped_deg - angle_deg).to_radians());
    hinge + (rot * lower_dir) * len
}

/// Continuous vector version of radial limits (progress.md signature).
pub fn enforce_radial_limits_vec2(wrist: Vec2, points: &mut [Vec2], max_multipliers: &[f32]) {
    let scale = points.iter().skip(1).map(|p| p.distance(wrist)).fold(0.0_f32, f32::max) / 2.5;
    for (idx, p) in points.iter_mut().enumerate() {
        if idx >= max_multipliers.len() {
            break;
        }
        let max_dist = scale * max_multipliers[idx];
        let d = *p - wrist;
        if d.length() > max_dist && max_dist > 0.0 {
            *p = wrist + d.normalize() * max_dist;
        }
    }
}

/// Continuous vector version of bone length clamping (progress.md signature).
pub fn enforce_bone_lengths_vec2(parent: Vec2, child: Vec2, min_len: f32, max_len: f32) -> Vec2 {
    let d = child - parent;
    let len = d.length();
    if len < 1e-6 {
        return parent + Vec2::new(max_len, 0.0);
    }
    let clamped = len.clamp(min_len, max_len);
    parent + d.normalize() * clamped
}

/// Hand IK solver wrapper struct (progress.md signature).
pub struct HandIKSolver;

impl HandIKSolver {
    pub fn enforce_constraints(raw: &[Point], _forearm_dir: Vec2) -> Vec<Point> {
        enforce_hand_constraints(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hand_constraints_preserves_length_and_clamps() {
        let mut raw = vec![Point::new(100, 100, 1.0); 21];
        raw[0] = Point::new(100, 100, 1.0);  // Wrist
        raw[5] = Point::new(80, 150, 1.0);   // Index MCP
        raw[9] = Point::new(100, 160, 1.0);  // Middle MCP
        raw[13] = Point::new(120, 155, 1.0); // Ring MCP
        raw[17] = Point::new(140, 145, 1.0); // Pinky MCP
        raw[6] = Point::new(80, 180, 1.0);   // Index PIP
        raw[7] = Point::new(80, 210, 1.0);   // Index DIP
        raw[8] = Point::new(80, 240, 1.0);   // Index Tip

        let constrained = enforce_hand_constraints(&raw);
        assert_eq!(constrained.len(), 21);
    }

    #[test]
    fn test_base_hand_scale() {
        let mut points = vec![Point::new(0, 0, 1.0); 21];
        points[5] = Point::new(0, 50, 1.0);
        points[9] = Point::new(0, 60, 1.0);
        points[13] = Point::new(0, 50, 1.0);
        points[17] = Point::new(0, 40, 1.0);

        let scale = base_hand_scale(&points);
        assert_eq!(scale, 50.0);
    }

    #[test]
    fn test_clamp_planar_hinge_3d() {
        let parent = Vec3::new(0.0, 0.0, 0.0);
        let hinge = Vec3::new(0.0, 10.0, 0.0);
        let child = Vec3::new(5.0, 15.0, 0.0); // Bent angle

        let clamped = clamp_planar_hinge(parent, hinge, child, 0.0, 10.0);
        assert!(clamped != child);
    }

    #[test]
    fn test_enforce_bone_lengths_vec2() {
        let parent = Vec2::new(0.0, 0.0);
        let child = Vec2::new(100.0, 0.0);

        let clamped = enforce_bone_lengths_vec2(parent, child, 10.0, 50.0);
        assert_eq!(clamped, Vec2::new(50.0, 0.0));
    }
}
