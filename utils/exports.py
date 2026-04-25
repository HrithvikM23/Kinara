from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast


FBX_TIME_UNIT = 46186158000


@dataclass(frozen=True, slots=True)
class JointSpec:
    name: str
    parent: str | None


SKELETON: tuple[JointSpec, ...] = (
    JointSpec("HipsRoot", None),
    JointSpec("LeftHip", "HipsRoot"),
    JointSpec("LeftKnee", "LeftHip"),
    JointSpec("LeftAnkle", "LeftKnee"),
    JointSpec("RightHip", "HipsRoot"),
    JointSpec("RightKnee", "RightHip"),
    JointSpec("RightAnkle", "RightKnee"),
    JointSpec("Chest", "HipsRoot"),
    JointSpec("LeftShoulder", "Chest"),
    JointSpec("LeftElbow", "LeftShoulder"),
    JointSpec("LeftWrist", "LeftElbow"),
    JointSpec("LeftThumb1", "LeftWrist"),
    JointSpec("LeftThumb2", "LeftThumb1"),
    JointSpec("LeftThumb3", "LeftThumb2"),
    JointSpec("LeftThumb4", "LeftThumb3"),
    JointSpec("LeftIndex1", "LeftWrist"),
    JointSpec("LeftIndex2", "LeftIndex1"),
    JointSpec("LeftIndex3", "LeftIndex2"),
    JointSpec("LeftIndex4", "LeftIndex3"),
    JointSpec("LeftMiddle1", "LeftWrist"),
    JointSpec("LeftMiddle2", "LeftMiddle1"),
    JointSpec("LeftMiddle3", "LeftMiddle2"),
    JointSpec("LeftMiddle4", "LeftMiddle3"),
    JointSpec("LeftRing1", "LeftWrist"),
    JointSpec("LeftRing2", "LeftRing1"),
    JointSpec("LeftRing3", "LeftRing2"),
    JointSpec("LeftRing4", "LeftRing3"),
    JointSpec("LeftPinky1", "LeftWrist"),
    JointSpec("LeftPinky2", "LeftPinky1"),
    JointSpec("LeftPinky3", "LeftPinky2"),
    JointSpec("LeftPinky4", "LeftPinky3"),
    JointSpec("RightShoulder", "Chest"),
    JointSpec("RightElbow", "RightShoulder"),
    JointSpec("RightWrist", "RightElbow"),
    JointSpec("RightThumb1", "RightWrist"),
    JointSpec("RightThumb2", "RightThumb1"),
    JointSpec("RightThumb3", "RightThumb2"),
    JointSpec("RightThumb4", "RightThumb3"),
    JointSpec("RightIndex1", "RightWrist"),
    JointSpec("RightIndex2", "RightIndex1"),
    JointSpec("RightIndex3", "RightIndex2"),
    JointSpec("RightIndex4", "RightIndex3"),
    JointSpec("RightMiddle1", "RightWrist"),
    JointSpec("RightMiddle2", "RightMiddle1"),
    JointSpec("RightMiddle3", "RightMiddle2"),
    JointSpec("RightMiddle4", "RightMiddle3"),
    JointSpec("RightRing1", "RightWrist"),
    JointSpec("RightRing2", "RightRing1"),
    JointSpec("RightRing3", "RightRing2"),
    JointSpec("RightRing4", "RightRing3"),
    JointSpec("RightPinky1", "RightWrist"),
    JointSpec("RightPinky2", "RightPinky1"),
    JointSpec("RightPinky3", "RightPinky2"),
    JointSpec("RightPinky4", "RightPinky3"),
)

SKELETON_BY_NAME = {joint.name: joint for joint in SKELETON}
CHILDREN_BY_PARENT: dict[str | None, list[str]] = {}
for joint in SKELETON:
    CHILDREN_BY_PARENT.setdefault(joint.parent, []).append(joint.name)

BODY_NAME_TO_INDEX = {
    "LeftShoulder": 5,
    "RightShoulder": 6,
    "LeftElbow": 7,
    "RightElbow": 8,
    "LeftWrist": 9,
    "RightWrist": 10,
    "LeftHip": 11,
    "RightHip": 12,
    "LeftKnee": 13,
    "RightKnee": 14,
    "LeftAnkle": 15,
    "RightAnkle": 16,
}

HAND_NAME_TO_INDEX = {
    "Thumb1": 1,
    "Thumb2": 2,
    "Thumb3": 3,
    "Thumb4": 4,
    "Index1": 5,
    "Index2": 6,
    "Index3": 7,
    "Index4": 8,
    "Middle1": 9,
    "Middle2": 10,
    "Middle3": 11,
    "Middle4": 12,
    "Ring1": 13,
    "Ring2": 14,
    "Ring3": 15,
    "Ring4": 16,
    "Pinky1": 17,
    "Pinky2": 18,
    "Pinky3": 19,
    "Pinky4": 20,
}


def _to_world(x: int, y: int, z: float = 0.0) -> tuple[float, float, float]:
    return float(x), float(-y), float(z)


def _average_points(points: list[tuple[int, int, float]]) -> tuple[float, float, float, float]:
    point_count = len(points)
    sum_x = 0
    sum_y = 0
    sum_conf = 0.0
    for x, y, conf in points:
        sum_x += x
        sum_y += y
        sum_conf += conf
    x, y, z = _to_world(int(round(sum_x / point_count)), int(round(sum_y / point_count)))
    return x, y, z, float(sum_conf / point_count)


def _zero_joint() -> dict[str, float]:
    return {"x": 0.0, "y": 0.0, "z": 0.0, "confidence": 0.0}


def build_joint_map(
    body_points,
    hands_by_side,
    joint_depths: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    joint_map: dict[str, dict[str, float]] = {joint.name: _zero_joint() for joint in SKELETON}
    joint_depths = joint_depths or {}

    for name, index in BODY_NAME_TO_INDEX.items():
        x, y, conf = body_points[index]
        wx, wy, wz = _to_world(x, y, joint_depths.get(name, 0.0))
        joint_map[name] = {"x": wx, "y": wy, "z": wz, "confidence": float(conf)}

    hips = [body_points[11], body_points[12]]
    shoulders = [body_points[5], body_points[6]]
    root_x, root_y, _, root_conf = _average_points(hips)
    chest_x, chest_y, _, chest_conf = _average_points(shoulders)
    root_z = float((joint_depths.get("LeftHip", 0.0) + joint_depths.get("RightHip", 0.0)) * 0.5)
    chest_z = float((joint_depths.get("LeftShoulder", 0.0) + joint_depths.get("RightShoulder", 0.0)) * 0.5)
    joint_map["HipsRoot"] = {"x": root_x, "y": root_y, "z": root_z, "confidence": root_conf}
    joint_map["Chest"] = {"x": chest_x, "y": chest_y, "z": chest_z, "confidence": chest_conf}

    for side_label, hand_payload in (("Left", hands_by_side.get("left")), ("Right", hands_by_side.get("right"))):
        if hand_payload is None:
            continue
        hand_points = hand_payload["points"]
        for suffix, index in HAND_NAME_TO_INDEX.items():
            x, y, conf = hand_points[index]
            wx, wy, wz = _to_world(x, y, joint_depths.get(f"{side_label}{suffix}", 0.0))
            joint_map[f"{side_label}{suffix}"] = {"x": wx, "y": wy, "z": wz, "confidence": float(conf)}

    return joint_map


def _localize_joint_map(joint_map: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    local_map: dict[str, dict[str, float]] = {}
    for joint in SKELETON:
        current = joint_map[joint.name]
        if joint.parent is None:
            local_map[joint.name] = dict(current)
            continue
        parent = joint_map[joint.parent]
        local_map[joint.name] = {
            "x": current["x"] - parent["x"],
            "y": current["y"] - parent["y"],
            "z": current["z"] - parent["z"],
            "confidence": current["confidence"],
        }
    return local_map


def _frame_joint_map(frame: dict[str, object]) -> dict[str, dict[str, float]]:
    return cast(dict[str, dict[str, float]], frame["joints"])


def _ground_joint_frames(frames: list[dict[str, object]]) -> list[dict[str, object]]:
    min_y: float | None = None
    for frame in frames:
        joints = frame["joints"]
        if not isinstance(joints, dict):
            continue
        for joint in joints.values():
            if not isinstance(joint, dict):
                continue
            if float(joint.get("confidence", 0.0)) <= 0.0:
                continue
            joint_y = float(joint["y"])
            min_y = joint_y if min_y is None else min(min_y, joint_y)

    if min_y is None:
        return frames

    grounded_frames: list[dict[str, object]] = []
    for frame in frames:
        joints = frame["joints"]
        if not isinstance(joints, dict):
            grounded_frames.append(frame)
            continue

        grounded_joints: dict[str, dict[str, float]] = {}
        for name, joint in joints.items():
            if not isinstance(joint, dict):
                continue
            grounded_joints[name] = {
                "x": float(joint["x"]),
                "y": float(joint["y"]) - min_y,
                "z": float(joint["z"]),
                "confidence": float(joint["confidence"]),
            }

        grounded_frame = dict(frame)
        grounded_frame["joints"] = grounded_joints
        grounded_frames.append(grounded_frame)

    return grounded_frames


def _z_up_joint_frames(frames: list[dict[str, object]]) -> list[dict[str, object]]:
    grounded_frames = _ground_joint_frames(frames)
    z_up_frames: list[dict[str, object]] = []
    for frame in grounded_frames:
        joints = frame["joints"]
        if not isinstance(joints, dict):
            z_up_frames.append(frame)
            continue

        z_up_joints: dict[str, dict[str, float]] = {}
        for name, joint in joints.items():
            if not isinstance(joint, dict):
                continue
            z_up_joints[name] = {
                "x": float(joint["x"]),
                "y": float(joint["z"]),
                "z": float(joint["y"]),
                "confidence": float(joint["confidence"]),
            }

        z_up_frame = dict(frame)
        z_up_frame["joints"] = z_up_joints
        z_up_frames.append(z_up_frame)

    return z_up_frames


def _ground_z_axis_frames(frames: list[dict[str, object]]) -> list[dict[str, object]]:
    min_z: float | None = None
    for frame in frames:
        joints = frame["joints"]
        if not isinstance(joints, dict):
            continue
        for joint in joints.values():
            if not isinstance(joint, dict):
                continue
            if float(joint.get("confidence", 0.0)) <= 0.0:
                continue
            joint_z = float(joint["z"])
            min_z = joint_z if min_z is None else min(min_z, joint_z)

    if min_z is None:
        return frames

    grounded_frames: list[dict[str, object]] = []
    for frame in frames:
        joints = frame["joints"]
        if not isinstance(joints, dict):
            grounded_frames.append(frame)
            continue

        grounded_joints: dict[str, dict[str, float]] = {}
        for name, joint in joints.items():
            if not isinstance(joint, dict):
                continue
            grounded_joints[name] = {
                "x": float(joint["x"]),
                "y": float(joint["y"]),
                "z": float(joint["z"]) - min_z,
                "confidence": float(joint["confidence"]),
            }

        grounded_frame = dict(frame)
        grounded_frame["joints"] = grounded_joints
        grounded_frames.append(grounded_frame)

    return grounded_frames


def _normalize_export_frames(frames: list[dict[str, object]]) -> list[dict[str, object]]:
    return _ground_z_axis_frames(_z_up_joint_frames(frames))


def export_motion_json(
    output_path: Path,
    fps: float,
    frames: list[dict[str, object]],
    metadata: dict[str, object],
    frames_are_normalized: bool = False,
) -> None:
    if not frames_are_normalized:
        frames = _normalize_export_frames(frames)
    payload = {
        "format": "kinara-motion-json-v1",
        "fps": fps,
        "frame_count": len(frames),
        "metadata": metadata,
        "frames": frames,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def export_multi_person_json(
    output_path: Path,
    fps: float,
    frames: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    payload = {
        "format": "kinara-multi-person-json-v1",
        "fps": fps,
        "frame_count": len(frames),
        "metadata": metadata,
        "frames": frames,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _sanitize_person_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", label.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "person"


def _empty_joint_map() -> dict[str, dict[str, float]]:
    return {joint.name: _zero_joint() for joint in SKELETON}


def export_multi_person_fbx_bundle(
    output_path: Path,
    fps: float,
    frames: list[dict[str, object]],
) -> list[Path]:
    if not frames:
        return []

    person_frames: dict[str, list[dict[str, object]]] = {}

    for frame in frames:
        frame_index = int(frame.get("frame_index", 0))
        present_people = frame.get("people", [])
        keyed_people: dict[str, dict[str, object]] = {}
        if isinstance(present_people, list):
            for person in present_people:
                if not isinstance(person, dict):
                    continue
                label = str(person.get("label") or f"person{person.get('id', '0')}")
                person_key = _sanitize_person_label(label)
                joints = person.get("joints")
                if isinstance(joints, dict):
                    keyed_people[person_key] = cast(dict[str, object], joints)

        for person_key in set(person_frames) | set(keyed_people):
            joint_map = keyed_people.get(person_key, _empty_joint_map())
            person_frames.setdefault(person_key, []).append(
                {
                    "frame_index": frame_index,
                    "joints": joint_map,
                }
            )

    exported_paths: list[Path] = []
    suffix = output_path.suffix or ".fbx"
    stem = output_path.stem
    for person_key, person_motion_frames in person_frames.items():
        person_output_path = output_path.with_name(f"{stem}_{person_key}{suffix}")
        export_motion_fbx(person_output_path, fps, person_motion_frames)
        exported_paths.append(person_output_path)

    return exported_paths


def _compute_offsets(first_local_frame: dict[str, dict[str, float]]) -> dict[str, tuple[float, float, float]]:
    offsets = {}
    for joint in SKELETON:
        current = first_local_frame[joint.name]
        offsets[joint.name] = (current["x"], current["y"], current["z"])
    return offsets


def _build_bvh_hierarchy_lines(joint_name: str, offsets: dict[str, tuple[float, float, float]], indent: int = 0) -> list[str]:
    joint = SKELETON_BY_NAME[joint_name]
    prefix = "  " * indent
    lines: list[str] = []
    joint_label = "ROOT" if joint.parent is None else "JOINT"
    lines.append(f"{prefix}{joint_label} {joint_name}")
    lines.append(f"{prefix}{{")
    ox, oy, oz = offsets[joint_name]
    lines.append(f"{prefix}  OFFSET {ox:.6f} {oy:.6f} {oz:.6f}")
    if joint.parent is None:
        lines.append(
            f"{prefix}  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"
        )
    else:
        lines.append(f"{prefix}  CHANNELS 3 Xposition Yposition Zposition")

    children = CHILDREN_BY_PARENT.get(joint_name, [])
    if not children:
        lines.append(f"{prefix}  End Site")
        lines.append(f"{prefix}  {{")
        lines.append(f"{prefix}    OFFSET 0.000000 0.000000 0.000000")
        lines.append(f"{prefix}  }}")
    else:
        for child_name in children:
            lines.extend(_build_bvh_hierarchy_lines(child_name, offsets, indent + 1))
    lines.append(f"{prefix}}}")
    return lines


def export_motion_bvh(output_path: Path, fps: float, frames: list[dict[str, object]], frames_are_normalized: bool = False) -> None:
    if not frames:
        return

    if not frames_are_normalized:
        frames = _normalize_export_frames(frames)
    local_frames = [_localize_joint_map(_frame_joint_map(frame)) for frame in frames]
    offsets = _compute_offsets(local_frames[0])

    hierarchy_lines = ["HIERARCHY"]
    hierarchy_lines.extend(_build_bvh_hierarchy_lines("HipsRoot", offsets))
    hierarchy_lines.append("MOTION")
    hierarchy_lines.append(f"Frames: {len(local_frames)}")
    hierarchy_lines.append(f"Frame Time: {1.0 / max(fps, 1.0):.8f}")

    for local_frame in local_frames:
        values: list[str] = []
        for joint in SKELETON:
            current = local_frame[joint.name]
            values.append(f"{current['x']:.6f}")
            values.append(f"{current['y']:.6f}")
            values.append(f"{current['z']:.6f}")
            if joint.parent is None:
                values.extend(("0.000000", "0.000000", "0.000000"))
        hierarchy_lines.append(" ".join(values))

    output_path.write_text("\n".join(hierarchy_lines) + "\n", encoding="utf-8")


def _fbx_template_header() -> list[str]:
    return [
        '; FBX 7.4.0 project file',
        'FBXHeaderExtension:  {',
        '  FBXHeaderVersion: 1003',
        '  FBXVersion: 7400',
        '  Creator: "Kinara"',
        '}',
        'GlobalSettings:  {',
        '  Version: 1000',
        '  Properties70:  {',
        '    P: "UpAxis", "int", "Integer", "",2',
        '    P: "UpAxisSign", "int", "Integer", "",1',
        '    P: "FrontAxis", "int", "Integer", "",2',
        '    P: "FrontAxisSign", "int", "Integer", "",1',
        '    P: "CoordAxis", "int", "Integer", "",0',
        '    P: "CoordAxisSign", "int", "Integer", "",1',
        '    P: "UnitScaleFactor", "double", "Number", "",1',
        '  }',
        '}',
    ]


def export_motion_fbx(output_path: Path, fps: float, frames: list[dict[str, object]], frames_are_normalized: bool = False) -> None:
    if not frames:
        return

    if not frames_are_normalized:
        frames = _normalize_export_frames(frames)
    local_frames = [_localize_joint_map(_frame_joint_map(frame)) for frame in frames]
    model_ids: dict[str, int] = {}
    curve_node_ids: dict[str, int] = {}
    curve_ids: dict[tuple[str, str], int] = {}
    animation_stack_id = 100000
    animation_layer_id = 100001
    next_id = 100100

    for joint in SKELETON:
        model_ids[joint.name] = next_id
        next_id += 1
    for joint in SKELETON:
        curve_node_ids[joint.name] = next_id
        next_id += 1
        for axis in ("X", "Y", "Z"):
            curve_ids[(joint.name, axis)] = next_id
            next_id += 1

    key_times = [int(round((frame_index / max(fps, 1.0)) * FBX_TIME_UNIT)) for frame_index in range(len(local_frames))]

    lines = _fbx_template_header()
    lines.append("Objects:  {")
    lines.append(f'  AnimationStack: {animation_stack_id}, "AnimStack::Take 001", "" {{')
    lines.append("    Properties70:  {")
    lines.append('      P: "LocalStart", "KTime", "Time", "",0')
    lines.append(f'      P: "LocalStop", "KTime", "Time", "",{key_times[-1] if key_times else 0}')
    lines.append("    }")
    lines.append("  }")
    lines.append(f'  AnimationLayer: {animation_layer_id}, "AnimLayer::BaseLayer", "" {{')
    lines.append("  }")

    for joint in SKELETON:
        model_id = model_ids[joint.name]
        lines.append(f'  Model: {model_id}, "Model::{joint.name}", "LimbNode" {{')
        lines.append("    Version: 232")
        lines.append("    Properties70:  {")
        lines.append('      P: "Lcl Translation", "Lcl Translation", "", "A",0,0,0')
        lines.append('      P: "Lcl Rotation", "Lcl Rotation", "", "A",0,0,0')
        lines.append('      P: "Lcl Scaling", "Lcl Scaling", "", "A",1,1,1')
        lines.append("    }")
        lines.append("    Shading: T")
        lines.append('    Culling: "CullingOff"')
        lines.append("  }")

    for joint in SKELETON:
        curve_node_id = curve_node_ids[joint.name]
        lines.append(f'  AnimationCurveNode: {curve_node_id}, "AnimCurveNode::{joint.name}_T", "" {{')
        lines.append("    Properties70:  {")
        lines.append('      P: "d|X", "Number", "", "A",0')
        lines.append('      P: "d|Y", "Number", "", "A",0')
        lines.append('      P: "d|Z", "Number", "", "A",0')
        lines.append("    }")
        lines.append("  }")

        for axis in ("X", "Y", "Z"):
            curve_id = curve_ids[(joint.name, axis)]
            values = [local_frame[joint.name][axis.lower()] for local_frame in local_frames]
            lines.append(f'  AnimationCurve: {curve_id}, "AnimCurve::{joint.name}_T_{axis}", "" {{')
            lines.append("    Default: 0")
            lines.append("    KeyVer: 4008")
            lines.append(f"    KeyTime: *{len(key_times)} {{")
            lines.append("      a: " + ",".join(str(value) for value in key_times))
            lines.append("    }")
            lines.append(f"    KeyValueFloat: *{len(values)} {{")
            lines.append("      a: " + ",".join(f"{value:.6f}" for value in values))
            lines.append("    }")
            lines.append(f"    KeyAttrFlags: *{len(values)} {{")
            lines.append("      a: " + ",".join("24836" for _ in values))
            lines.append("    }")
            lines.append(f"    KeyAttrDataFloat: *{len(values) * 4} {{")
            lines.append("      a: " + ",".join("0,0,255790911,0" for _ in values))
            lines.append("    }")
            lines.append(f"    KeyAttrRefCount: *{len(values)} {{")
            lines.append("      a: " + ",".join("1" for _ in values))
            lines.append("    }")
            lines.append("  }")

    lines.append("}")
    lines.append("Connections:  {")
    lines.append(f'  C: "OO",{animation_layer_id},{animation_stack_id}')
    for joint in SKELETON:
        parent_id = 0 if joint.parent is None else model_ids[joint.parent]
        lines.append(f'  C: "OO",{model_ids[joint.name]},{parent_id}')
        lines.append(f'  C: "OO",{curve_node_ids[joint.name]},{animation_layer_id}')
        lines.append(f'  C: "OP",{curve_node_ids[joint.name]},{model_ids[joint.name]},"Lcl Translation"')
        for axis in ("X", "Y", "Z"):
            lines.append(f'  C: "OP",{curve_ids[(joint.name, axis)]},{curve_node_ids[joint.name]},"d|{axis}"')
    lines.append("}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
