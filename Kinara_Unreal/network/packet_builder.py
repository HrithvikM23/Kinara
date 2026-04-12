from __future__ import annotations

import json

from config import BODY_LANDMARKS, HAND_LANDMARKS
from process.angle_calculator import build_joint_angles
from process.rotation_solver import build_joint_rotations
from process.skeleton_builder import build_bone_vectors, build_skeleton


def _round_optional(value):
    if value is None:
        return None
    return round(float(value), 6)


def _serialize_vector(values):
    if values is None:
        return None
    return [round(float(value), 6) for value in values]


def _serialize_joint(joint):
    if joint is None:
        return None

    payload = {
        "x": round(float(joint["x"]), 6),
        "y": round(float(joint["y"]), 6),
        "z": round(float(joint["z"]), 6),
    }

    if "visibility" in joint:
        payload["visibility"] = round(float(joint["visibility"]), 6)
    if "source" in joint:
        payload["source"] = joint["source"]

    return payload


def _serialize_joint_map(joints, ordered_names):
    joints = joints or {}
    return {name: _serialize_joint(joints.get(name)) for name in ordered_names}


def _serialize_skeleton(skeleton: dict) -> dict:
    payload = {
        "root": skeleton.get("root"),
        "joints": {},
        "bones": {},
    }

    for joint_name, joint in skeleton.get("joints", {}).items():
        payload["joints"][joint_name] = {
            "parent": joint.get("parent"),
            "children": list(joint.get("children", [])),
            "present": bool(joint.get("present")),
            "position": _serialize_joint(joint.get("position")),
            "local_offset": _serialize_vector(joint.get("local_offset")),
            "source": joint.get("source"),
        }

    for bone_name, bone in skeleton.get("bones", {}).items():
        if bone is None:
            payload["bones"][bone_name] = None
            continue
        payload["bones"][bone_name] = {
            "parent": bone.get("parent"),
            "child": bone.get("child"),
            "vector": _serialize_vector(bone.get("vector")),
            "direction": _serialize_vector(bone.get("direction")),
            "length": _round_optional(bone.get("length")),
        }

    return payload


def _serialize_rotations(rotations: dict) -> dict:
    payload = {}
    for joint_name, rotation in rotations.items():
        if rotation is None:
            payload[joint_name] = None
            continue
        payload[joint_name] = {
            "parent": rotation.get("parent"),
            "global_quaternion": _serialize_vector(rotation.get("global_quaternion")),
            "local_quaternion": _serialize_vector(rotation.get("local_quaternion")),
            "global_euler": _serialize_vector(rotation.get("global_euler")),
            "local_euler": _serialize_vector(rotation.get("local_euler")),
            "forward": _serialize_vector(rotation.get("forward")),
        }
    return payload


def _serialize_identity(identity: dict | None) -> dict | None:
    if not identity:
        return None
    return {
        "label": identity.get("label"),
        "profile_slot": identity.get("profile_slot"),
        "profile_color": identity.get("profile_color"),
        "profile_region": identity.get("profile_region"),
        "profile_score": _round_optional(identity.get("profile_score")),
        "top_color": identity.get("top_color"),
        "torso_color": identity.get("torso_color"),
        "yolo_track_id": identity.get("yolo_track_id"),
        "seen_since_frame": identity.get("seen_since_frame"),
        "last_seen_frame": identity.get("last_seen_frame"),
        "seen_since_timestamp_ms": identity.get("seen_since_timestamp_ms"),
        "last_seen_timestamp_ms": identity.get("last_seen_timestamp_ms"),
    }


def _present(section_map) -> bool:
    return any(value is not None for value in section_map.values())


def build_person_payload(person: dict) -> dict:
    body_joints = _serialize_joint_map(person.get("body"), BODY_LANDMARKS)
    left_hand_joints = _serialize_joint_map(person.get("left_hand"), HAND_LANDMARKS)
    right_hand_joints = _serialize_joint_map(person.get("right_hand"), HAND_LANDMARKS)
    skeleton = build_skeleton(person)
    rotations = build_joint_rotations(skeleton)

    return {
        "id": int(person.get("id", 0)),
        "identity": _serialize_identity(person.get("identity")),
        "body": {
            "present": _present(body_joints),
            "joints": body_joints,
        },
        "left_hand": {
            "present": _present(left_hand_joints),
            "confidence": _round_optional(person.get("left_hand_confidence")),
            "joints": left_hand_joints,
        },
        "right_hand": {
            "present": _present(right_hand_joints),
            "confidence": _round_optional(person.get("right_hand_confidence")),
            "joints": right_hand_joints,
        },
        "bones": build_bone_vectors(person.get("body", {})),
        "angles": build_joint_angles(person.get("body", {}), skeleton=skeleton),
        "skeleton": _serialize_skeleton(skeleton),
        "rotations": _serialize_rotations(rotations),
    }


def build_packet(persons: list, frame_index: int, timestamp_ms: int, source_fps: float) -> bytes:
    packet = {
        "frame": frame_index,
        "timestamp_ms": int(timestamp_ms),
        "source_fps": round(float(source_fps), 3),
        "count": len(persons),
        "persons": [build_person_payload(person) for person in persons],
    }

    return json.dumps(packet, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
