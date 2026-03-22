from __future__ import annotations

import json

from config import BODY_LANDMARKS, HAND_LANDMARKS
from process.angle_calculator import build_joint_angles
from process.skeleton_builder import build_bone_vectors


def _round_optional(value):
    if value is None:
        return None
    return round(float(value), 6)


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

    return payload


def _serialize_joint_map(joints, ordered_names):
    joints = joints or {}
    return {name: _serialize_joint(joints.get(name)) for name in ordered_names}


def _present(section_map) -> bool:
    return any(value is not None for value in section_map.values())


def build_packet(persons: list, frame_index: int, timestamp_ms: int, source_fps: float) -> bytes:
    serialized_people = []

    for person in persons:
        body_joints = _serialize_joint_map(person.get("body"), BODY_LANDMARKS)
        left_hand_joints = _serialize_joint_map(person.get("left_hand"), HAND_LANDMARKS)
        right_hand_joints = _serialize_joint_map(person.get("right_hand"), HAND_LANDMARKS)

        serialized_people.append(
            {
                "id": int(person.get("id", len(serialized_people))),
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
                "angles": build_joint_angles(person.get("body", {})),
            }
        )

    packet = {
        "frame": frame_index,
        "timestamp_ms": int(timestamp_ms),
        "source_fps": round(float(source_fps), 3),
        "count": len(serialized_people),
        "persons": serialized_people,
    }

    return json.dumps(packet, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
