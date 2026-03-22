from __future__ import annotations

from utils.math_utils import normalize, vector_between, vector_length


BONE_PAIRS = {
    "upper_arm_left": ("left_shoulder", "left_elbow"),
    "lower_arm_left": ("left_elbow", "left_wrist"),
    "upper_arm_right": ("right_shoulder", "right_elbow"),
    "lower_arm_right": ("right_elbow", "right_wrist"),
    "upper_leg_left": ("left_hip", "left_knee"),
    "lower_leg_left": ("left_knee", "left_ankle"),
    "upper_leg_right": ("right_hip", "right_knee"),
    "lower_leg_right": ("right_knee", "right_ankle"),
    "shoulder_line": ("left_shoulder", "right_shoulder"),
    "hip_line": ("left_hip", "right_hip"),
}


def _round_vector(vector):
    if vector is None:
        return None
    return [round(float(component), 6) for component in vector]


def _build_bone(start_joint, end_joint):
    if start_joint is None or end_joint is None:
        return None

    vector = vector_between(start_joint, end_joint)
    if vector is None:
        return None

    return {
        "vector": _round_vector(vector),
        "direction": _round_vector(normalize(vector)),
        "length": round(float(vector_length(vector)), 6),
    }


def build_bone_vectors(body: dict) -> dict:
    bones = {}
    for name, (start_joint, end_joint) in BONE_PAIRS.items():
        bones[name] = _build_bone(body.get(start_joint), body.get(end_joint))
    return bones
