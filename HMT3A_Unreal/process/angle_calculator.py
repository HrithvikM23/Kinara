from __future__ import annotations

from utils.math_utils import angle_between_points


ANGLE_TRIPLETS = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
}


def build_joint_angles(body: dict) -> dict:
    angles = {}
    for name, (joint_a, joint_b, joint_c) in ANGLE_TRIPLETS.items():
        angle = angle_between_points(body.get(joint_a), body.get(joint_b), body.get(joint_c))
        angles[name] = round(angle, 3) if angle is not None else None
    return angles
