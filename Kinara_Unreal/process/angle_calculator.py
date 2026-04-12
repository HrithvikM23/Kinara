from __future__ import annotations

from utils.math_utils import angle_between_points


ANGLE_TRIPLETS = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
    "left_shoulder": ("left_clavicle", "left_shoulder", "left_elbow"),
    "right_shoulder": ("right_clavicle", "right_shoulder", "right_elbow"),
    "left_hip": ("root", "left_hip", "left_knee"),
    "right_hip": ("root", "right_hip", "right_knee"),
    "left_ankle": ("left_knee", "left_ankle", "left_foot_index"),
    "right_ankle": ("right_knee", "right_ankle", "right_foot_index"),
    "left_wrist": ("left_elbow", "left_wrist", "left_middle_finger_mcp"),
    "right_wrist": ("right_elbow", "right_wrist", "right_middle_finger_mcp"),
    "left_index_finger_curl": ("left_index_finger_mcp", "left_index_finger_pip", "left_index_finger_tip"),
    "right_index_finger_curl": ("right_index_finger_mcp", "right_index_finger_pip", "right_index_finger_tip"),
    "left_middle_finger_curl": ("left_middle_finger_mcp", "left_middle_finger_pip", "left_middle_finger_tip"),
    "right_middle_finger_curl": ("right_middle_finger_mcp", "right_middle_finger_pip", "right_middle_finger_tip"),
}


def _joint_lookup(body: dict, skeleton: dict | None, joint_name: str):
    if skeleton is not None:
        joint_record = skeleton.get("joints", {}).get(joint_name)
        if joint_record is not None:
            return joint_record.get("position")
    return body.get(joint_name)


def build_joint_angles(body: dict, skeleton: dict | None = None) -> dict:
    angles = {}
    for angle_name, (joint_a, joint_b, joint_c) in ANGLE_TRIPLETS.items():
        angle = angle_between_points(
            _joint_lookup(body, skeleton, joint_a),
            _joint_lookup(body, skeleton, joint_b),
            _joint_lookup(body, skeleton, joint_c),
        )
        angles[angle_name] = round(angle, 3) if angle is not None else None
    return angles
