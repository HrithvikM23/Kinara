from __future__ import annotations

from utils.math_utils import midpoint, vector_between, vector_length, vector_to_point


BODY_BONE_PAIRS = {
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

SKELETON_PARENTS = {
    "root": None,
    "spine_lower": "root",
    "spine_upper": "spine_lower",
    "left_clavicle": "spine_upper",
    "left_shoulder": "left_clavicle",
    "left_elbow": "left_shoulder",
    "left_wrist": "left_elbow",
    "right_clavicle": "spine_upper",
    "right_shoulder": "right_clavicle",
    "right_elbow": "right_shoulder",
    "right_wrist": "right_elbow",
    "left_hip": "root",
    "left_knee": "left_hip",
    "left_ankle": "left_knee",
    "left_heel": "left_ankle",
    "left_foot_index": "left_heel",
    "right_hip": "root",
    "right_knee": "right_hip",
    "right_ankle": "right_knee",
    "right_heel": "right_ankle",
    "right_foot_index": "right_heel",
    "left_thumb_cmc": "left_wrist",
    "left_thumb_mcp": "left_thumb_cmc",
    "left_thumb_ip": "left_thumb_mcp",
    "left_thumb_tip": "left_thumb_ip",
    "left_index_finger_mcp": "left_wrist",
    "left_index_finger_pip": "left_index_finger_mcp",
    "left_index_finger_dip": "left_index_finger_pip",
    "left_index_finger_tip": "left_index_finger_dip",
    "left_middle_finger_mcp": "left_wrist",
    "left_middle_finger_pip": "left_middle_finger_mcp",
    "left_middle_finger_dip": "left_middle_finger_pip",
    "left_middle_finger_tip": "left_middle_finger_dip",
    "left_ring_finger_mcp": "left_wrist",
    "left_ring_finger_pip": "left_ring_finger_mcp",
    "left_ring_finger_dip": "left_ring_finger_pip",
    "left_ring_finger_tip": "left_ring_finger_dip",
    "left_pinky_mcp": "left_wrist",
    "left_pinky_pip": "left_pinky_mcp",
    "left_pinky_dip": "left_pinky_pip",
    "left_pinky_tip": "left_pinky_dip",
    "right_thumb_cmc": "right_wrist",
    "right_thumb_mcp": "right_thumb_cmc",
    "right_thumb_ip": "right_thumb_mcp",
    "right_thumb_tip": "right_thumb_ip",
    "right_index_finger_mcp": "right_wrist",
    "right_index_finger_pip": "right_index_finger_mcp",
    "right_index_finger_dip": "right_index_finger_pip",
    "right_index_finger_tip": "right_index_finger_dip",
    "right_middle_finger_mcp": "right_wrist",
    "right_middle_finger_pip": "right_middle_finger_mcp",
    "right_middle_finger_dip": "right_middle_finger_pip",
    "right_middle_finger_tip": "right_middle_finger_dip",
    "right_ring_finger_mcp": "right_wrist",
    "right_ring_finger_pip": "right_ring_finger_mcp",
    "right_ring_finger_dip": "right_ring_finger_pip",
    "right_ring_finger_tip": "right_ring_finger_dip",
    "right_pinky_mcp": "right_wrist",
    "right_pinky_pip": "right_pinky_mcp",
    "right_pinky_dip": "right_pinky_pip",
    "right_pinky_tip": "right_pinky_dip",
}

SKELETON_CHILDREN = {}
for joint_name, parent_name in SKELETON_PARENTS.items():
    SKELETON_CHILDREN.setdefault(joint_name, [])
    if parent_name is not None:
        SKELETON_CHILDREN.setdefault(parent_name, []).append(joint_name)

LEFT_HAND_CHAIN = {
    "thumb_cmc": "left_thumb_cmc",
    "thumb_mcp": "left_thumb_mcp",
    "thumb_ip": "left_thumb_ip",
    "thumb_tip": "left_thumb_tip",
    "index_finger_mcp": "left_index_finger_mcp",
    "index_finger_pip": "left_index_finger_pip",
    "index_finger_dip": "left_index_finger_dip",
    "index_finger_tip": "left_index_finger_tip",
    "middle_finger_mcp": "left_middle_finger_mcp",
    "middle_finger_pip": "left_middle_finger_pip",
    "middle_finger_dip": "left_middle_finger_dip",
    "middle_finger_tip": "left_middle_finger_tip",
    "ring_finger_mcp": "left_ring_finger_mcp",
    "ring_finger_pip": "left_ring_finger_pip",
    "ring_finger_dip": "left_ring_finger_dip",
    "ring_finger_tip": "left_ring_finger_tip",
    "pinky_mcp": "left_pinky_mcp",
    "pinky_pip": "left_pinky_pip",
    "pinky_dip": "left_pinky_dip",
    "pinky_tip": "left_pinky_tip",
}

RIGHT_HAND_CHAIN = {
    source_name: target_name.replace("left_", "right_", 1)
    for source_name, target_name in LEFT_HAND_CHAIN.items()
}


def _round_vector(vector):
    if vector is None:
        return None
    return [round(float(component), 6) for component in vector]


def _build_bone(parent_name: str, child_name: str, joints: dict):
    parent_joint = joints.get(parent_name)
    child_joint = joints.get(child_name)
    if parent_joint is None or child_joint is None:
        return None

    vector = vector_between(parent_joint, child_joint)
    if vector is None:
        return None

    length = vector_length(vector)
    if length <= 0.0:
        return None

    return {
        "parent": parent_name,
        "child": child_name,
        "vector": _round_vector(vector),
        "direction": _round_vector(tuple(component / length for component in vector)),
        "length": round(float(length), 6),
    }


def _copy_joint(joint, source: str):
    if joint is None:
        return None
    payload = {
        "x": round(float(joint["x"]), 6),
        "y": round(float(joint["y"]), 6),
        "z": round(float(joint["z"]), 6),
        "source": source,
    }
    if "visibility" in joint:
        payload["visibility"] = round(float(joint["visibility"]), 6)
    return payload


def _interpolate(point_a, point_b, factor: float, source: str):
    if point_a is None or point_b is None:
        return None
    vector = vector_between(point_a, point_b)
    if vector is None:
        return None
    blended = (
        float(point_a["x"]) + (vector[0] * factor),
        float(point_a["y"]) + (vector[1] * factor),
        float(point_a["z"]) + (vector[2] * factor),
    )
    return _copy_joint(vector_to_point(blended), source=source)


def _align_hand_joints(body_wrist, hand_joints: dict | None, source: str) -> dict[str, dict | None]:
    aligned = {}
    hand_joints = hand_joints or {}
    hand_wrist = hand_joints.get("wrist")

    for hand_joint_name, hand_joint in hand_joints.items():
        if hand_joint_name == "wrist":
            continue
        if hand_joint is None or hand_wrist is None or body_wrist is None:
            aligned[hand_joint_name] = None if hand_joint is None else _copy_joint(hand_joint, source=source)
            continue

        offset = vector_between(hand_wrist, hand_joint)
        if offset is None:
            aligned[hand_joint_name] = _copy_joint(hand_joint, source=source)
            continue

        aligned_joint = {
            "x": round(float(body_wrist["x"]) + offset[0], 6),
            "y": round(float(body_wrist["y"]) + offset[1], 6),
            "z": round(float(body_wrist["z"]) + offset[2], 6),
            "source": source,
        }
        aligned[hand_joint_name] = aligned_joint

    return aligned


def _build_joint_positions(person: dict) -> dict[str, dict | None]:
    body = person.get("body", {})
    left_hand = person.get("left_hand", {})
    right_hand = person.get("right_hand", {})

    joint_positions = {
        "root": _copy_joint(midpoint(body.get("left_hip"), body.get("right_hip")), source="synthetic_root"),
        "left_shoulder": _copy_joint(body.get("left_shoulder"), source="body"),
        "right_shoulder": _copy_joint(body.get("right_shoulder"), source="body"),
        "left_elbow": _copy_joint(body.get("left_elbow"), source="body"),
        "right_elbow": _copy_joint(body.get("right_elbow"), source="body"),
        "left_wrist": _copy_joint(body.get("left_wrist"), source="body"),
        "right_wrist": _copy_joint(body.get("right_wrist"), source="body"),
        "left_hip": _copy_joint(body.get("left_hip"), source="body"),
        "right_hip": _copy_joint(body.get("right_hip"), source="body"),
        "left_knee": _copy_joint(body.get("left_knee"), source="body"),
        "right_knee": _copy_joint(body.get("right_knee"), source="body"),
        "left_ankle": _copy_joint(body.get("left_ankle"), source="body"),
        "right_ankle": _copy_joint(body.get("right_ankle"), source="body"),
        "left_heel": _copy_joint(body.get("left_heel"), source="body"),
        "right_heel": _copy_joint(body.get("right_heel"), source="body"),
        "left_foot_index": _copy_joint(body.get("left_foot_index"), source="body"),
        "right_foot_index": _copy_joint(body.get("right_foot_index"), source="body"),
    }

    shoulder_center = midpoint(body.get("left_shoulder"), body.get("right_shoulder"))
    joint_positions["spine_lower"] = _interpolate(joint_positions.get("root"), shoulder_center, 0.4, source="synthetic_spine")
    joint_positions["spine_upper"] = _interpolate(joint_positions.get("root"), shoulder_center, 0.8, source="synthetic_spine")
    joint_positions["left_clavicle"] = _copy_joint(midpoint(joint_positions.get("spine_upper"), body.get("left_shoulder")), source="synthetic_clavicle")
    joint_positions["right_clavicle"] = _copy_joint(midpoint(joint_positions.get("spine_upper"), body.get("right_shoulder")), source="synthetic_clavicle")

    aligned_left = _align_hand_joints(joint_positions.get("left_wrist"), left_hand, source="left_hand")
    aligned_right = _align_hand_joints(joint_positions.get("right_wrist"), right_hand, source="right_hand")

    for hand_joint_name, skeleton_joint_name in LEFT_HAND_CHAIN.items():
        joint_positions[skeleton_joint_name] = aligned_left.get(hand_joint_name)

    for hand_joint_name, skeleton_joint_name in RIGHT_HAND_CHAIN.items():
        joint_positions[skeleton_joint_name] = aligned_right.get(hand_joint_name)

    return joint_positions


def build_skeleton(person: dict) -> dict:
    joint_positions = _build_joint_positions(person)
    joints = {}
    bones = {}

    for joint_name, parent_name in SKELETON_PARENTS.items():
        position = joint_positions.get(joint_name)
        local_offset = None
        if parent_name is not None and position is not None and joint_positions.get(parent_name) is not None:
            local_offset = _round_vector(vector_between(joint_positions[parent_name], position))

        joints[joint_name] = {
            "name": joint_name,
            "parent": parent_name,
            "children": list(SKELETON_CHILDREN.get(joint_name, [])),
            "present": position is not None,
            "position": position,
            "local_offset": local_offset,
            "source": position.get("source") if position is not None else None,
        }

        if parent_name is not None:
            bone_name = f"{parent_name}__{joint_name}"
            bones[bone_name] = _build_bone(parent_name, joint_name, joint_positions)

    return {
        "root": "root",
        "joints": joints,
        "bones": bones,
    }


def build_bone_vectors(body: dict) -> dict:
    bones = {}
    for bone_name, (start_joint, end_joint) in BODY_BONE_PAIRS.items():
        bones[bone_name] = _build_bone(start_joint, end_joint, body)
    return bones

