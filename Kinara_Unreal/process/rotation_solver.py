from __future__ import annotations

from utils.math_utils import (
    average,
    cross,
    dot,
    normalize,
    quaternion_identity,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_from_forward_up,
    quaternion_to_euler_degrees,
    round_vector,
    vector_between,
    vector_length,
)


PRIMARY_CHILDREN = {
    "root": "spine_lower",
    "spine_lower": "spine_upper",
    "left_clavicle": "left_shoulder",
    "left_shoulder": "left_elbow",
    "left_elbow": "left_wrist",
    "right_clavicle": "right_shoulder",
    "right_shoulder": "right_elbow",
    "right_elbow": "right_wrist",
    "left_hip": "left_knee",
    "left_knee": "left_ankle",
    "left_ankle": "left_heel",
    "left_heel": "left_foot_index",
    "right_hip": "right_knee",
    "right_knee": "right_ankle",
    "right_ankle": "right_heel",
    "right_heel": "right_foot_index",
    "left_thumb_cmc": "left_thumb_mcp",
    "left_thumb_mcp": "left_thumb_ip",
    "left_thumb_ip": "left_thumb_tip",
    "left_index_finger_mcp": "left_index_finger_pip",
    "left_index_finger_pip": "left_index_finger_dip",
    "left_index_finger_dip": "left_index_finger_tip",
    "left_middle_finger_mcp": "left_middle_finger_pip",
    "left_middle_finger_pip": "left_middle_finger_dip",
    "left_middle_finger_dip": "left_middle_finger_tip",
    "left_ring_finger_mcp": "left_ring_finger_pip",
    "left_ring_finger_pip": "left_ring_finger_dip",
    "left_ring_finger_dip": "left_ring_finger_tip",
    "left_pinky_mcp": "left_pinky_pip",
    "left_pinky_pip": "left_pinky_dip",
    "left_pinky_dip": "left_pinky_tip",
    "right_thumb_cmc": "right_thumb_mcp",
    "right_thumb_mcp": "right_thumb_ip",
    "right_thumb_ip": "right_thumb_tip",
    "right_index_finger_mcp": "right_index_finger_pip",
    "right_index_finger_pip": "right_index_finger_dip",
    "right_index_finger_dip": "right_index_finger_tip",
    "right_middle_finger_mcp": "right_middle_finger_pip",
    "right_middle_finger_pip": "right_middle_finger_dip",
    "right_middle_finger_dip": "right_middle_finger_tip",
    "right_ring_finger_mcp": "right_ring_finger_pip",
    "right_ring_finger_pip": "right_ring_finger_dip",
    "right_ring_finger_dip": "right_ring_finger_tip",
    "right_pinky_mcp": "right_pinky_pip",
    "right_pinky_pip": "right_pinky_dip",
    "right_pinky_dip": "right_pinky_tip",
}


def _serialize_quaternion(quaternion):
    return [round(float(component), 6) for component in quaternion]


def _joint_position(skeleton: dict, joint_name: str):
    joint = skeleton["joints"].get(joint_name)
    if joint is None:
        return None
    return joint.get("position")


def _average_direction(vectors):
    normalized_vectors = []
    for vector in vectors:
        if vector is None or vector_length(vector) <= 0.0:
            continue
        normalized = normalize(vector)
        if normalized is not None:
            normalized_vectors.append(normalized)

    if not normalized_vectors:
        return None
    return normalize(
        (
            average(vector[0] for vector in normalized_vectors),
            average(vector[1] for vector in normalized_vectors),
            average(vector[2] for vector in normalized_vectors),
        )
    )


def _vector_for_joint(skeleton: dict, joint_name: str):
    joint = skeleton.get("joints", {}).get(joint_name)
    if joint is None:
        return None

    position = _joint_position(skeleton, joint_name)
    if position is None:
        return None

    if joint_name == "spine_upper":
        parent_position = _joint_position(skeleton, "spine_lower")
        if parent_position is not None:
            return normalize(vector_between(parent_position, position))

    child_name = PRIMARY_CHILDREN.get(joint_name)
    if child_name is not None:
        child_position = _joint_position(skeleton, child_name)
        if child_position is not None:
            return normalize(vector_between(position, child_position))

    children = joint.get("children", [])
    child_vectors = [vector_between(position, _joint_position(skeleton, child_name)) for child_name in children]
    forward = _average_direction(child_vectors)
    if forward is not None:
        return forward

    parent_name = joint.get("parent")
    if parent_name is not None:
        parent_position = _joint_position(skeleton, parent_name)
        if parent_position is not None:
            return normalize(vector_between(parent_position, position))

    return None


def _torso_side_vector(skeleton: dict):
    left_shoulder = _joint_position(skeleton, "left_shoulder")
    right_shoulder = _joint_position(skeleton, "right_shoulder")
    if left_shoulder is not None and right_shoulder is not None:
        return normalize(vector_between(left_shoulder, right_shoulder))
    left_hip = _joint_position(skeleton, "left_hip")
    right_hip = _joint_position(skeleton, "right_hip")
    if left_hip is not None and right_hip is not None:
        return normalize(vector_between(left_hip, right_hip))
    return None


def _wrist_plane_normal(skeleton: dict, side: str):
    wrist = _joint_position(skeleton, f"{side}_wrist")
    index_mcp = _joint_position(skeleton, f"{side}_index_finger_mcp")
    pinky_mcp = _joint_position(skeleton, f"{side}_pinky_mcp")
    if wrist is None or index_mcp is None or pinky_mcp is None:
        return None

    index_vector = vector_between(wrist, index_mcp)
    pinky_vector = vector_between(wrist, pinky_mcp)
    normal = cross(index_vector, pinky_vector)
    if vector_length(normal) <= 0.0:
        return None
    return normalize(normal)


def _up_hint_for_joint(skeleton: dict, joint_name: str, forward):
    side_vector = _torso_side_vector(skeleton)

    if joint_name in {"root", "spine_lower", "spine_upper"}:
        if side_vector is not None:
            torso_normal = cross(side_vector, forward)
            if vector_length(torso_normal) > 0.0:
                return normalize(torso_normal)
        return (0.0, 0.0, 1.0)

    if joint_name.startswith("left_") or joint_name.startswith("right_"):
        side = "left" if joint_name.startswith("left_") else "right"
        if joint_name.endswith("wrist"):
            palm_normal = _wrist_plane_normal(skeleton, side)
            if palm_normal is not None:
                return palm_normal
        if side_vector is not None:
            if side == "left":
                return tuple(-component for component in side_vector)
            return side_vector

    if joint_name.startswith("left_") or joint_name.startswith("right_"):
        wrist_normal = _wrist_plane_normal(skeleton, "left" if joint_name.startswith("left_") else "right")
        if wrist_normal is not None and abs(dot(wrist_normal, forward)) < 0.98:
            return wrist_normal

    return (0.0, 1.0, 0.0)


def build_joint_rotations(skeleton: dict) -> dict:
    rotations = {}

    for joint_name, joint in skeleton.get("joints", {}).items():
        if joint is None:
            rotations[joint_name] = None
            continue

        position = joint.get("position")
        if position is None:
            rotations[joint_name] = None
            continue

        forward = _vector_for_joint(skeleton, joint_name)
        if forward is None or vector_length(forward) <= 0.0:
            global_quaternion = quaternion_identity()
        else:
            up_hint = _up_hint_for_joint(skeleton, joint_name, forward)
            global_quaternion = quaternion_from_forward_up(forward, up_hint)

        parent_name = joint.get("parent")
        parent_rotation = None
        if parent_name is not None:
            parent_rotation = rotations.get(parent_name)

        parent_global_quaternion = None
        if isinstance(parent_rotation, dict):
            parent_global_quaternion = parent_rotation.get("global_quaternion")

        if parent_global_quaternion is None:
            local_quaternion = global_quaternion
        else:
            local_quaternion = quaternion_multiply(
                quaternion_inverse(tuple(parent_global_quaternion)),
                global_quaternion,
            )

        rotations[joint_name] = {
            "parent": parent_name,
            "global_quaternion": _serialize_quaternion(global_quaternion),
            "local_quaternion": _serialize_quaternion(local_quaternion),
            "global_euler": quaternion_to_euler_degrees(global_quaternion),
            "local_euler": quaternion_to_euler_degrees(local_quaternion),
            "forward": round_vector(forward),
        }

    return rotations
