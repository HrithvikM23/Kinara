from __future__ import annotations

import copy
from statistics import median

from config import BODY_LANDMARKS, HAND_LANDMARKS
from network.packet_builder import build_person_payload
from utils.math_utils import normalize, vector_between, vector_length


BODY_VISIBILITY_THRESHOLD = 0.45
HAND_CONFIDENCE_THRESHOLD = 0.35
MAX_PERSON_GAP_FRAMES = 4
MAX_INTERPOLATION_GAP_FRAMES = 8
MAX_HOLD_GAP_FRAMES = 3
LIMB_LENGTH_TOLERANCE_RATIO = 0.18
EPSILON = 1e-6

BODY_CHAINS = (
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
)

BODY_SEGMENTS = tuple(
    (chain[index], chain[index + 1])
    for chain in BODY_CHAINS
    for index in range(len(chain) - 1)
)


def repair_motion_frames(frames: list[dict], source_fps: float | None = None) -> list[dict]:
    repaired_frames = copy.deepcopy(frames)
    person_ids = sorted({person["id"] for frame in repaired_frames for person in frame.get("persons", [])})

    for person_id in person_ids:
        sequence = [_extract_person(frame, person_id) for frame in repaired_frames]
        sequence = _fill_missing_people(sequence)
        if not any(person is not None for person in sequence):
            continue

        _repair_body_joints(sequence)
        _repair_hand_joints(sequence, "left_hand")
        _repair_hand_joints(sequence, "right_hand")
        _enforce_body_constraints(sequence)
        _store_person_sequence(repaired_frames, person_id, sequence)

    return repaired_frames


def _extract_person(frame: dict, person_id: int):
    for payload in frame.get("persons", []):
        if int(payload.get("id", -1)) == person_id:
            return _payload_to_person(payload)
    return None


def _payload_to_person(payload: dict) -> dict:
    return {
        "id": int(payload.get("id", 0)),
        "body": copy.deepcopy(payload.get("body", {}).get("joints", {})),
        "left_hand": copy.deepcopy(payload.get("left_hand", {}).get("joints", {})),
        "right_hand": copy.deepcopy(payload.get("right_hand", {}).get("joints", {})),
        "left_hand_confidence": payload.get("left_hand", {}).get("confidence"),
        "right_hand_confidence": payload.get("right_hand", {}).get("confidence"),
    }


def _fill_missing_people(sequence: list[dict | None]) -> list[dict | None]:
    filled = list(sequence)
    for frame_index, person in enumerate(filled):
        if person is not None:
            continue

        prev_index, prev_person = _nearest_person(filled, frame_index, -1)
        next_index, next_person = _nearest_person(filled, frame_index, 1)
        if prev_person is not None and next_person is not None:
            if (next_index - prev_index - 1) <= MAX_PERSON_GAP_FRAMES:
                alpha = (frame_index - prev_index) / max(next_index - prev_index, 1)
                filled[frame_index] = _interpolate_person(prev_person, next_person, alpha, source="repair_person_interp")
                continue

        if prev_person is not None and (frame_index - prev_index) <= MAX_HOLD_GAP_FRAMES:
            filled[frame_index] = _clone_person(prev_person, source="repair_person_hold")
            continue

        if next_person is not None and (next_index - frame_index) <= MAX_HOLD_GAP_FRAMES:
            filled[frame_index] = _clone_person(next_person, source="repair_person_prefill")

    return filled


def _repair_body_joints(sequence: list[dict | None]) -> None:
    for joint_name in BODY_LANDMARKS:
        _repair_joint_series(sequence, "body", joint_name, _body_joint_is_valid)


def _repair_hand_joints(sequence: list[dict | None], side_key: str) -> None:
    confidence_key = f"{side_key}_confidence"
    for joint_name in HAND_LANDMARKS:
        _repair_joint_series(
            sequence,
            side_key,
            joint_name,
            lambda person, section_name, current_joint_name: _hand_joint_is_valid(person, section_name, current_joint_name),
        )

    for person in sequence:
        if person is None:
            continue
        joints = person.get(side_key, {})
        if any(joints.get(joint_name) is not None for joint_name in HAND_LANDMARKS):
            confidence = person.get(confidence_key)
            if confidence is None or float(confidence) < HAND_CONFIDENCE_THRESHOLD:
                person[confidence_key] = round(HAND_CONFIDENCE_THRESHOLD, 6)


def _repair_joint_series(sequence: list[dict | None], section_name: str, joint_name: str, validity_fn) -> None:
    for frame_index, person in enumerate(sequence):
        if person is None:
            continue

        current_joint = person.get(section_name, {}).get(joint_name)
        if validity_fn(person, section_name, joint_name):
            continue

        prev_index, prev_person = _nearest_valid_joint(sequence, frame_index, section_name, joint_name, -1, validity_fn)
        next_index, next_person = _nearest_valid_joint(sequence, frame_index, section_name, joint_name, 1, validity_fn)

        repaired_joint = None
        if prev_person is not None and next_person is not None:
            gap = next_index - prev_index - 1
            if gap <= MAX_INTERPOLATION_GAP_FRAMES:
                alpha = (frame_index - prev_index) / max(next_index - prev_index, 1)
                repaired_joint = _interpolate_joint(
                    prev_person[section_name][joint_name],
                    next_person[section_name][joint_name],
                    alpha,
                    source=f"repair_interp_{section_name}",
                    body_joint=section_name == "body",
                )

        if repaired_joint is None and prev_person is not None and (frame_index - prev_index) <= MAX_HOLD_GAP_FRAMES:
            repaired_joint = _clone_joint(
                prev_person[section_name][joint_name],
                source=f"repair_hold_{section_name}",
                body_joint=section_name == "body",
            )

        if repaired_joint is None and next_person is not None and (next_index - frame_index) <= MAX_HOLD_GAP_FRAMES:
            repaired_joint = _clone_joint(
                next_person[section_name][joint_name],
                source=f"repair_prefill_{section_name}",
                body_joint=section_name == "body",
            )

        if repaired_joint is not None:
            person[section_name][joint_name] = repaired_joint


def _enforce_body_constraints(sequence: list[dict | None]) -> None:
    reference_lengths = _estimate_reference_lengths(sequence)
    reference_directions = _estimate_reference_directions(sequence)

    for frame_index, person in enumerate(sequence):
        if person is None:
            continue

        body = person.get("body", {})
        for chain in BODY_CHAINS:
            for segment_index in range(len(chain) - 1):
                parent_name = chain[segment_index]
                child_name = chain[segment_index + 1]
                parent_joint = body.get(parent_name)
                child_joint = body.get(child_name)
                target_length = reference_lengths.get((parent_name, child_name))
                if parent_joint is None or target_length is None:
                    continue

                direction = None
                if child_joint is not None:
                    direction_vector = vector_between(parent_joint, child_joint)
                    if direction_vector is not None and vector_length(direction_vector) > EPSILON:
                        direction = normalize(direction_vector)

                if direction is None:
                    direction = _lookup_segment_direction(sequence, frame_index, parent_name, child_name)
                if direction is None:
                    direction = reference_directions.get((parent_name, child_name))
                if direction is None:
                    continue

                should_rebuild = child_joint is None
                if child_joint is not None:
                    current_length = vector_length(vector_between(parent_joint, child_joint))
                    should_rebuild = abs(current_length - target_length) > (target_length * LIMB_LENGTH_TOLERANCE_RATIO)

                if should_rebuild:
                    body[child_name] = _joint_from_parent(
                        parent_joint,
                        direction,
                        target_length,
                        source="repair_limb_lock",
                    )


def _estimate_reference_lengths(sequence: list[dict | None]) -> dict[tuple[str, str], float]:
    lengths = {}
    for parent_name, child_name in BODY_SEGMENTS:
        samples = []
        for person in sequence:
            if person is None:
                continue
            parent_joint = person.get("body", {}).get(parent_name)
            child_joint = person.get("body", {}).get(child_name)
            if parent_joint is None or child_joint is None:
                continue
            vector = vector_between(parent_joint, child_joint)
            length = vector_length(vector)
            if length > EPSILON:
                samples.append(length)
        if samples:
            lengths[(parent_name, child_name)] = float(median(samples))
    return lengths


def _estimate_reference_directions(sequence: list[dict | None]) -> dict[tuple[str, str], tuple[float, float, float]]:
    directions = {}
    for parent_name, child_name in BODY_SEGMENTS:
        samples = []
        for person in sequence:
            if person is None:
                continue
            parent_joint = person.get("body", {}).get(parent_name)
            child_joint = person.get("body", {}).get(child_name)
            if parent_joint is None or child_joint is None:
                continue
            direction = normalize(vector_between(parent_joint, child_joint))
            if direction is not None and vector_length(direction) > EPSILON:
                samples.append(direction)
        if samples:
            average_direction = normalize(
                (
                    sum(direction[0] for direction in samples) / len(samples),
                    sum(direction[1] for direction in samples) / len(samples),
                    sum(direction[2] for direction in samples) / len(samples),
                )
            )
            if average_direction is not None:
                directions[(parent_name, child_name)] = average_direction
    return directions


def _lookup_segment_direction(sequence: list[dict | None], frame_index: int, parent_name: str, child_name: str):
    prev_index, prev_person = _nearest_person(sequence, frame_index, -1)
    while prev_person is not None:
        direction = _segment_direction(prev_person, parent_name, child_name)
        if direction is not None:
            return direction
        prev_index, prev_person = _nearest_person(sequence, prev_index, -1)

    next_index, next_person = _nearest_person(sequence, frame_index, 1)
    while next_person is not None:
        direction = _segment_direction(next_person, parent_name, child_name)
        if direction is not None:
            return direction
        next_index, next_person = _nearest_person(sequence, next_index, 1)

    return None


def _segment_direction(person: dict, parent_name: str, child_name: str):
    parent_joint = person.get("body", {}).get(parent_name)
    child_joint = person.get("body", {}).get(child_name)
    if parent_joint is None or child_joint is None:
        return None
    direction = normalize(vector_between(parent_joint, child_joint))
    if direction is None or vector_length(direction) <= EPSILON:
        return None
    return direction


def _store_person_sequence(frames: list[dict], person_id: int, sequence: list[dict | None]) -> None:
    for frame, person in zip(frames, sequence):
        if person is None:
            continue

        payload = build_person_payload(person)
        persons = frame.setdefault("persons", [])
        replaced = False
        for index, existing in enumerate(persons):
            if int(existing.get("id", -1)) == person_id:
                persons[index] = payload
                replaced = True
                break
        if not replaced:
            persons.append(payload)
        persons.sort(key=lambda record: int(record.get("id", 0)))


def _nearest_person(sequence: list[dict | None], frame_index: int, direction: int):
    cursor = frame_index + direction
    while 0 <= cursor < len(sequence):
        person = sequence[cursor]
        if person is not None:
            return cursor, person
        cursor += direction
    return None, None


def _nearest_valid_joint(sequence: list[dict | None], frame_index: int, section_name: str, joint_name: str, direction: int, validity_fn):
    cursor = frame_index + direction
    while 0 <= cursor < len(sequence):
        person = sequence[cursor]
        if person is not None and validity_fn(person, section_name, joint_name):
            return cursor, person
        cursor += direction
    return None, None


def _body_joint_is_valid(person: dict, section_name: str, joint_name: str) -> bool:
    joint = person.get(section_name, {}).get(joint_name)
    if joint is None:
        return False
    return float(joint.get("visibility", 0.0) or 0.0) >= BODY_VISIBILITY_THRESHOLD


def _hand_joint_is_valid(person: dict, section_name: str, joint_name: str) -> bool:
    joint = person.get(section_name, {}).get(joint_name)
    if joint is None:
        return False
    confidence = person.get(f"{section_name}_confidence")
    return float(confidence or 0.0) >= HAND_CONFIDENCE_THRESHOLD


def _clone_person(person: dict, source: str) -> dict:
    cloned = {
        "id": int(person.get("id", 0)),
        "body": {joint_name: _clone_joint(joint, source=source, body_joint=True) for joint_name, joint in person.get("body", {}).items()},
        "left_hand": {joint_name: _clone_joint(joint, source=source, body_joint=False) for joint_name, joint in person.get("left_hand", {}).items()},
        "right_hand": {joint_name: _clone_joint(joint, source=source, body_joint=False) for joint_name, joint in person.get("right_hand", {}).items()},
        "left_hand_confidence": person.get("left_hand_confidence"),
        "right_hand_confidence": person.get("right_hand_confidence"),
    }
    return cloned


def _interpolate_person(person_a: dict, person_b: dict, alpha: float, source: str) -> dict:
    interpolated = {
        "id": int(person_a.get("id", person_b.get("id", 0))),
        "body": {},
        "left_hand": {},
        "right_hand": {},
        "left_hand_confidence": _interpolate_scalar(person_a.get("left_hand_confidence"), person_b.get("left_hand_confidence"), alpha),
        "right_hand_confidence": _interpolate_scalar(person_a.get("right_hand_confidence"), person_b.get("right_hand_confidence"), alpha),
    }

    for joint_name in BODY_LANDMARKS:
        interpolated["body"][joint_name] = _interpolate_joint(
            person_a.get("body", {}).get(joint_name),
            person_b.get("body", {}).get(joint_name),
            alpha,
            source=source,
            body_joint=True,
        )

    for joint_name in HAND_LANDMARKS:
        interpolated["left_hand"][joint_name] = _interpolate_joint(
            person_a.get("left_hand", {}).get(joint_name),
            person_b.get("left_hand", {}).get(joint_name),
            alpha,
            source=source,
            body_joint=False,
        )
        interpolated["right_hand"][joint_name] = _interpolate_joint(
            person_a.get("right_hand", {}).get(joint_name),
            person_b.get("right_hand", {}).get(joint_name),
            alpha,
            source=source,
            body_joint=False,
        )

    return interpolated


def _clone_joint(joint, source: str, body_joint: bool):
    if joint is None:
        return None
    cloned = {
        "x": round(float(joint["x"]), 6),
        "y": round(float(joint["y"]), 6),
        "z": round(float(joint["z"]), 6),
        "source": source,
    }
    if body_joint:
        cloned["visibility"] = round(max(float(joint.get("visibility", 0.0)), BODY_VISIBILITY_THRESHOLD), 6)
    return cloned


def _interpolate_joint(joint_a, joint_b, alpha: float, source: str, body_joint: bool):
    if joint_a is None and joint_b is None:
        return None
    if joint_a is None:
        return _clone_joint(joint_b, source=source, body_joint=body_joint)
    if joint_b is None:
        return _clone_joint(joint_a, source=source, body_joint=body_joint)

    repaired = {
        "x": round((float(joint_a["x"]) * (1.0 - alpha)) + (float(joint_b["x"]) * alpha), 6),
        "y": round((float(joint_a["y"]) * (1.0 - alpha)) + (float(joint_b["y"]) * alpha), 6),
        "z": round((float(joint_a["z"]) * (1.0 - alpha)) + (float(joint_b["z"]) * alpha), 6),
        "source": source,
    }
    if body_joint:
        visibility_a = float(joint_a.get("visibility", BODY_VISIBILITY_THRESHOLD))
        visibility_b = float(joint_b.get("visibility", BODY_VISIBILITY_THRESHOLD))
        repaired["visibility"] = round(max((visibility_a * (1.0 - alpha)) + (visibility_b * alpha), BODY_VISIBILITY_THRESHOLD), 6)
    return repaired


def _interpolate_scalar(value_a, value_b, alpha: float):
    if value_a is None and value_b is None:
        return None
    if value_a is None:
        return round(float(value_b), 6)
    if value_b is None:
        return round(float(value_a), 6)
    return round((float(value_a) * (1.0 - alpha)) + (float(value_b) * alpha), 6)


def _joint_from_parent(parent_joint, direction, length: float, source: str):
    return {
        "x": round(float(parent_joint["x"]) + (float(direction[0]) * float(length)), 6),
        "y": round(float(parent_joint["y"]) + (float(direction[1]) * float(length)), 6),
        "z": round(float(parent_joint["z"]) + (float(direction[2]) * float(length)), 6),
        "visibility": BODY_VISIBILITY_THRESHOLD,
        "source": source,
    }
