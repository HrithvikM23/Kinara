from __future__ import annotations

from dataclasses import dataclass


BODY_TORSO_POINTS = (5, 6, 11, 12)
CAMERA_VIEW_WEIGHTS = {
    "FRONT": 1.20,
    "BACK": 1.00,
    "LEFT": 1.05,
    "RIGHT": 1.05,
}


@dataclass(frozen=True, slots=True)
class FrameReference:
    origin_x: float
    origin_y: float
    scale: float


def compute_body_reference(points, threshold: float) -> FrameReference | None:
    confident = [points[index] for index in BODY_TORSO_POINTS if points[index][2] > threshold]
    if len(confident) < 2:
        return None

    xs = [point[0] for point in confident]
    ys = [point[1] for point in confident]
    origin_x = sum(xs) / len(xs)
    origin_y = sum(ys) / len(ys)
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    return FrameReference(origin_x=origin_x, origin_y=origin_y, scale=scale)


def compute_hand_reference(hand_payload, threshold: float) -> FrameReference | None:
    if hand_payload is None:
        return None

    points = hand_payload["points"]
    confident = [point for point in points if point[2] > threshold]
    if len(confident) < 2:
        return None

    wrist_x, wrist_y, _ = points[0]
    x1, y1, x2, y2 = hand_payload["box"]
    scale = max(x2 - x1, y2 - y1, 1)
    return FrameReference(origin_x=float(wrist_x), origin_y=float(wrist_y), scale=float(scale))


def project_points(points, source_reference: FrameReference, target_reference: FrameReference):
    projected = []
    for x, y, conf in points:
        local_x = (x - source_reference.origin_x) / source_reference.scale
        local_y = (y - source_reference.origin_y) / source_reference.scale
        projected_x = int(round(target_reference.origin_x + local_x * target_reference.scale))
        projected_y = int(round(target_reference.origin_y + local_y * target_reference.scale))
        projected.append((projected_x, projected_y, float(conf)))
    return projected


def _choose_reference(camera_points_by_label, threshold: float, reference_label: str, reference_builder):
    preferred = camera_points_by_label.get(reference_label)
    if preferred is not None:
        reference = reference_builder(preferred, threshold)
        if reference is not None:
            return preferred, reference

    for payload in camera_points_by_label.values():
        reference = reference_builder(payload, threshold)
        if reference is not None:
            return payload, reference
    return None, None


def fuse_body_views(camera_bodies: dict[str, list[tuple[int, int, float]]], threshold: float, reference_label: str = "FRONT"):
    reference_points, reference_frame = _choose_reference(camera_bodies, threshold, reference_label, compute_body_reference)
    if reference_points is None or reference_frame is None:
        return None

    projected_by_label: dict[str, list[tuple[int, int, float]]] = {}
    for label, points in camera_bodies.items():
        source_reference = compute_body_reference(points, threshold)
        if source_reference is None:
            continue
        projected_by_label[label] = project_points(points, source_reference, reference_frame)

    fused_points = []
    point_count = len(reference_points)
    for point_index in range(point_count):
        weighted_x = 0.0
        weighted_y = 0.0
        total_weight = 0.0
        best_conf = 0.0

        for label, points in projected_by_label.items():
            x, y, conf = points[point_index]
            if conf <= 0:
                continue
            weight = conf * CAMERA_VIEW_WEIGHTS.get(label, 1.0)
            weighted_x += x * weight
            weighted_y += y * weight
            total_weight += weight
            best_conf = max(best_conf, conf)

        if total_weight <= 0:
            fused_points.append(reference_points[point_index])
            continue

        fused_points.append(
            (
                int(round(weighted_x / total_weight)),
                int(round(weighted_y / total_weight)),
                float(best_conf),
            )
        )

    return fused_points


def fuse_hand_views(camera_hands: dict[str, dict], threshold: float, reference_label: str = "FRONT"):
    reference_hand, reference_frame = _choose_reference(camera_hands, threshold, reference_label, compute_hand_reference)
    if reference_hand is None or reference_frame is None:
        return None

    projected_by_label: dict[str, dict] = {}
    for label, hand_payload in camera_hands.items():
        source_reference = compute_hand_reference(hand_payload, threshold)
        if source_reference is None:
            continue
        projected_points = project_points(hand_payload["points"], source_reference, reference_frame)
        projected_box = hand_payload["box"]
        projected_by_label[label] = {"points": projected_points, "box": projected_box}

    fused_points = []
    point_count = len(reference_hand["points"])
    for point_index in range(point_count):
        weighted_x = 0.0
        weighted_y = 0.0
        total_weight = 0.0
        best_conf = 0.0

        for label, hand_payload in projected_by_label.items():
            x, y, conf = hand_payload["points"][point_index]
            if conf <= 0:
                continue
            weight = conf * CAMERA_VIEW_WEIGHTS.get(label, 1.0)
            weighted_x += x * weight
            weighted_y += y * weight
            total_weight += weight
            best_conf = max(best_conf, conf)

        if total_weight <= 0:
            fused_points.append(reference_hand["points"][point_index])
            continue

        fused_points.append(
            (
                int(round(weighted_x / total_weight)),
                int(round(weighted_y / total_weight)),
                float(best_conf),
            )
        )

    wrist_x, wrist_y, _ = fused_points[0]
    scale = int(round(reference_frame.scale))
    half = max(scale // 2, 1)
    fused_box = (
        int(round(wrist_x - half)),
        int(round(wrist_y - half)),
        int(round(wrist_x + half)),
        int(round(wrist_y + half)),
    )
    return {"points": fused_points, "box": fused_box}
