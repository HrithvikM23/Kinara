from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np


COLOR_RULES = {
    "red": [((0, 80, 50), (10, 255, 255)), ((170, 80, 50), (180, 255, 255))],
    "orange": [((11, 80, 50), (24, 255, 255))],
    "yellow": [((25, 80, 50), (34, 255, 255))],
    "green": [((35, 60, 40), (85, 255, 255))],
    "cyan": [((86, 60, 40), (100, 255, 255))],
    "blue": [((101, 70, 40), (130, 255, 255))],
    "purple": [((131, 70, 40), (155, 255, 255))],
    "pink": [((156, 60, 60), (169, 255, 255))],
    "white": [((0, 0, 180), (180, 70, 255))],
    "gray": [((0, 0, 70), (180, 60, 179))],
    "black": [((0, 0, 0), (180, 255, 69))],
}
REGION_WINDOWS = {
    "top": (0.0, 0.33),
    "torso": (0.2, 0.72),
    "full": (0.0, 1.0),
}
DEFAULT_REGION = "top"


def normalize_color_name(color_name: str | None) -> str:
    if not color_name:
        return "unknown"
    normalized = str(color_name).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in COLOR_RULES else normalized.replace("_", "")


def normalize_region_name(region_name: str | None) -> str:
    if not region_name:
        return DEFAULT_REGION
    normalized = str(region_name).strip().lower()
    return normalized if normalized in REGION_WINDOWS else DEFAULT_REGION


def clamp_bbox(bbox: dict | None, frame_width: int, frame_height: int) -> dict | None:
    if bbox is None:
        return None

    x0 = max(0, min(int(round(float(bbox.get("x0", 0)))), frame_width - 1))
    y0 = max(0, min(int(round(float(bbox.get("y0", 0)))), frame_height - 1))
    x1 = max(x0 + 1, min(int(round(float(bbox.get("x1", frame_width)))), frame_width))
    y1 = max(y0 + 1, min(int(round(float(bbox.get("y1", frame_height)))), frame_height))
    return build_bbox(x0, y0, x1, y1)


def build_bbox(x0: int, y0: int, x1: int, y1: int) -> dict:
    width = max(int(x1) - int(x0), 1)
    height = max(int(y1) - int(y0), 1)
    return {
        "x0": int(x0),
        "y0": int(y0),
        "x1": int(x1),
        "y1": int(y1),
        "width": width,
        "height": height,
        "cx": int(x0 + (width / 2)),
        "cy": int(y0 + (height / 2)),
    }


def expand_bbox(bbox: dict, frame_width: int, frame_height: int, scale: float = 0.12) -> dict:
    if bbox is None:
        return None

    pad_x = int(round(float(bbox["width"]) * float(scale)))
    pad_y = int(round(float(bbox["height"]) * float(scale)))
    return clamp_bbox(
        build_bbox(
            bbox["x0"] - pad_x,
            bbox["y0"] - pad_y,
            bbox["x1"] + pad_x,
            bbox["y1"] + pad_y,
        ),
        frame_width,
        frame_height,
    )


def bbox_iou(box_a: dict | None, box_b: dict | None) -> float:
    if box_a is None or box_b is None:
        return 0.0

    inter_x0 = max(box_a["x0"], box_b["x0"])
    inter_y0 = max(box_a["y0"], box_b["y0"])
    inter_x1 = min(box_a["x1"], box_b["x1"])
    inter_y1 = min(box_a["y1"], box_b["y1"])
    inter_w = max(inter_x1 - inter_x0, 0)
    inter_h = max(inter_y1 - inter_y0, 0)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(box_a["width"] * box_a["height"], 1)
    area_b = max(box_b["width"] * box_b["height"], 1)
    union = area_a + area_b - inter_area
    return float(inter_area) / max(float(union), 1.0)


def estimate_pose_bbox(pose_landmarks, frame_width: int, frame_height: int, margin: float = 0.12) -> dict | None:
    if not pose_landmarks:
        return None

    xs = []
    ys = []
    for landmark in pose_landmarks:
        x_value = float(getattr(landmark, "x", 0.0)) * frame_width
        y_value = float(getattr(landmark, "y", 0.0)) * frame_height
        visibility = float(getattr(landmark, "visibility", 1.0))
        if visibility < 0.08:
            continue
        xs.append(x_value)
        ys.append(y_value)

    if not xs or not ys:
        return None

    x0 = min(xs)
    y0 = min(ys)
    x1 = max(xs)
    y1 = max(ys)
    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)
    pad_x = width * margin
    pad_y = height * margin
    return clamp_bbox(build_bbox(int(x0 - pad_x), int(y0 - pad_y), int(x1 + pad_x), int(y1 + pad_y)), frame_width, frame_height)


def crop_region(frame, bbox: dict, region_name: str):
    region_name = normalize_region_name(region_name)
    start_ratio, end_ratio = REGION_WINDOWS[region_name]
    y0 = bbox["y0"] + int(round(bbox["height"] * start_ratio))
    y1 = bbox["y0"] + int(round(bbox["height"] * end_ratio))
    y0 = max(bbox["y0"], min(y0, bbox["y1"] - 1))
    y1 = max(y0 + 1, min(y1, bbox["y1"]))
    return frame[y0:y1, bbox["x0"]:bbox["x1"]]


def _color_mask(hsv_image, ranges) -> np.ndarray:
    combined = None
    for lower, upper in ranges:
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)
        current = cv2.inRange(hsv_image, lower_bound, upper_bound)
        combined = current if combined is None else cv2.bitwise_or(combined, current)
    return combined if combined is not None else np.zeros(hsv_image.shape[:2], dtype=np.uint8)


def compute_color_scores(crop) -> dict[str, float]:
    if crop is None or crop.size == 0:
        return {}

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total_pixels = max(hsv.shape[0] * hsv.shape[1], 1)
    scores = {}

    for color_name, ranges in COLOR_RULES.items():
        mask = _color_mask(hsv, ranges)
        score = float(cv2.countNonZero(mask)) / float(total_pixels)
        if score >= 0.01:
            scores[color_name] = round(score, 4)

    return scores


def dominant_color(scores: dict[str, float]) -> tuple[str | None, float]:
    if not scores:
        return None, 0.0
    color_name = max(scores, key=scores.get)
    return color_name, float(scores[color_name])


def extract_identity_features(frame, bbox: dict | None, identity_profiles: list | None = None) -> dict | None:
    if frame is None or bbox is None:
        return None

    frame_height, frame_width = frame.shape[:2]
    bbox = clamp_bbox(bbox, frame_width, frame_height)
    if bbox is None:
        return None

    requested_regions = {"top", "torso", "full"}
    for profile in identity_profiles or []:
        requested_regions.add(normalize_region_name(getattr(profile, "region", DEFAULT_REGION)))

    regions = {}
    profile_scores = {}
    best_profile_slot = None
    best_profile_score = 0.0

    for region_name in sorted(requested_regions):
        region_crop = crop_region(frame, bbox, region_name)
        scores = compute_color_scores(region_crop)
        color_name, score = dominant_color(scores)
        regions[region_name] = {
            "color": color_name,
            "score": round(float(score), 4),
            "scores": scores,
        }

    for profile in identity_profiles or []:
        region_name = normalize_region_name(getattr(profile, "region", DEFAULT_REGION))
        color_name = normalize_color_name(getattr(profile, "color_name", ""))
        score = float(regions.get(region_name, {}).get("scores", {}).get(color_name, 0.0))
        profile_scores[int(profile.slot_id)] = round(score, 4)
        if score > best_profile_score:
            best_profile_slot = int(profile.slot_id)
            best_profile_score = float(score)

    return {
        "bbox": bbox,
        "regions": regions,
        "profile_scores": profile_scores,
        "best_profile_slot": best_profile_slot,
        "best_profile_score": round(best_profile_score, 4),
    }


def merge_region_memories(appearance_items: list[dict | None]) -> dict:
    merged = defaultdict(lambda: {"color": None, "score": 0.0})
    for appearance in appearance_items:
        if not appearance:
            continue
        for region_name, payload in (appearance.get("regions") or {}).items():
            score = float(payload.get("score") or 0.0)
            if score >= float(merged[region_name].get("score") or 0.0):
                merged[region_name] = {
                    "color": payload.get("color"),
                    "score": round(score, 4),
                }
    return dict(merged)
