from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from network.osc_sender import OSCSender
from pipeline.pipeline import PoseHandPipeline
from utils.hand_constraints import enforce_hand_constraints
from utils.hand_fallback import generate_default_hand
from utils.smoothing import LandmarkSmoother

Point = tuple[int, int, float]
Box = tuple[int, int, int, int]

COLOR_HSV_RANGES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "black": ((0, 0, 0), (180, 255, 60)),
    "white": ((0, 0, 180), (180, 50, 255)),
    "gray": ((0, 0, 80), (180, 40, 190)),
    "silver": ((0, 0, 120), (180, 45, 235)),
    "red": ((0, 90, 60), (10, 255, 255)),
    "orange": ((10, 90, 60), (25, 255, 255)),
    "yellow": ((25, 90, 60), (35, 255, 255)),
    "green": ((35, 60, 50), (85, 255, 255)),
    "blue": ((90, 70, 50), (135, 255, 255)),
    "purple": ((135, 70, 50), (165, 255, 255)),
    "pink": ((165, 60, 60), (179, 255, 255)),
    "brown": ((5, 80, 30), (25, 255, 160)),
}


@dataclass(slots=True)
class PersonDetection:
    track_id: int | None
    box: Box
    body_points: list[Point]
    score: float
    color_scores: dict[str, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


@dataclass(slots=True)
class PersonTrack:
    id: int
    box: Box
    pipeline: PoseHandPipeline
    missed_frames: int = 0
    label: str | None = None
    body_points: list[Point] = field(default_factory=list)
    hands_by_side: dict[str, dict] = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max((ax2 - ax1) * (ay2 - ay1), 1)
    area_b = max((bx2 - bx1) * (by2 - by1), 1)
    return inter_area / float(area_a + area_b - inter_area)


def _center_distance_score(box_a: Box, box_b: Box) -> float:
    ax = (box_a[0] + box_a[2]) * 0.5
    ay = (box_a[1] + box_a[3]) * 0.5
    bx = (box_b[0] + box_b[2]) * 0.5
    by = (box_b[1] + box_b[3]) * 0.5
    distance = math.hypot(ax - bx, ay - by)
    scale = max(box_a[2] - box_a[0], box_a[3] - box_a[1], box_b[2] - box_b[0], box_b[3] - box_b[1], 1)
    return max(0.0, 1.0 - distance / (scale * 2.0))


def _non_max_suppress(boxes_with_weights: list[tuple[Box, float]], iou_threshold: float = 0.35) -> list[Box]:
    ordered = sorted(boxes_with_weights, key=lambda item: item[1], reverse=True)
    selected: list[Box] = []
    for box, _ in ordered:
        if any(_iou(box, kept_box) > iou_threshold for kept_box in selected):
            continue
        selected.append(box)
    return selected


def _expand_box(box: Box, frame_width: int, frame_height: int, scale: float) -> Box:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    expanded_w = width * scale
    expanded_h = height * scale
    nx1 = max(0, int(round(cx - expanded_w * 0.5)))
    ny1 = max(0, int(round(cy - expanded_h * 0.5)))
    nx2 = min(frame_width, int(round(cx + expanded_w * 0.5)))
    ny2 = min(frame_height, int(round(cy + expanded_h * 0.5)))
    return nx1, ny1, nx2, ny2


def _torso_crop(frame, box: Box):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    tx1 = x1 + int(width * 0.2)
    tx2 = x2 - int(width * 0.2)
    ty1 = y1 + int(height * 0.12)
    ty2 = y1 + int(height * 0.60)
    return frame[max(0, ty1):max(0, ty2), max(0, tx1):max(0, tx2)]


def _color_scores(frame, box: Box) -> dict[str, float]:
    crop = _torso_crop(frame, box)
    if crop.size == 0:
        return {}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixel_count = max(hsv.shape[0] * hsv.shape[1], 1)
    scores: dict[str, float] = {}
    for color_name, (lower, upper) in COLOR_HSV_RANGES.items():
        lower_np = np.array([int(v) for v in lower], dtype=np.uint8)
        upper_np = np.array([int(v) for v in upper], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_np, upper_np)
        scores[color_name] = float(cv2.countNonZero(mask)) / float(pixel_count)
    return scores


def _translate_body_points(points: list[Point], offset_x: int, offset_y: int) -> list[Point]:
    return [(x + offset_x, y + offset_y, conf) for x, y, conf in points]


def _translate_hands(hands_by_side: dict[str, dict], offset_x: int, offset_y: int) -> dict[str, dict]:
    translated: dict[str, dict] = {}
    for side, payload in hands_by_side.items():
        x1, y1, x2, y2 = payload["box"]
        translated[side] = {
            "box": (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
            "points": [(x + offset_x, y + offset_y, conf) for x, y, conf in payload["points"]],
        }
    return translated


class MultiPersonTracker:
    def __init__(self, config, runner):
        self.config = config
        self.runner = runner
        self._tracks: list[PersonTrack] = []
        self._next_id = 1

    def update(self, frame) -> list[PersonTrack]:
        detections = self._detect_people(frame)
        self._associate_tracks(frame, detections)
        self._resolve_cross_person_hands()
        return [track for track in self._tracks if track.missed_frames == 0 and track.body_points]

    def _detect_people(self, frame) -> list[PersonDetection]:
        raw_detections = self.runner.detect_bodies(frame, max_people=self.config.max_people, track=True)
        frame_height, frame_width = frame.shape[:2]
        detections: list[PersonDetection] = []
        for detection in raw_detections:
            expanded_box = _expand_box(detection["box"], frame_width, frame_height, self.config.person_box_scale)
            detections.append(
                PersonDetection(
                    track_id=detection["id"],
                    box=expanded_box,
                    body_points=detection["body_points"],
                    score=float(detection["score"]),
                    color_scores=_color_scores(frame, expanded_box),
                )
            )
        return detections

    def _associate_tracks(self, frame, detections: list[PersonDetection]) -> None:
        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()
        used_detections: set[int] = set()

        scored_pairs: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            for track_index, track in enumerate(self._tracks):
                score = self._match_score(track, detection)
                if score > self.config.person_match_threshold:
                    scored_pairs.append((score, track_index, detection_index))

        for _, track_index, detection_index in sorted(scored_pairs, reverse=True):
            if track_index in used_tracks or detection_index in used_detections:
                continue
            assignments[track_index] = detection_index
            used_tracks.add(track_index)
            used_detections.add(detection_index)

        updated_tracks: list[PersonTrack] = []
        for track_index, track in enumerate(self._tracks):
            detection_index = assignments.get(track_index)
            if detection_index is None:
                track.missed_frames += 1
                if track.missed_frames <= self.config.person_track_hold_frames:
                    updated_tracks.append(track)
                continue

            detection = detections[detection_index]
            self._update_track_from_detection(track, detection, frame)
            updated_tracks.append(track)

        assigned_labels = {track.label for track in updated_tracks if track.label}
        for detection_index, detection in enumerate(detections):
            if detection_index in used_detections:
                continue
            track = self._create_track()
            track.label = self._best_identity_label(detection, assigned_labels)
            self._update_track_from_detection(track, detection, frame)
            updated_tracks.append(track)
            if track.label:
                assigned_labels.add(track.label)

        self._tracks = updated_tracks[: self.config.max_people]

    def _create_track(self) -> PersonTrack:
        track = PersonTrack(
            id=self._next_id,
            box=(0, 0, 0, 0),
            pipeline=PoseHandPipeline(
                self.config,
                self.runner,
                LandmarkSmoother(self.config),
                OSCSender(enabled=False),
            ),
        )
        self._next_id += 1
        return track

    def _match_score(self, track: PersonTrack, detection: PersonDetection) -> float:
        if detection.track_id is not None and detection.track_id == track.id:
            return 10.0
        score = _iou(track.box, detection.box) * 0.65 + _center_distance_score(track.box, detection.box) * 0.35
        if track.label:
            score += self._identity_score(track.label, detection.color_scores) * 0.5
        return score

    def _best_identity_label(self, detection: PersonDetection, assigned_labels: set[str]) -> str | None:
        best_label: str | None = None
        best_score = 0.0
        for label in self.config.identity_hints:
            if label in assigned_labels:
                continue
            score = self._identity_score(label, detection.color_scores)
            if score > best_score:
                best_score = score
                best_label = label
        if best_score < self.config.identity_min_score:
            return None
        return best_label

    def _identity_score(self, label: str, color_scores: dict[str, float]) -> float:
        colors = self.config.identity_hints.get(label, ())
        if not colors:
            return 0.0
        return sum(color_scores.get(color, 0.0) for color in colors) / float(len(colors))

    def _update_track_from_detection(self, track: PersonTrack, detection: PersonDetection, frame) -> None:
        body_points = detection.body_points
        hands_by_side = track.pipeline.detect_hands(frame, body_points)
        track.id = detection.track_id if detection.track_id is not None else track.id
        track.box = detection.box
        track.missed_frames = 0
        track.body_points = body_points
        track.hands_by_side = hands_by_side

    def _resolve_cross_person_hands(self) -> None:
        side_indices = {
            "left": (9, 7),
            "right": (10, 8),
        }
        active_tracks = [track for track in self._tracks if track.missed_frames == 0 and len(track.body_points) >= 11]
        for track in active_tracks:
            for side, hand_payload in list(track.hands_by_side.items()):
                wrist_index, elbow_index = side_indices[side]
                owner_wrist = track.body_points[wrist_index]
                owner_elbow = track.body_points[elbow_index]
                if owner_wrist[2] <= self.config.body_conf_threshold or owner_elbow[2] <= self.config.body_conf_threshold:
                    continue

                hand_wrist = hand_payload["points"][0]
                owner_distance = math.hypot(hand_wrist[0] - owner_wrist[0], hand_wrist[1] - owner_wrist[1])
                closest_track = track
                closest_distance = owner_distance

                for other_track in active_tracks:
                    if other_track.id == track.id or len(other_track.body_points) <= wrist_index:
                        continue
                    other_wrist = other_track.body_points[wrist_index]
                    if other_wrist[2] <= self.config.body_conf_threshold:
                        continue
                    other_distance = math.hypot(hand_wrist[0] - other_wrist[0], hand_wrist[1] - other_wrist[1])
                    if other_distance < closest_distance:
                        closest_distance = other_distance
                        closest_track = other_track

                if closest_track.id == track.id:
                    continue

                if closest_distance > owner_distance * self.config.person_cross_wrist_ratio:
                    continue

                fallback_hand = generate_default_hand(owner_wrist, owner_elbow, side, self.config)
                track.hands_by_side[side] = {
                    "box": hand_payload["box"],
                    "points": enforce_hand_constraints(fallback_hand),
                }
