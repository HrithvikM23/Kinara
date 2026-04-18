from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.pose as mp_pose

from config import (
    BODY_INDEX_BY_NAME,
    HAND_INDEX_BY_NAME,
    HAND_MODEL_PATH,
    HAND_MODEL_URL,
    LEFT_ELBOW_INDEX,
    LEFT_HIP_INDEX,
    LEFT_SHOULDER_INDEX,
    LEFT_WRIST_INDEX,
    POSE_MODEL_PATH,
    POSE_MODEL_URL,
    RIGHT_ELBOW_INDEX,
    RIGHT_HIP_INDEX,
    RIGHT_SHOULDER_INDEX,
    RIGHT_WRIST_INDEX,
)
from utils.math_utils import average, distance_2d


POSE_CONNECTIONS = [
    connection
    for connection in mp_pose.POSE_CONNECTIONS
    if connection[0] > 10 and connection[1] > 10
]
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS
PERSON_COLORS = [
    (0, 255, 0),
    (255, 80, 80),
    (80, 160, 255),
    (255, 220, 0),
    (200, 120, 255),
]
HAND_COLORS = {
    "left": (255, 140, 40),
    "right": (0, 210, 255),
    "unknown": (180, 180, 180),
}
VISIBILITY_THRESHOLD = 0.2
SHOULDER_VISIBILITY_MIN = 0.5
WRIST_VISIBILITY_MIN = 0.3
MEMORY_FLIP_FRAMES = 2   # consecutive frames needed to commit a side flip
MEMORY_MAX_FRAMES = 60   # cap on stability accumulation


def download_model(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {path.name}...")
        urlretrieve(url, str(path))


# ---------------------------------------------------------------------------
# Hand-side memory entry — stored per hand slot index
# ---------------------------------------------------------------------------
class _SideMemory:
    __slots__ = ("side", "frames", "pending", "pending_count")

    def __init__(self, side: str) -> None:
        self.side: str = side
        self.frames: int = 1
        self.pending: str | None = None
        self.pending_count: int = 0

    def reinforce(self) -> None:
        self.frames = min(self.frames + 1, MEMORY_MAX_FRAMES)
        self.pending = None
        self.pending_count = 0

    def propose_flip(self, new_side: str) -> bool:
        """Returns True if the flip should be committed."""
        if self.pending == new_side:
            self.pending_count += 1
        else:
            self.pending = new_side
            self.pending_count = 1
        if self.pending_count >= MEMORY_FLIP_FRAMES:
            self.side = new_side
            self.frames = 1
            self.pending = None
            self.pending_count = 0
            return True
        return False


class PoseDetector:
    def __init__(self, config):
        self.config = config
        self._hand_side_memory: dict[int, _SideMemory] = {}

        download_model(POSE_MODEL_PATH, POSE_MODEL_URL)
        download_model(HAND_MODEL_PATH, HAND_MODEL_URL)

        self.pose_landmarker = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
                running_mode=RunningMode.VIDEO,
                num_poses=config.max_persons,
                min_pose_detection_confidence=config.min_pose_detection_confidence,
                min_pose_presence_confidence=config.min_pose_presence_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
            )
        )

        self.hand_landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
                running_mode=RunningMode.VIDEO,
                num_hands=config.max_persons * 2,
                min_hand_detection_confidence=config.min_hand_detection_confidence,
                min_hand_presence_confidence=config.min_hand_presence_confidence,
                min_tracking_confidence=config.min_hand_tracking_confidence,
            )
        )

        self.hand_landmarker_roi = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
                running_mode=RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=config.min_hand_detection_confidence,
                min_hand_presence_confidence=config.min_hand_presence_confidence,
                min_tracking_confidence=config.min_hand_tracking_confidence,
            )
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def detect(self, frame, timestamp_ms: int):
        rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = self._detect_hands(rgb_frame, mp_image, pose_result, timestamp_ms)

        rendered = frame.copy() if self.config.render_output else frame
        if self.config.render_output:
            self._draw_results(rendered, pose_result, hand_result)

        people = self._combine_results(pose_result, hand_result)
        return people, rendered

    # ------------------------------------------------------------------
    # Hand detection routing
    # ------------------------------------------------------------------

    def _detect_hands(self, rgb_frame, full_mp_image, pose_result, timestamp_ms: int):
        if not self.config.enable_hand_roi:
            return self.hand_landmarker.detect_for_video(full_mp_image, timestamp_ms)

        frame_height, frame_width = rgb_frame.shape[:2]
        roi_records = self._build_hand_rois(pose_result, frame_width, frame_height)
        roi_results = []

        for roi_record in roi_records:
            x0, y0, x1, y1 = roi_record["bounds"]
            crop = np.ascontiguousarray(rgb_frame[y0:y1, x0:x1])
            if crop.size == 0:
                continue

            roi_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
            roi_result = self.hand_landmarker_roi.detect(roi_mp_image)
            selected = self._select_roi_detection(roi_result, roi_record["side"])
            if selected is None:
                continue

            roi_results.append(
                self._map_roi_detection(
                    selected, x0, y0, x1 - x0, y1 - y0, frame_width, frame_height
                )
            )

        expected_count = len(roi_records)
        if not self.config.hand_roi_fallback_to_full_frame:
            return self._build_hand_result(roi_results)

        if expected_count == 0 or len(roi_results) < expected_count:
            full_result = self.hand_landmarker.detect_for_video(full_mp_image, timestamp_ms)
            return self._merge_hand_results(self._build_hand_result(roi_results), full_result)

        return self._build_hand_result(roi_results)

    # ------------------------------------------------------------------
    # ROI helpers
    # ------------------------------------------------------------------

    def _build_hand_rois(self, pose_result, frame_width: int, frame_height: int):
        rois = []
        for pose_image in list(pose_result.pose_landmarks or [])[: self.config.max_persons]:
            for side in ("left", "right"):
                roi = self._build_hand_roi_for_side(pose_image, side, frame_width, frame_height)
                if roi is not None:
                    rois.append(roi)
        return rois

    def _build_hand_roi_for_side(self, pose_image, side: str, frame_width: int, frame_height: int):
        wrist_index = LEFT_WRIST_INDEX if side == "left" else RIGHT_WRIST_INDEX
        elbow_index = LEFT_ELBOW_INDEX if side == "left" else RIGHT_ELBOW_INDEX
        shoulder_index = LEFT_SHOULDER_INDEX if side == "left" else RIGHT_SHOULDER_INDEX

        wrist = self._safe_pose_landmark(pose_image, wrist_index)
        if wrist is None:
            return None
        if float(getattr(wrist, "visibility", 1.0)) < VISIBILITY_THRESHOLD:
            return None

        elbow = self._safe_pose_landmark(pose_image, elbow_index)
        shoulder = self._safe_pose_landmark(pose_image, shoulder_index)

        wrist_px = self._landmark_to_pixel(wrist, frame_width, frame_height)
        elbow_px = self._landmark_to_pixel(elbow, frame_width, frame_height) if elbow else None
        shoulder_px = self._landmark_to_pixel(shoulder, frame_width, frame_height) if shoulder else None

        roi_size = float(self.config.hand_roi_min_size)
        if elbow_px is not None:
            roi_size = max(roi_size, distance_2d(wrist_px, elbow_px) * self.config.hand_roi_scale)
        if shoulder_px is not None:
            roi_size = max(
                roi_size,
                distance_2d(wrist_px, shoulder_px) * (self.config.hand_roi_scale * 0.7),
            )

        center_x = wrist_px["x"]
        center_y = wrist_px["y"]
        if elbow_px is not None:
            center_x += (wrist_px["x"] - elbow_px["x"]) * 0.35
            center_y += (wrist_px["y"] - elbow_px["y"]) * 0.35

        half_size = int(max(roi_size / 2.0, 1.0))
        center_x = int(round(center_x))
        center_y = int(round(center_y))

        x0 = max(0, center_x - half_size)
        y0 = max(0, center_y - half_size)
        x1 = min(frame_width, center_x + half_size)
        y1 = min(frame_height, center_y + half_size)

        if x1 - x0 < 8 or y1 - y0 < 8:
            return None

        return {"side": side, "bounds": (x0, y0, x1, y1)}

    # ------------------------------------------------------------------
    # Hand-side resolution  (geometry → mp label → memory)
    # ------------------------------------------------------------------

    def _geometry_side(self, hand_image, people) -> str | None:
        """
        Determine hand side purely from landmark geometry.

        Strategy (in priority order):
        1. Compare hand wrist x against both pose wrist x positions — closest wins.
           Both pose wrists must have acceptable visibility.
        2. If only one pose wrist is visible, check which half of the torso the
           hand falls in relative to the shoulder midpoint.
        3. Return None if shoulders aren't visible enough to be trusted.

        All coordinates are in normalised image space [0,1] so no mirroring is
        needed — MediaPipe pose and hand landmarks share the same coordinate frame.
        """
        if not hand_image or not people:
            return None

        hand_x = float(hand_image[0].x)

        # pick the person whose shoulder centre is horizontally closest to the hand
        best_person = None
        best_dist = float("inf")
        for person in people:
            pose_image = person.get("_pose_image")
            if pose_image is None:
                continue
            ls = self._safe_pose_landmark(pose_image, LEFT_SHOULDER_INDEX)
            rs = self._safe_pose_landmark(pose_image, RIGHT_SHOULDER_INDEX)
            if ls is None or rs is None:
                continue
            ls_vis = float(getattr(ls, "visibility", 0.0))
            rs_vis = float(getattr(rs, "visibility", 0.0))
            if ls_vis < SHOULDER_VISIBILITY_MIN or rs_vis < SHOULDER_VISIBILITY_MIN:
                continue
            mid_x = (float(ls.x) + float(rs.x)) / 2.0
            dist = abs(mid_x - hand_x)
            if dist < best_dist:
                best_dist = dist
                best_person = person

        if best_person is None:
            return None

        pose_image = best_person["_pose_image"]
        ls = self._safe_pose_landmark(pose_image, LEFT_SHOULDER_INDEX)
        rs = self._safe_pose_landmark(pose_image, RIGHT_SHOULDER_INDEX)
        lw = self._safe_pose_landmark(pose_image, LEFT_WRIST_INDEX)
        rw = self._safe_pose_landmark(pose_image, RIGHT_WRIST_INDEX)

        lw_vis = float(getattr(lw, "visibility", 0.0)) if lw else 0.0
        rw_vis = float(getattr(rw, "visibility", 0.0)) if rw else 0.0

        # --- strategy 1: both pose wrists visible → nearest pose wrist wins ----
        if lw_vis >= WRIST_VISIBILITY_MIN and rw_vis >= WRIST_VISIBILITY_MIN:
            dist_left = abs(hand_x - float(lw.x))   # type: ignore[union-attr]
            dist_right = abs(hand_x - float(rw.x))  # type: ignore[union-attr]
            return "left" if dist_left < dist_right else "right"

        # --- strategy 2: one pose wrist visible → use it as anchor -------------
        if lw_vis >= WRIST_VISIBILITY_MIN and lw is not None:
            # we know where left wrist is; if hand is close to it → left
            mid_x = (float(ls.x) + float(rs.x)) / 2.0  # type: ignore[union-attr]
            dist_to_left_wrist = abs(hand_x - float(lw.x))
            dist_to_mid = abs(hand_x - mid_x)
            return "left" if dist_to_left_wrist < dist_to_mid else "right"

        if rw_vis >= WRIST_VISIBILITY_MIN and rw is not None:
            mid_x = (float(ls.x) + float(rs.x)) / 2.0  # type: ignore[union-attr]
            dist_to_right_wrist = abs(hand_x - float(rw.x))
            dist_to_mid = abs(hand_x - mid_x)
            return "right" if dist_to_right_wrist < dist_to_mid else "left"

        # --- strategy 3: no reliable wrist → shoulder midpoint split -----------
        # In MediaPipe normalised coords (subject facing camera):
        #   subject's LEFT hand  → larger x  (right side of frame)
        #   subject's RIGHT hand → smaller x (left side of frame)
        mid_x = (float(ls.x) + float(rs.x)) / 2.0  # type: ignore[union-attr]
        return "left" if hand_x > mid_x else "right"

    def _resolve_side(
        self,
        hand_slot: int,
        mp_label: str | None,
        mp_score: float | None,
        geo_side: str | None,
    ) -> str:
        """
        Final side decision for one detected hand.

        Priority:
          1. Geometry (when available) — override everything, update memory.
          2. Weighted vote: mp_label (scaled by mp_score) + memory stability bonus.
          3. Flip protection: require MEMORY_FLIP_FRAMES consecutive frames of the
             new side before committing, to avoid single-frame hallucinations.
        """
        mem = self._hand_side_memory.get(hand_slot)

        # ---- geometry is available: trust it directly ----------------------
        if geo_side is not None:
            if mem is None:
                self._hand_side_memory[hand_slot] = _SideMemory(geo_side)
            elif mem.side == geo_side:
                mem.reinforce()
            else:
                mem.propose_flip(geo_side)
            # always return geometry result immediately
            return geo_side

        # ---- geometry unavailable: vote ------------------------------------
        votes: dict[str, float] = {"left": 0.0, "right": 0.0}

        if mp_label in ("left", "right"):
            votes[mp_label] += float(mp_score or 0.5)

        if mem is not None:
            stability = min(mem.frames / 10.0, 1.5)
            votes[mem.side] += stability

        resolved = "left" if votes["left"] >= votes["right"] else "right"

        if mem is None:
            self._hand_side_memory[hand_slot] = _SideMemory(resolved)
            return resolved

        if mem.side == resolved:
            mem.reinforce()
            return resolved

        # proposed flip — only commit after enough consecutive frames
        mem.propose_flip(resolved)
        return mem.side  # return stable side until flip commits

    def _deduplicate_sides(self, candidates: list[dict]) -> list[dict]:
        """
        If two hands end up claiming the same side, keep the higher-confidence
        one and flip the other to the opposite side, clearing its memory.
        """
        side_map: dict[str, list[dict]] = defaultdict(list)
        for c in candidates:
            side_map[c["side"]].append(c)

        resolved: list[dict] = []
        for side, group in side_map.items():
            if len(group) == 1:
                resolved.append(group[0])
                continue
            group.sort(key=lambda c: c["mp_score"], reverse=True)
            resolved.append(group[0])
            opposite = "right" if side == "left" else "left"
            for loser in group[1:]:
                loser["side"] = opposite
                self._hand_side_memory.pop(loser["hand_index"], None)
                resolved.append(loser)

        return resolved

    # ------------------------------------------------------------------
    # Attaching hands to people
    # ------------------------------------------------------------------

    def _attach_hands(self, people, hand_result) -> None:
        hand_images = list(hand_result.hand_landmarks or [])
        hand_worlds = list(hand_result.hand_world_landmarks or [])
        handedness_sets = list(hand_result.handedness or [])

        candidates: list[dict] = []
        for hand_index, hand_image in enumerate(hand_images):
            hand_world = hand_worlds[hand_index] if hand_index < len(hand_worlds) else None
            mp_label, mp_score = self._get_handedness(handedness_sets, hand_index)
            geo_side = self._geometry_side(hand_image, people)
            side = self._resolve_side(hand_index, mp_label, mp_score, geo_side)
            candidates.append(
                {
                    "hand_index": hand_index,
                    "hand_image": hand_image,
                    "hand_world": hand_world,
                    "side": side,
                    "mp_score": mp_score or 0.0,
                }
            )

        candidates = self._deduplicate_sides(candidates)

        for candidate in candidates:
            hand_image = candidate["hand_image"]
            hand_world = candidate["hand_world"]
            side_key = f"{candidate['side']}_hand"
            distance_key = "_left_distance" if candidate["side"] == "left" else "_right_distance"

            match = self._match_hand_to_person(people, hand_image, candidate["side"])
            if match is None:
                continue

            person_index, _, score = match
            if score >= people[person_index][distance_key]:
                continue

            people[person_index][side_key] = self._extract_hand(hand_world, hand_image)
            people[person_index][f"{side_key}_confidence"] = candidate["mp_score"]
            people[person_index][distance_key] = score

    # ------------------------------------------------------------------
    # Person / result assembly
    # ------------------------------------------------------------------

    def _combine_results(self, pose_result, hand_result):
        pose_images = list(pose_result.pose_landmarks or [])
        pose_worlds = list(pose_result.pose_world_landmarks or [])

        people = []
        for pose_index, pose_image in enumerate(pose_images[: self.config.max_persons]):
            pose_world = pose_worlds[pose_index] if pose_index < len(pose_worlds) else None
            people.append(
                {
                    "id": pose_index,
                    "body": self._extract_body(pose_world, pose_image),
                    "left_hand": self._empty_joint_map(HAND_INDEX_BY_NAME),
                    "right_hand": self._empty_joint_map(HAND_INDEX_BY_NAME),
                    "left_hand_confidence": None,
                    "right_hand_confidence": None,
                    "_pose_image": pose_image,
                    "_sort_x": self._pose_sort_key(pose_image),
                    "_left_distance": float("inf"),
                    "_right_distance": float("inf"),
                }
            )

        people.sort(key=lambda person: person["_sort_x"])
        for person_index, person in enumerate(people):
            person["id"] = person_index

        if people:
            self._attach_hands(people, hand_result)

        return [
            {
                "id": person["id"],
                "body": person["body"],
                "left_hand": person["left_hand"],
                "right_hand": person["right_hand"],
                "left_hand_confidence": person["left_hand_confidence"],
                "right_hand_confidence": person["right_hand_confidence"],
            }
            for person in people
        ]

    # ------------------------------------------------------------------
    # Landmark extraction
    # ------------------------------------------------------------------

    def _extract_body(self, pose_world, pose_image):
        body = {}
        for name, index in BODY_INDEX_BY_NAME.items():
            image_landmark = pose_image[index] if pose_image and index < len(pose_image) else None
            world_landmark = pose_world[index] if pose_world and index < len(pose_world) else None
            source_landmark = world_landmark if world_landmark is not None else image_landmark

            if source_landmark is None:
                body[name] = None
                continue

            body[name] = {
                "x": round(float(source_landmark.x), 6),
                "y": round(float(source_landmark.y), 6),
                "z": round(float(source_landmark.z), 6),
                "visibility": round(float(getattr(image_landmark, "visibility", 1.0)), 6),
            }
        return body

    def _extract_hand(self, hand_world, hand_image):
        hand = {}
        for name, index in HAND_INDEX_BY_NAME.items():
            image_landmark = hand_image[index] if hand_image and index < len(hand_image) else None
            world_landmark = hand_world[index] if hand_world and index < len(hand_world) else None
            source_landmark = world_landmark if world_landmark is not None else image_landmark

            if source_landmark is None:
                hand[name] = None
                continue

            hand[name] = {
                "x": round(float(source_landmark.x), 6),
                "y": round(float(source_landmark.y), 6),
                "z": round(float(source_landmark.z), 6),
            }
        return hand

    # ------------------------------------------------------------------
    # Hand matching
    # ------------------------------------------------------------------

    def _match_hand_to_person(self, people, hand_image, side: str):
        if not hand_image:
            return None

        hand_wrist = {"x": float(hand_image[0].x), "y": float(hand_image[0].y)}
        side_key = f"{side}_hand"
        best: tuple | None = None

        for person_index, person in enumerate(people):
            score = self._score_hand_for_person(person, hand_wrist, side_key)
            if best is None or score < best[0]:
                best = (score, person_index, side_key)

        if best is None or best[0] == float("inf"):
            return None
        return best[1], best[2], best[0]

    def _score_hand_for_person(self, person, hand_wrist, side_key: str) -> float:
        ref = self._best_body_reference(person["_pose_image"], side_key)
        if ref is None:
            return float("inf")
        return distance_2d(hand_wrist, {"x": float(ref.x), "y": float(ref.y)})

    def _best_body_reference(self, pose_image, side_key: str):
        candidates = (
            (LEFT_WRIST_INDEX, LEFT_SHOULDER_INDEX, LEFT_HIP_INDEX)
            if side_key == "left_hand"
            else (RIGHT_WRIST_INDEX, RIGHT_SHOULDER_INDEX, RIGHT_HIP_INDEX)
        )
        for index in candidates:
            if index >= len(pose_image):
                continue
            lm = pose_image[index]
            if getattr(lm, "visibility", 1.0) >= 0.25:
                return lm
        for index in candidates:
            if index < len(pose_image):
                return pose_image[index]
        return None

    # ------------------------------------------------------------------
    # ROI result helpers
    # ------------------------------------------------------------------

    def _select_roi_detection(self, roi_result, expected_side: str):
        hand_images = list(roi_result.hand_landmarks or [])
        handedness_sets = list(roi_result.handedness or [])
        if not hand_images:
            return None

        matches = []
        for hand_index, hand_image in enumerate(hand_images):
            label, score = self._get_handedness(handedness_sets, hand_index)
            quality = float(score) if score is not None else 0.0
            if label == expected_side:
                quality += 1.0
            hs = handedness_sets[hand_index] if hand_index < len(handedness_sets) else []
            matches.append((quality, hand_index, hand_image, hs))

        _, _, hand_image, handedness = max(matches, key=lambda item: item[0])
        return {"hand_image": hand_image, "handedness": handedness}

    def _map_roi_detection(
        self,
        selected,
        x0: int,
        y0: int,
        roi_width: int,
        roi_height: int,
        frame_width: int,
        frame_height: int,
    ):
        mapped_hand = [
            SimpleNamespace(
                x=(x0 + float(lm.x) * roi_width) / frame_width,
                y=(y0 + float(lm.y) * roi_height) / frame_height,
                z=float(lm.z),
            )
            for lm in selected["hand_image"]
        ]
        return {
            "hand_landmarks": mapped_hand,
            "hand_world_landmarks": None,
            "handedness": list(selected["handedness"]),
        }

    def _build_hand_result(self, hand_entries):
        return SimpleNamespace(
            hand_landmarks=[e["hand_landmarks"] for e in hand_entries],
            hand_world_landmarks=[e["hand_world_landmarks"] for e in hand_entries],
            handedness=[e["handedness"] for e in hand_entries],
        )

    def _merge_hand_results(self, roi_result, full_result):
        merged: list[dict] = []
        for result in (roi_result, full_result):
            hand_images = list(result.hand_landmarks or [])
            hand_worlds = list(result.hand_world_landmarks or [])
            handedness_sets = list(result.handedness or [])
            for i, hand_landmarks in enumerate(hand_images):
                merged.append(
                    {
                        "hand_landmarks": hand_landmarks,
                        "hand_world_landmarks": hand_worlds[i] if i < len(hand_worlds) else None,
                        "handedness": handedness_sets[i] if i < len(handedness_sets) else [],
                    }
                )
        return self._build_hand_result(merged)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _get_handedness(self, handedness_sets, hand_index: int):
        if hand_index >= len(handedness_sets) or not handedness_sets[hand_index]:
            return None, None
        category = handedness_sets[hand_index][0]
        label = (getattr(category, "category_name", "") or "").strip().lower()
        if label not in {"left", "right"}:
            label = None
        score = getattr(category, "score", None)
        return label, float(score) if score is not None else None

    def _safe_pose_landmark(self, pose_image, index: int):
        if pose_image is None or index >= len(pose_image):
            return None
        return pose_image[index]

    def _landmark_to_pixel(self, landmark, frame_width: int, frame_height: int):
        return {
            "x": float(landmark.x) * frame_width,
            "y": float(landmark.y) * frame_height,
        }

    def _pose_sort_key(self, pose_image) -> float:
        x_values = [
            float(pose_image[i].x)
            for i in (LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX, LEFT_HIP_INDEX, RIGHT_HIP_INDEX)
            if i < len(pose_image)
        ]
        return average(x_values)

    def _empty_joint_map(self, joint_index_map):
        return {name: None for name in joint_index_map}

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_results(self, frame, pose_result, hand_result) -> None:
        for index, pose_landmarks in enumerate(list(pose_result.pose_landmarks or [])):
            color = PERSON_COLORS[index % len(PERSON_COLORS)]
            self._draw_landmarks(frame, pose_landmarks, POSE_CONNECTIONS, color, start_index=11)

        hand_images = list(hand_result.hand_landmarks or [])
        handedness_sets = list(hand_result.handedness or [])

        for hand_index, hand_landmarks in enumerate(hand_images):
            label, _ = self._get_handedness(handedness_sets, hand_index)
            color = HAND_COLORS.get(label or "unknown", HAND_COLORS["unknown"])
            self._draw_landmarks(frame, hand_landmarks, HAND_CONNECTIONS, color, start_index=0)

            wrist = hand_landmarks[0]
            h, w = frame.shape[:2]
            cv2.putText(
                frame,
                (label or "hand").upper(),
                (int(wrist.x * w), int(wrist.y * h) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
                cv2.LINE_AA,
            )

    def _draw_landmarks(self, frame, landmarks, connections, color, start_index: int) -> None:
        h, w = frame.shape[:2]

        for s, e in connections:
            if s >= len(landmarks) or e >= len(landmarks):
                continue
            cv2.line(
                frame,
                (int(landmarks[s].x * w), int(landmarks[s].y * h)),
                (int(landmarks[e].x * w), int(landmarks[e].y * h)),
                color,
                2,
            )

        for i in range(start_index, len(landmarks)):
            lm = landmarks[i]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, color, -1)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.pose_landmarker.close()
        self.hand_landmarker.close()
        self.hand_landmarker_roi.close()
        self._hand_side_memory.clear()