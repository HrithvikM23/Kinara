from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
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
    LEFT_HIP_INDEX,
    LEFT_SHOULDER_INDEX,
    LEFT_WRIST_INDEX,
    POSE_MODEL_PATH,
    POSE_MODEL_URL,
    RIGHT_HIP_INDEX,
    RIGHT_SHOULDER_INDEX,
    RIGHT_WRIST_INDEX,
)
from utils.math_utils import average, distance_2d
from utils.smoothing import LandmarkSmoother


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


def download_model(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {path.name}...")
        urlretrieve(url, str(path))


class PoseDetector:
    def __init__(self, config):
        self.config = config

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

        self.smoother = LandmarkSmoother(alpha=config.smoothing_alpha)

    def detect(self, frame, timestamp_ms: int):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        rendered = frame.copy()
        self._draw_results(rendered, pose_result, hand_result)

        people = self._combine_results(pose_result, hand_result)
        people = self.smoother.smooth_people(people)
        return people, rendered

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

    def _attach_hands(self, people, hand_result) -> None:
        hand_images = list(hand_result.hand_landmarks or [])
        hand_worlds = list(hand_result.hand_world_landmarks or [])
        handedness_sets = list(hand_result.handedness or [])

        for hand_index, hand_image in enumerate(hand_images):
            hand_world = hand_worlds[hand_index] if hand_index < len(hand_worlds) else None
            handedness_label, handedness_score = self._get_handedness(handedness_sets, hand_index)
            match = self._match_hand_to_person(people, hand_image, handedness_label)
            if match is None:
                continue

            person_index, side_key, score = match
            distance_key = "_left_distance" if side_key == "left_hand" else "_right_distance"
            if score >= people[person_index][distance_key]:
                continue

            people[person_index][side_key] = self._extract_hand(hand_world, hand_image)
            people[person_index][f"{side_key}_confidence"] = handedness_score
            people[person_index][distance_key] = score

    def _match_hand_to_person(self, people, hand_image, handedness_label):
        if not hand_image:
            return None

        hand_wrist = {"x": float(hand_image[0].x), "y": float(hand_image[0].y)}
        candidates = []

        preferred_side = None
        if handedness_label in {"left", "right"}:
            preferred_side = f"{handedness_label}_hand"

        for person_index, person in enumerate(people):
            if preferred_side is None:
                for side_key in ("left_hand", "right_hand"):
                    score = self._score_hand_for_person(person, hand_wrist, side_key)
                    candidates.append((score, person_index, side_key))
            else:
                score = self._score_hand_for_person(person, hand_wrist, preferred_side)
                candidates.append((score, person_index, preferred_side))

        if not candidates:
            return None

        score, person_index, side_key = min(candidates, key=lambda item: item[0])
        if score == float("inf"):
            return None
        return person_index, side_key, score

    def _score_hand_for_person(self, person, hand_wrist, side_key: str) -> float:
        reference_landmark = self._best_body_reference(person["_pose_image"], side_key)
        if reference_landmark is None:
            return float("inf")

        body_point = {"x": float(reference_landmark.x), "y": float(reference_landmark.y)}
        return distance_2d(hand_wrist, body_point)

    def _best_body_reference(self, pose_image, side_key: str):
        if side_key == "left_hand":
            candidates = (LEFT_WRIST_INDEX, LEFT_SHOULDER_INDEX, LEFT_HIP_INDEX)
        else:
            candidates = (RIGHT_WRIST_INDEX, RIGHT_SHOULDER_INDEX, RIGHT_HIP_INDEX)

        for index in candidates:
            if index >= len(pose_image):
                continue
            landmark = pose_image[index]
            if getattr(landmark, "visibility", 1.0) >= 0.25:
                return landmark

        for index in candidates:
            if index < len(pose_image):
                return pose_image[index]
        return None

    def _get_handedness(self, handedness_sets, hand_index: int):
        if hand_index >= len(handedness_sets) or not handedness_sets[hand_index]:
            return None, None

        category = handedness_sets[hand_index][0]
        label = (getattr(category, "category_name", "") or "").strip().lower()
        if label not in {"left", "right"}:
            label = None

        score = getattr(category, "score", None)
        return label, float(score) if score is not None else None

    def _pose_sort_key(self, pose_image) -> float:
        x_values = []
        for index in (
            LEFT_SHOULDER_INDEX,
            RIGHT_SHOULDER_INDEX,
            LEFT_HIP_INDEX,
            RIGHT_HIP_INDEX,
        ):
            if index < len(pose_image):
                x_values.append(float(pose_image[index].x))
        return average(x_values)

    def _empty_joint_map(self, joint_index_map):
        return {joint_name: None for joint_name in joint_index_map}

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
            height, width = frame.shape[:2]
            text_position = (int(wrist.x * width), int(wrist.y * height) - 10)
            cv2.putText(
                frame,
                (label or "hand").upper(),
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
                cv2.LINE_AA,
            )

    def _draw_landmarks(self, frame, landmarks, connections, color, start_index: int) -> None:
        height, width = frame.shape[:2]

        for start_index_raw, end_index_raw in connections:
            if start_index_raw >= len(landmarks) or end_index_raw >= len(landmarks):
                continue

            point_a = (
                int(landmarks[start_index_raw].x * width),
                int(landmarks[start_index_raw].y * height),
            )
            point_b = (
                int(landmarks[end_index_raw].x * width),
                int(landmarks[end_index_raw].y * height),
            )
            cv2.line(frame, point_a, point_b, color, 2)

        for landmark_index in range(start_index, len(landmarks)):
            landmark = landmarks[landmark_index]
            point = (int(landmark.x * width), int(landmark.y * height))
            cv2.circle(frame, point, 3, color, -1)

    def close(self) -> None:
        self.pose_landmarker.close()
        self.hand_landmarker.close()
