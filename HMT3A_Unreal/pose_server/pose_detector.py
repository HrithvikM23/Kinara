import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions,
    RunningMode
)

# Explicit imports to satisfy Pylance
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.hands as mp_hands

try:
    import config
except ImportError:
    class MockConfig:
        MAX_PERSONS = 2
        MIN_DETECTION_CONFIDENCE = 0.5
        MIN_POSE_PRESENCE_CONFIDENCE = 0.5
        MIN_TRACKING_CONFIDENCE = 0.5
        MIN_HAND_DETECTION_CONFIDENCE = 0.5
        MIN_HAND_PRESENCE_CONFIDENCE = 0.5
        MIN_HAND_TRACKING_CONFIDENCE = 0.5
    config = MockConfig()

POSE_MODEL_PATH = "pose_landmarker_full.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# Filter out head-related connections (indices 0-10)
POSE_CONNECTIONS = [conn for conn in mp_pose.POSE_CONNECTIONS if conn[0] > 10 and conn[1] > 10]
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS
PERSON_COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

def download_model(path, url):
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        urllib.request.urlretrieve(url, path)

class PoseDetector:
    def __init__(self):
        download_model(POSE_MODEL_PATH, POSE_MODEL_URL)
        download_model(HAND_MODEL_PATH, HAND_MODEL_URL)

        self.pose_landmarker = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
                running_mode=RunningMode.VIDEO,
                num_poses=config.MAX_PERSONS,
                min_pose_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                min_pose_presence_confidence=config.MIN_POSE_PRESENCE_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
            )
        )

        self.hand_landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
                running_mode=RunningMode.VIDEO,
                num_hands=config.MAX_PERSONS * 2,
                min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
                min_tracking_confidence=config.MIN_HAND_TRACKING_CONFIDENCE
            )
        )
        self.timestamp_ms = 0

    def _draw_landmarks(self, frame, landmarks, connections, color, is_pose=False):
        h, w, _ = frame.shape
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                p1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                p2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                cv2.line(frame, p1, p2, color, 2)
        
        start_range = 11 if is_pose else 0
        for i in range(start_range, len(landmarks)):
            lm = landmarks[i]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, color, -1)

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.timestamp_ms += 33
        
        pose_res = self.pose_landmarker.detect_for_video(mp_image, self.timestamp_ms)
        hand_res = self.hand_landmarker.detect_for_video(mp_image, self.timestamp_ms)

        combined_results = []
        
        # 1. Process Pose (Landmarks 11-32)
        if pose_res.pose_world_landmarks:
            for pose in pose_res.pose_world_landmarks:
                # Start with body landmarks (22 points)
                person_data = list(pose[11:]) 
                
                # 2. Append Hand Landmarks if they exist
                # This keeps the list structure consistent for the packet builder
                if hand_res.hand_world_landmarks:
                    for hand in hand_res.hand_world_landmarks:
                        person_data.extend(hand)
                
                combined_results.append(person_data)

        # Visual Feedback (Drawing)
        if pose_res.pose_landmarks:
            for i, lm in enumerate(pose_res.pose_landmarks):
                self._draw_landmarks(frame, lm, POSE_CONNECTIONS, PERSON_COLORS[i % len(PERSON_COLORS)], is_pose=True)
        
        if hand_res.hand_landmarks:
            for lm in hand_res.hand_landmarks:
                self._draw_landmarks(frame, lm, HAND_CONNECTIONS, (0, 200, 255))

        return combined_results, frame

    def close(self):
        """Cleanup method called by main.py"""
        self.pose_landmarker.close()
        self.hand_landmarker.close()