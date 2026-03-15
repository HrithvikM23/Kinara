import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import urllib.request
import os
import config


MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31),
    (30, 32), (27, 31), (28, 32)
]

PERSON_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
]


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe pose model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


def draw_landmarks(frame, landmarks, color):
    h, w = frame.shape[:2]

    for lm in landmarks:
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, color, -1)

    for a, b in POSE_CONNECTIONS:
        ax = int(landmarks[a].x * w)
        ay = int(landmarks[a].y * h)
        bx = int(landmarks[b].x * w)
        by = int(landmarks[b].y * h)
        cv2.line(frame, (ax, ay), (bx, by), color, 2)

    return frame


class PoseDetector:

    def __init__(self):
        download_model()

        options = PoseLandmarkerOptions(
            base_options                  = python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode                  = RunningMode.VIDEO,
            num_poses                     = config.MAX_PERSONS,
            min_pose_detection_confidence = config.MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence  = config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence       = config.MIN_TRACKING_CONFIDENCE,
            output_segmentation_masks     = False,
        )

        self.landmarker  = PoseLandmarker.create_from_options(options)
        self.frame_index = 0
        print(f"PoseDetector ready — tracking up to {config.MAX_PERSONS} people.")

    def detect(self, frame):
        """
        Input:  BGR numpy frame
        Output: (people, rendered_frame)
                people → list of lists, each inner list is 33 world landmarks
                          empty list if no one detected
        """
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts       = int(self.frame_index * (1000 / 30))

        result = self.landmarker.detect_for_video(mp_image, ts)

        self.frame_index += 1

        if not result.pose_world_landmarks:
            return [], frame

        if result.pose_landmarks:
            for i, person_landmarks in enumerate(result.pose_landmarks):
                color = PERSON_COLORS[i % len(PERSON_COLORS)]
                draw_landmarks(frame, person_landmarks, color)

        return result.pose_world_landmarks, frame

    def close(self):
        self.landmarker.close()