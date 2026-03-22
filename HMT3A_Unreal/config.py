from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

UDP_IP = "127.0.0.1"
UDP_PORT = 7000

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker_full.task"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

BODY_LANDMARKS = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

HAND_LANDMARKS = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]

BODY_INDEX_BY_NAME = {name: 11 + index for index, name in enumerate(BODY_LANDMARKS)}
HAND_INDEX_BY_NAME = {name: index for index, name in enumerate(HAND_LANDMARKS)}

LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12
LEFT_WRIST_INDEX = 15
RIGHT_WRIST_INDEX = 16
LEFT_HIP_INDEX = 23
RIGHT_HIP_INDEX = 24


@dataclass(slots=True)
class PipelineConfig:
    max_persons: int = 1
    udp_ip: str = UDP_IP
    udp_port: int = UDP_PORT
    min_pose_detection_confidence: float = 0.7
    min_pose_presence_confidence: float = 0.7
    min_tracking_confidence: float = 0.8
    min_hand_detection_confidence: float = 0.7
    min_hand_presence_confidence: float = 0.7
    min_hand_tracking_confidence: float = 0.7
    smoothing_alpha: float = 0.65
    preview: bool = True
    record_output: bool = True
    source_fps_fallback: float = 30.0
    output_dir: Path = OUTPUT_DIR
    model_dir: Path = MODEL_DIR


def ensure_runtime_directories(config: PipelineConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.model_dir.mkdir(parents=True, exist_ok=True)
