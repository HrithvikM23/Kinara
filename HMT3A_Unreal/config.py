from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
RAW_CAPTURE_DIR = OUTPUT_DIR / "raw_captures"
FINAL_RENDER_DIR = OUTPUT_DIR / "final_renders"
MOTION_EXPORT_DIR = OUTPUT_DIR / "motion_exports"

UDP_IP = "127.0.0.1"
UDP_PORT = 7000

PRIMARY_CAMERA_ROLE = "front"
OPTIONAL_CAMERA_ROLES = ("back", "right", "left", "up")
MAX_OPTIONAL_CAMERAS = len(OPTIONAL_CAMERA_ROLES)

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
LEFT_ELBOW_INDEX = 13
RIGHT_ELBOW_INDEX = 14
LEFT_WRIST_INDEX = 15
RIGHT_WRIST_INDEX = 16
LEFT_HIP_INDEX = 23
RIGHT_HIP_INDEX = 24


@dataclass(slots=True)
class CameraCalibration:
    role: str
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0
    confidence_weight: float = 1.0


def default_camera_calibrations() -> dict[str, CameraCalibration]:
    return {
        PRIMARY_CAMERA_ROLE: CameraCalibration(role=PRIMARY_CAMERA_ROLE),
        "back": CameraCalibration(role="back", rotation_deg=(0.0, 180.0, 0.0), confidence_weight=0.95),
        "right": CameraCalibration(role="right", rotation_deg=(0.0, 90.0, 0.0), confidence_weight=0.98),
        "left": CameraCalibration(role="left", rotation_deg=(0.0, -90.0, 0.0), confidence_weight=0.98),
        "up": CameraCalibration(role="up", rotation_deg=(-90.0, 0.0, 0.0), confidence_weight=0.92),
    }


def load_camera_calibrations(path: str | Path | None) -> dict[str, CameraCalibration]:
    calibrations = default_camera_calibrations()
    if path is None:
        return calibrations

    calibration_path = Path(path).expanduser()
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Calibration file must contain a JSON object keyed by camera role.")

    for role, entry in payload.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Calibration entry for {role!r} must be a JSON object.")

        rotation = tuple(float(value) for value in entry.get("rotation_deg", (0.0, 0.0, 0.0)))
        translation = tuple(float(value) for value in entry.get("translation", (0.0, 0.0, 0.0)))
        if len(rotation) != 3 or len(translation) != 3:
            raise ValueError(f"Calibration entry for {role!r} must define 3 rotation and 3 translation values.")

        calibrations[str(role)] = CameraCalibration(
            role=str(role),
            rotation_deg=rotation,
            translation=translation,
            scale=float(entry.get("scale", 1.0)),
            confidence_weight=float(entry.get("confidence_weight", 1.0)),
        )

    return calibrations


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
    body_smoothing_alpha: float = 0.6
    hand_smoothing_alpha: float = 0.38
    joint_deadband: float = 0.004
    hand_joint_deadband: float = 0.002
    max_position_jump: float = 0.28
    max_hand_position_jump: float = 0.12
    missing_joint_hold_frames: int = 3
    preview: bool = True
    record_output: bool = True
    render_output: bool = True
    enable_hand_roi: bool = True
    hand_roi_scale: float = 2.2
    hand_roi_min_size: int = 160
    hand_roi_fallback_to_full_frame: bool = True
    preview_target_fps: float | None = None
    source_fps_fallback: float = 30.0
    manual_fps_cap: float | None = None
    manual_resolution_width: int | None = None
    manual_resolution_height: int | None = None
    cluster_match_distance: float = 0.28
    track_match_distance: float = 0.36
    max_track_missed_frames: int = 18
    enable_motion_export: bool = True
    export_json: bool = True
    export_bvh: bool = True
    export_fbx: bool = True
    output_dir: Path = OUTPUT_DIR
    model_dir: Path = MODEL_DIR
    raw_capture_dir: Path = RAW_CAPTURE_DIR
    final_render_dir: Path = FINAL_RENDER_DIR
    motion_export_dir: Path = MOTION_EXPORT_DIR
    camera_calibration_path: str | None = None
    calibrations: dict[str, CameraCalibration] = field(default_factory=default_camera_calibrations)


def ensure_runtime_directories(config: PipelineConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.model_dir.mkdir(parents=True, exist_ok=True)
    config.raw_capture_dir.mkdir(parents=True, exist_ok=True)
    config.final_render_dir.mkdir(parents=True, exist_ok=True)
    config.motion_export_dir.mkdir(parents=True, exist_ok=True)
