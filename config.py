from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PipelineConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    body_model_path: Path = field(init=False)
    hand_model_path: Path = field(init=False)
    video_path: int | Path = field(
        default_factory=lambda: Path(r"C:\Users\HrithvikM's PC\Downloads\WhatsApp Video 2026-04-18 at 10.01.30 PM.mp4")
    )
    output_path: Path = field(init=False)
    body_input_name: str = "input"
    hand_input_name: str = "images"
    body_input_size: int = 192
    hand_input_size: int = 640
    body_conf_threshold: float = 0.30
    video_source: int | Path | str = 0 
    hand_det_threshold: float = 0.15
    hand_kp_threshold: float = 0.20
    hand_box_min_size: int = 160
    hand_box_scale: float = 2.0
    enable_preview: bool = True
    provider_names: tuple[str, ...] = ("CUDAExecutionProvider",)

    def __post_init__(self) -> None:
        self.body_model_path = self.project_root / "models" / "movenet.onnx"
        self.hand_model_path = self.project_root / "models" / "hand_pose.onnx"

        # Derive an output filename from the video path when available,
        # otherwise fall back to a generic name for webcam input.
        if isinstance(self.video_path, Path):
            stem = self.video_path.stem
        else:
            stem = f"webcam_{self.video_path}"

        self.output_path = self.project_root / "outputs" / f"{stem}_tracked.mp4"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)


BODY_EDGES = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6), (5, 11),
    (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

BODY_KEYPOINTS = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

WRIST_TO_ELBOW = {9: 7, 10: 8}