from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen


@dataclass(frozen=True, slots=True)
class ModelSpec:
    source_url: str
    relative_path: Path
    input_size: int
    input_name: str
    input_dtype: str


DEFAULT_BODY_MODEL = "yolo11x-pose.pt"
BODY_MODEL_URLS: dict[str, str] = {
    "yolo11n-pose.pt": "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11n-pose.pt",
    "yolo11s-pose.pt": "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11s-pose.pt",
    "yolo11m-pose.pt": "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11m-pose.pt",
    "yolo11l-pose.pt": "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11l-pose.pt",
    "yolo11x-pose.pt": "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11x-pose.pt",
}

HAND_MODEL_SPECS: dict[str, ModelSpec] = {
    "low": ModelSpec(
        source_url="https://huggingface.co/poptoz/yolo26-hand-pose-face-detection/resolve/main/models/yolo26_hand_pose_fp16.onnx",
        relative_path=Path("models") / "hand" / "yolo26_hand_pose_fp16.onnx",
        input_size=640,
        input_name="images",
        input_dtype="float32",
    ),
    "mid": ModelSpec(
        source_url="https://huggingface.co/poptoz/yolo26-hand-pose-face-detection/resolve/main/models/yolo26_hand_pose_fp16.onnx",
        relative_path=Path("models") / "hand" / "yolo26_hand_pose_fp16.onnx",
        input_size=640,
        input_name="images",
        input_dtype="float32",
    ),
    "high": ModelSpec(
        source_url="https://huggingface.co/poptoz/yolo26-hand-pose-face-detection/resolve/main/models/yolo26_hand_pose_fp32.onnx",
        relative_path=Path("models") / "hand" / "yolo26_hand_pose_fp32.onnx",
        input_size=640,
        input_name="images",
        input_dtype="float32",
    ),
    "max": ModelSpec(
        source_url="https://huggingface.co/poptoz/yolo26-hand-pose-face-detection/resolve/main/models/yolo26_hand_pose_fp32.onnx",
        relative_path=Path("models") / "hand" / "yolo26_hand_pose_fp32.onnx",
        input_size=640,
        input_name="images",
        input_dtype="float32",
    ),
}


def _download_to_path(source_url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    try:
        with urlopen(source_url) as response, temp_path.open("wb") as output_file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output_file.write(chunk)
        temp_path.replace(destination)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def ensure_model_file(project_root: Path, spec: ModelSpec) -> Path:
    destination = project_root / spec.relative_path
    if destination.exists():
        return destination

    _download_to_path(spec.source_url, destination)
    return destination


def ensure_body_model_file(project_root: Path, model_name_or_path: str) -> Path:
    candidate_path = Path(model_name_or_path)
    if candidate_path.is_absolute() or candidate_path.parent != Path("."):
        return candidate_path

    destination = project_root / "models" / "body" / candidate_path.name
    if destination.exists():
        return destination

    source_url = BODY_MODEL_URLS.get(candidate_path.name)
    if source_url is None:
        return destination

    _download_to_path(source_url, destination)
    return destination
