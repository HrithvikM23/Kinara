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


BODY_MODEL_SPECS: dict[str, ModelSpec] = {
    "low": ModelSpec(
        source_url="https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/main/onnx/model.onnx",
        relative_path=Path("models") / "body" / "movenet_lightning.onnx",
        input_size=192,
        input_name="input",
        input_dtype="int32",
    ),
    "mid": ModelSpec(
        source_url="https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/main/onnx/model.onnx",
        relative_path=Path("models") / "body" / "movenet_lightning.onnx",
        input_size=192,
        input_name="input",
        input_dtype="int32",
    ),
    "high": ModelSpec(
        source_url="https://huggingface.co/Xenova/movenet-singlepose-thunder/resolve/main/onnx/model.onnx",
        relative_path=Path("models") / "body" / "movenet_thunder.onnx",
        input_size=256,
        input_name="input",
        input_dtype="int32",
    ),
    "max": ModelSpec(
        source_url="https://huggingface.co/Xenova/movenet-singlepose-thunder/resolve/main/onnx/model.onnx",
        relative_path=Path("models") / "body" / "movenet_thunder.onnx",
        input_size=256,
        input_name="input",
        input_dtype="int32",
    ),
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

BODY_MODEL_ALIASES = {
    "body": "body",
    "movenet": "body",
}

HAND_MODEL_ALIASES = {
    "hand": "hand",
}


def ensure_model_file(project_root: Path, spec: ModelSpec) -> Path:
    destination = project_root / spec.relative_path
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    try:
        with urlopen(spec.source_url) as response, temp_path.open("wb") as output_file:
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

    return destination
