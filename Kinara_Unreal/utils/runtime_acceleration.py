from __future__ import annotations

from dataclasses import dataclass

from mediapipe.tasks import python as mp_python


@dataclass(slots=True)
class AccelerationProfile:
    prefer_gpu: bool
    mediapipe_delegate: object | None
    yolo_device: str
    torch_device: str
    cuda_available: bool
    cuda_version: str | None
    cudnn_available: bool
    cudnn_version: int | None
    gpu_name: str | None
    notes: list[str]


def detect_acceleration(prefer_gpu: bool = True) -> AccelerationProfile:
    notes: list[str] = []
    cuda_available = False
    cuda_version: str | None = None
    cudnn_available = False
    cudnn_version: int | None = None
    gpu_name: str | None = None
    yolo_device = "cpu"
    torch_device = "cpu"
    mediapipe_delegate = None

    if prefer_gpu:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            cuda_version = getattr(torch.version, "cuda", None)
            cudnn_available = bool(torch.backends.cudnn.is_available())
            cudnn_version = torch.backends.cudnn.version() if cudnn_available else None
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                yolo_device = "cuda:0"
                torch_device = "cuda"
                mediapipe_delegate = mp_python.BaseOptions.Delegate.GPU
                notes.append(f"CUDA available on {gpu_name}")
                if cuda_version:
                    notes.append(f"PyTorch CUDA runtime {cuda_version}")
                if cudnn_available:
                    notes.append(f"cuDNN available ({cudnn_version})")
                else:
                    notes.append("cuDNN not available in the active PyTorch runtime")
            else:
                notes.append("PyTorch CUDA runtime not available; using CPU backends")
        except Exception as exc:
            notes.append(f"PyTorch CUDA probe failed; using CPU backends ({exc})")
    else:
        notes.append("GPU acceleration disabled by configuration")

    return AccelerationProfile(
        prefer_gpu=prefer_gpu,
        mediapipe_delegate=mediapipe_delegate,
        yolo_device=yolo_device,
        torch_device=torch_device,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        cudnn_available=cudnn_available,
        cudnn_version=cudnn_version,
        gpu_name=gpu_name,
        notes=notes,
    )
