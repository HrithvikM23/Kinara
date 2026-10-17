from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from utils.logging import safe_print

IS_FROZEN = getattr(sys, "frozen", False)
EXE_DIR = Path(sys.executable).resolve().parent if IS_FROZEN else Path(__file__).resolve().parent.parent
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", str(EXE_DIR))) if IS_FROZEN else Path(__file__).resolve().parent.parent

PROJECT_ROOT = RESOURCE_ROOT
LOCAL_RUNTIME_DIR = EXE_DIR / ".kinara_runtime"
LOCAL_MODELS_DIR = LOCAL_RUNTIME_DIR / "models"
VENDOR_DIR = (LOCAL_RUNTIME_DIR if IS_FROZEN else PROJECT_ROOT) / f".vendor_py{sys.version_info.major}{sys.version_info.minor}"
ULTRALYTICS_CONFIG_DIR = (LOCAL_RUNTIME_DIR if IS_FROZEN else PROJECT_ROOT) / ".ultralytics"
RUNTIME_CACHE_DIR = (LOCAL_RUNTIME_DIR if IS_FROZEN else PROJECT_ROOT) / ".kinara_runtime" / "cache"

REQUIRED_PROJECT_FILES = (Path("pyproject.toml"),)

MODULE_TO_PACKAGE = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
    "onnxruntime": "onnxruntime",
    "mediapipe": "mediapipe==0.10.21",
    "rtmlib": "rtmlib",
    "aniposelib": "aniposelib>=0.7,<0.8",
}


@dataclass(frozen=True, slots=True)
class ModuleStatus:
    module_name: str
    ok: bool
    error: str | None = None


@dataclass(slots=True)
class RuntimeReport:
    nvidia_driver_detected: bool = False
    cuda_bin_dirs: list[Path] = field(default_factory=list)
    cudnn_bin_dirs: list[Path] = field(default_factory=list)
    cuda_include_dirs: list[Path] = field(default_factory=list)
    cudnn_include_dirs: list[Path] = field(default_factory=list)
    path_updates: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class TerminalProgress:
    def __init__(self, total_steps: int, width: int = 30) -> None:
        self.total_steps = max(1, total_steps)
        self.width = max(10, width)
        self.current_step = 0
        self._last_render_length = 0

    def _render(self, message: str) -> None:
        ratio = min(1.0, self.current_step / self.total_steps)
        filled = int(round(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        line = f"\r[{bar}] {self.current_step}/{self.total_steps} {int(ratio * 100):3d}% {message}"
        padding = max(0, self._last_render_length - len(line))
        safe_print(line + (" " * padding), end="")
        self._last_render_length = len(line)

    def note(self, message: str) -> None:
        self._render(message)

    def advance(self, message: str) -> None:
        self.current_step = min(self.total_steps, self.current_step + 1)
        self._render(message)

    def break_line(self) -> None:
        safe_print()
        self._last_render_length = 0

    def finish(self, message: str) -> None:
        self.current_step = self.total_steps
        self._render(message)
        self.break_line()
