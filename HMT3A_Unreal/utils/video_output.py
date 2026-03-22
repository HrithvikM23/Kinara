from __future__ import annotations

from pathlib import Path

import cv2


class VideoWriter:
    def __init__(self, output_dir, enabled: bool = True, fps: float = 30.0, base_name: str = "tracked_output"):
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self._next_output_path(base_name)
        self.writer = None
        self.fps = fps if 0 < fps <= 240 else 30.0

    def _next_output_path(self, base_name: str) -> Path:
        candidate = self.output_dir / f"{base_name}.mp4"
        if not candidate.exists():
            return candidate

        index = 1
        while True:
            candidate = self.output_dir / f"{base_name}_{index}.mp4"
            if not candidate.exists():
                return candidate
            index += 1

    def _initialize(self, frame) -> None:
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (width, height))
        print(f"VideoWriter ready -> {self.output_path}")

    def write(self, frame) -> None:
        if not self.enabled:
            return

        if self.writer is None:
            self._initialize(frame)

        self.writer.write(frame)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            print(f"Video saved -> {self.output_path}")
