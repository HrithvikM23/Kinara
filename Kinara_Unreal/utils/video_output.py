from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2


class VideoWriter:
    def __init__(
        self,
        output_dir,
        enabled: bool = True,
        fps: float = 30.0,
        base_name: str = "tracked_output",
        frame_size: tuple[int, int] | None = None,
    ):
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self._next_output_path(base_name)
        self.writer: cv2.VideoWriter | None = None
        self.fps = fps if 0 < fps <= 240 else 30.0
        self.frame_size = frame_size

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

    def _initialize(self, frame: Any) -> None:
        if self.frame_size is not None:
            width, height = self.frame_size
        else:
            height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (width, height))
        print(f"VideoWriter ready -> {self.output_path}")

    def write(self, frame: Any) -> None:
        if not self.enabled:
            return

        if self.writer is None:
            self._initialize(frame)
        writer = self.writer
        if writer is None:
            return

        output_frame = frame
        if self.frame_size is not None:
            target_width, target_height = self.frame_size
            if frame.shape[1] != target_width or frame.shape[0] != target_height:
                output_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        writer.write(output_frame)

    def close(self) -> None:
        writer = self.writer
        if writer is not None:
            writer.release()
            print(f"Video saved -> {self.output_path}")
