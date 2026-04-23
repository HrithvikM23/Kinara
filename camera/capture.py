from __future__ import annotations

from pathlib import Path

import cv2


class VideoCaptureSession:
    def __init__(self, video_path: int | Path, output_path: Path):
        source = video_path if isinstance(video_path, int) else str(video_path)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.fps <= 0:
            self.fps = 30.0

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.frame_width, self.frame_height))
        if not self.writer.isOpened():
            self.cap.release()
            raise RuntimeError(f"Could not open video writer: {output_path}")

    def read(self):
        return self.cap.read()

    def write(self, frame) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()
