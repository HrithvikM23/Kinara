import cv2
import os
from datetime import datetime


class VideoWriter:

    def __init__(self, source_path: str = None):
        """
        source_path: original video path (used to match fps/resolution)
                     pass None for webcam
        """
        os.makedirs("output", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = f"output/processed_{timestamp}.mp4"

        self.writer = None
        self.fps    = 30
        self.source_path = source_path

    def initialize(self, frame):
        h, w    = frame.shape[:2]
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")

        if self.source_path:
            cap = cv2.VideoCapture(self.source_path)
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
            cap.release()

        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
        print(f"VideoWriter ready → {self.output_path}")

    def write(self, frame):
        if self.writer is None:
            self.initialize(frame)
        self.writer.write(frame)

    def close(self):
        if self.writer:
            self.writer.release()
            print(f"Video saved → {self.output_path}")