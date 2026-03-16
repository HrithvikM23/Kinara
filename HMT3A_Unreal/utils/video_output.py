import cv2
import os
from typing import Optional


class VideoWriter:

    def __init__(self, source_path: Optional[str] = None):

        self.folder = "outputs"
        os.makedirs(self.folder, exist_ok=True)

        self.output_path = self._generate_filename()

        self.writer: Optional[cv2.VideoWriter] = None
        self.fps: float = 30.0
        self.source_path = source_path


    def _generate_filename(self) -> str:

        base = "video"
        ext = ".mp4"

        path = os.path.join(self.folder, base + ext)

        if not os.path.exists(path):
            return path

        i = 1
        while True:

            path = os.path.join(self.folder, f"{base}{i}{ext}")

            if not os.path.exists(path):
                return path

            i += 1


    def initialize(self, frame):

     h, w = frame.shape[:2]

     fourcc = cv2.VideoWriter.fourcc('m','p','4','v')

     if self.source_path is not None:
      cap = cv2.VideoCapture(self.source_path)
      fps = cap.get(cv2.CAP_PROP_FPS)
      if fps > 0:
        self.fps = fps
      cap.release()

     self.writer = cv2.VideoWriter(
        self.output_path,
        fourcc,
        self.fps,
        (w, h)
    )

     print(f"VideoWriter ready → {self.output_path}")


    def write(self, frame):

        if self.writer is None:
            self.initialize(frame)

        assert self.writer is not None
        self.writer.write(frame)


    def close(self):

        if self.writer is not None:
            self.writer.release()
            print(f"Video saved → {self.output_path}")