from __future__ import annotations

import argparse
import cv2
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

from camera.capture import VideoCaptureSession
from config import PipelineConfig
from inference.rtmpose import ONNXPoseHandRunner
from network.osc_sender import OSCSender
from pipeline.pipeline import PoseHandPipeline
from utils.smoothing import LandmarkSmoother


def choose_video_gui() -> str | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[
            ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
            ("All Files", "*.*"),
        ],
    )
    root.destroy()
    return path or None


def resolve_source(args: argparse.Namespace) -> int | Path | None:
    """
    Determine the video source from CLI args or interactive prompt.
    Returns:
        int  -> webcam index
        Path -> video file path
        None -> user cancelled
    """
    # CLI: --source 0  or  --source path/to/video.mp4
    if args.source is not None:
        s = args.source.strip()
        if s.isdigit():
            return int(s)
        p = Path(s)
        if not p.exists():
            print(f"Error: file not found: {p}")
            return None
        return p

    # Interactive prompt
    print("Select input source:")
    print("  1. Webcam")
    print("  2. Video file ")
    choice = input("Enter choice [1/2]: ").strip()

    if choice == "1":
        idx = input("Webcam index: ").strip()
        return int(idx) if idx.isdigit() else 0

    if choice == "2":
        path = choose_video_gui()
        if not path:
            print("No file selected.")
            return None
        return Path(path)
    
    print("Invalid choice.")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Pose and Hand Landmark Pipeline")
    parser.add_argument(
        "--source",
        help="Webcam index (e.g. 0) or path to a video file. "
             "If omitted, an interactive prompt runs.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the live OpenCV preview window.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Resolve source BEFORE building any objects                          #
    # ------------------------------------------------------------------ #
    source = resolve_source(args)
    if source is None:
        return

    # ------------------------------------------------------------------ #
    # Build config and inject source                                      #
    # ------------------------------------------------------------------ #
    config = PipelineConfig(video_path=source)
    if args.no_preview:
        config.enable_preview = False

    # ------------------------------------------------------------------ #
    # Build pipeline components                                           #
    # ------------------------------------------------------------------ #
    session = VideoCaptureSession(config.video_path, config.output_path)
    runner = ONNXPoseHandRunner(config)
    smoother = LandmarkSmoother()
    osc_sender = OSCSender()
    pipeline = PoseHandPipeline(config, runner, smoother, osc_sender)

    # ------------------------------------------------------------------ #
    # Run                                                                 #
    # ------------------------------------------------------------------ #
    try:
        while True:
            ok, frame = session.read()
            if not ok or frame is None:
                break

            rendered = pipeline.process_frame(frame)
            session.write(rendered)

            if config.enable_preview:
                cv2.imshow("Pose + Hand Landmarks", rendered)
                if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
                    break
    finally:
        session.close()
        osc_sender.close()
        cv2.destroyAllWindows()

    print(f"Saved: {config.output_path}")


if __name__ == "__main__":
    main()