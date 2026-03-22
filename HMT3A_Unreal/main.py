from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from camera.video_input import choose_video
from config import PipelineConfig, ensure_runtime_directories
from network.packet_builder import build_packet
from network.udp_sender import UDPSender
from pose_server.pose_detector import PoseDetector
from utils.video_output import VideoWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HMT3A Unreal v2 motion pipeline")
    parser.add_argument("--source", help="Webcam index like 0, or a video path")
    parser.add_argument("--max-persons", type=int, help="Maximum number of people to track")
    parser.add_argument("--udp-ip", help="UDP target IP")
    parser.add_argument("--udp-port", type=int, help="UDP target port")
    parser.add_argument("--smoothing-alpha", type=float, help="EMA smoothing alpha between 0 and 1")
    parser.add_argument("--no-preview", action="store_true", help="Disable live OpenCV preview")
    parser.add_argument("--no-record", action="store_true", help="Disable processed video recording")
    return parser.parse_args()


def prompt_int(prompt: str, minimum: int, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        return value


def prompt_for_source() -> int | str:
    while True:
        print("=" * 50)
        print("Select input source:")
        print("1. Webcam")
        print("2. Video file")
        print("=" * 50)
        choice = input("Enter choice: ").strip()

        if choice == "1":
            return 0
        if choice == "2":
            return choose_video()
        print("Please choose 1 or 2.")


def resolve_source(source_arg: str | None) -> int | str:
    if not source_arg:
        return prompt_for_source()

    if source_arg.isdigit():
        return int(source_arg)

    return str(Path(source_arg).expanduser())


def build_config(args: argparse.Namespace) -> PipelineConfig:
    max_persons = args.max_persons if args.max_persons is not None else prompt_int(
        "Enter number of people to track",
        minimum=1,
        default=1,
    )

    config = PipelineConfig(
        max_persons=max_persons,
        preview=not args.no_preview,
        record_output=not args.no_record,
    )

    if args.udp_ip:
        config.udp_ip = args.udp_ip
    if args.udp_port is not None:
        config.udp_port = args.udp_port
    if args.smoothing_alpha is not None:
        config.smoothing_alpha = args.smoothing_alpha

    ensure_runtime_directories(config)
    return config


def get_source_fps(cap: cv2.VideoCapture, fallback: float) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        return fallback
    return fps


def get_timestamp_ms(
    cap: cv2.VideoCapture,
    source: int | str,
    frame_index: int,
    source_fps: float,
    start_time: float,
) -> int:
    if isinstance(source, str):
        position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if position_ms > 0:
            return int(position_ms)
        return int(frame_index * (1000.0 / max(source_fps, 0.001)))

    return int((time.perf_counter() - start_time) * 1000.0)


def draw_runtime_overlay(frame, frame_index: int, people_count: int, runtime_fps: float) -> None:
    cv2.putText(
        frame,
        f"Frame: {frame_index}",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (40, 255, 40),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"People: {people_count}",
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (40, 255, 40),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Runtime FPS: {runtime_fps:.1f}",
        (12, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (40, 255, 40),
        2,
        cv2.LINE_AA,
    )


def run_pipeline(source: int | str, config: PipelineConfig) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    source_fps = get_source_fps(cap, config.source_fps_fallback)
    detector = PoseDetector(config)
    sender = UDPSender(config.udp_ip, config.udp_port)
    writer = VideoWriter(
        output_dir=config.output_dir,
        enabled=config.record_output,
        fps=source_fps,
    )

    frame_index = 0
    started_at = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = get_timestamp_ms(cap, source, frame_index, source_fps, started_at)
            people, rendered = detector.detect(frame, timestamp_ms)

            elapsed = max(time.perf_counter() - started_at, 0.001)
            runtime_fps = (frame_index + 1) / elapsed

            draw_runtime_overlay(rendered, frame_index, len(people), runtime_fps)

            packet = build_packet(
                persons=people,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                source_fps=source_fps,
            )
            sender.send(packet)

            writer.write(rendered)

            print(
                f"Frame {frame_index:04d} | "
                f"{len(people)} person(s) | "
                f"{len(packet)} bytes | "
                f"{runtime_fps:05.1f} FPS"
            )

            if config.preview:
                cv2.imshow("HMT3A Unreal v2", rendered)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_index += 1
    finally:
        cap.release()
        writer.close()
        detector.close()
        sender.close()
        cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    config = build_config(args)
    source = resolve_source(args.source)

    try:
        run_pipeline(source, config)
    except Exception as exc:
        print(f"Pipeline failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
