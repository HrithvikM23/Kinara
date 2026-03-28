from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2

from camera.video_input import choose_video
from config import (
    MAX_OPTIONAL_CAMERAS,
    OPTIONAL_CAMERA_ROLES,
    PRIMARY_CAMERA_ROLE,
    PipelineConfig,
    ensure_runtime_directories,
)
from network.packet_builder import build_packet
from network.udp_sender import UDPSender
from pose_server.pose_detector import PoseDetector
from process.multi_camera_fusion import MultiCameraFusion
from utils.video_output import VideoWriter


@dataclass(slots=True)
class SourceAssignment:
    role: str
    source: int | str
    label: str


@dataclass(slots=True)
class SourceProfile:
    role: str
    fps: float
    width: int
    height: int


@dataclass(slots=True)
class SourceState:
    assignment: SourceAssignment
    cap: cv2.VideoCapture
    detector: PoseDetector
    source_fps: float
    current_source_index: int = -1
    latest_frame: Any | None = None


class SessionEnded(Exception):
    pass



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HMT3A Unreal v2 motion pipeline")
    parser.add_argument("--source", help="Single webcam index like 0, or a video path")
    parser.add_argument("--max-persons", type=int, help="Maximum number of people to track")
    parser.add_argument("--udp-ip", help="UDP target IP")
    parser.add_argument("--udp-port", type=int, help="UDP target port")
    parser.add_argument("--smoothing-alpha", type=float, help="EMA smoothing alpha between 0 and 1")
    parser.add_argument("--preview-fps", type=float, help="Target FPS for the live block-animation preview")
    parser.add_argument("--fps-cap", type=float, help="Manual FPS cap for final processing")
    parser.add_argument("--width", type=int, help="Manual output width override")
    parser.add_argument("--height", type=int, help="Manual output height override")
    parser.add_argument("--no-preview", action="store_true", help="Disable live OpenCV preview")
    parser.add_argument("--no-record", action="store_true", help="Disable processed video recording")
    return parser.parse_args()



def prompt_int(prompt: str, minimum: int, default: int, maximum: int | None = None) -> int:
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
        if maximum is not None and value > maximum:
            print(f"Please enter a value <= {maximum}.")
            continue
        return value



def prompt_float(prompt: str, minimum: float, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue
        if value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        return value



def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    default_hint = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{default_hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}



def prompt_for_input_mode() -> str:
    while True:
        print("=" * 50)
        print("Select input source:")
        print("1. Webcam")
        print("2. Video file")
        print("=" * 50)
        choice = input("Enter choice: ").strip()
        if choice == "1":
            return "webcam"
        if choice == "2":
            return "video"
        print("Please choose 1 or 2.")



def resolve_source(source_arg: str | None) -> int | str:
    if not source_arg:
        raise ValueError("A source argument is required.")
    if source_arg.isdigit():
        return int(source_arg)
    return str(Path(source_arg).expanduser())



def prompt_camera_roles() -> list[str]:
    print("Primary camera is FRONT by default.")
    extra_cameras = prompt_int(
        "Enter number of extra cameras besides FRONT",
        minimum=0,
        default=0,
        maximum=MAX_OPTIONAL_CAMERAS,
    )

    roles = [PRIMARY_CAMERA_ROLE]
    available_roles = list(OPTIONAL_CAMERA_ROLES)

    for camera_number in range(extra_cameras):
        print("=" * 50)
        print(f"Assign role for extra camera {camera_number + 1}:")
        for index, role in enumerate(available_roles, start=1):
            print(f"{index}. {role.title()}")
        print("=" * 50)

        choice = prompt_int(
            "Enter role number",
            minimum=1,
            default=1,
            maximum=len(available_roles),
        )
        roles.append(available_roles.pop(choice - 1))

    return roles



def prompt_source_assignments() -> list[SourceAssignment]:
    input_mode = prompt_for_input_mode()
    roles = prompt_camera_roles()
    assignments: list[SourceAssignment] = []

    for index, role in enumerate(roles):
        if input_mode == "webcam":
            source_index = prompt_int(
                f"Enter webcam index for {role.upper()}",
                minimum=0,
                default=index,
            )
            assignments.append(SourceAssignment(role=role, source=source_index, label=f"{role.title()} camera"))
            continue

        video_path = choose_video(title=f"Select video file for {role.title()} camera")
        assignments.append(SourceAssignment(role=role, source=video_path, label=f"{role.title()} video"))

    return assignments



def build_assignments(args: argparse.Namespace) -> list[SourceAssignment]:
    if args.source:
        return [SourceAssignment(role=PRIMARY_CAMERA_ROLE, source=resolve_source(args.source), label="Front source")]
    return prompt_source_assignments()



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
    if args.preview_fps is not None and args.preview_fps > 0:
        config.preview_target_fps = args.preview_fps

    if args.fps_cap is not None and args.fps_cap > 0:
        config.manual_fps_cap = args.fps_cap
    elif prompt_yes_no("Manually set an FPS cap?", default=False):
        config.manual_fps_cap = prompt_float("Enter FPS cap", minimum=1.0, default=30.0)

    if args.width is not None or args.height is not None:
        if args.width is None or args.height is None:
            raise ValueError("Both --width and --height must be provided together.")
        if args.width < 1 or args.height < 1:
            raise ValueError("Width and height must be positive integers.")
        config.manual_resolution_width = args.width
        config.manual_resolution_height = args.height
    elif prompt_yes_no("Manually set a fixed resolution?", default=False):
        config.manual_resolution_width = prompt_int("Enter width", minimum=1, default=1280)
        config.manual_resolution_height = prompt_int("Enter height", minimum=1, default=720)

    ensure_runtime_directories(config)
    return config



def get_target_frame_size(config: PipelineConfig) -> tuple[int, int] | None:
    if config.manual_resolution_width and config.manual_resolution_height:
        return (config.manual_resolution_width, config.manual_resolution_height)
    return None



def get_source_fps(cap: cv2.VideoCapture, fallback: float) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        return fallback
    return fps



def get_source_dimensions(cap: cv2.VideoCapture) -> tuple[int, int]:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return width, height



def apply_capture_overrides(cap: cv2.VideoCapture, source: int | str, config: PipelineConfig) -> None:
    if not isinstance(source, int):
        return

    target_size = get_target_frame_size(config)
    if target_size is not None:
        target_width, target_height = target_size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

    if config.manual_fps_cap is not None:
        cap.set(cv2.CAP_PROP_FPS, config.manual_fps_cap)



def prepare_frame(frame, config: PipelineConfig) -> Any:
    target_size = get_target_frame_size(config)
    if target_size is None:
        return frame

    target_width, target_height = target_size
    if frame.shape[1] == target_width and frame.shape[0] == target_height:
        return frame

    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)



def inspect_source_profile(assignment: SourceAssignment, config: PipelineConfig) -> SourceProfile:
    cap = cv2.VideoCapture(assignment.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not inspect source for {assignment.role}: {assignment.source}")

    try:
        apply_capture_overrides(cap, assignment.source, config)
        fps = get_source_fps(cap, config.source_fps_fallback)
        width, height = get_source_dimensions(cap)
        target_size = get_target_frame_size(config)
        if target_size is not None:
            width, height = target_size
        return SourceProfile(role=assignment.role, fps=fps, width=width, height=height)
    finally:
        cap.release()



def collect_source_profiles(assignments: list[SourceAssignment], config: PipelineConfig) -> list[SourceProfile]:
    return [inspect_source_profile(assignment, config) for assignment in assignments]



def determine_common_fps(profiles: list[SourceProfile], config: PipelineConfig) -> float:
    fps_values = [profile.fps for profile in profiles if profile.fps > 0]
    effective_fps = min(fps_values) if fps_values else config.source_fps_fallback
    if config.manual_fps_cap is not None:
        effective_fps = min(effective_fps, config.manual_fps_cap)
    return max(effective_fps, 1.0)



def get_timestamp_ms(source: int | str, frame_index: int, source_fps: float, start_time: float) -> int:
    if isinstance(source, str):
        return int(frame_index * (1000.0 / max(source_fps, 0.001)))
    return int((time.perf_counter() - start_time) * 1000.0)



def draw_runtime_overlay(frame, frame_index: int, people_count: int, runtime_fps: float, stage_label: str) -> None:
    cv2.putText(frame, f"Stage: {stage_label}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 255, 40), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {frame_index}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 255, 40), 2, cv2.LINE_AA)
    cv2.putText(frame, f"People: {people_count}", (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 255, 40), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {runtime_fps:.1f}", (12, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 255, 40), 2, cv2.LINE_AA)



def open_source_states(assignments: list[SourceAssignment], config: PipelineConfig) -> list[SourceState]:
    states: list[SourceState] = []
    primary_role = assignments[0].role if assignments else PRIMARY_CAMERA_ROLE

    try:
        for assignment in assignments:
            cap = cv2.VideoCapture(assignment.source)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open source for {assignment.role}: {assignment.source}")

            source_config = replace(
                config,
                render_output=config.render_output and assignment.role == primary_role,
            )
            apply_capture_overrides(cap, assignment.source, config)
            states.append(
                SourceState(
                    assignment=assignment,
                    cap=cap,
                    detector=PoseDetector(source_config),
                    source_fps=get_source_fps(cap, config.source_fps_fallback),
                )
            )
    except Exception:
        close_source_states(states)
        raise

    return states



def close_source_states(states: list[SourceState]) -> None:
    for state in states:
        try:
            state.cap.release()
        except Exception:
            pass
        try:
            state.detector.close()
        except Exception:
            pass



def read_live_frames(states: list[SourceState], config: PipelineConfig) -> dict[str, Any]:
    frames_by_role: dict[str, Any] = {}

    for state in states:
        ret, frame = state.cap.read()
        if not ret:
            raise SessionEnded()
        prepared_frame = prepare_frame(frame, config)
        frames_by_role[state.assignment.role] = prepared_frame
        state.latest_frame = prepared_frame
        state.current_source_index += 1

    return frames_by_role



def read_frame_for_output(state: SourceState, output_frame_index: int, session_fps: float, config: PipelineConfig) -> Any | None:
    target_source_index = int(round(output_frame_index * (state.source_fps / max(session_fps, 0.001))))

    while state.current_source_index < target_source_index:
        ret, frame = state.cap.read()
        if not ret:
            return None
        state.latest_frame = prepare_frame(frame, config)
        state.current_source_index += 1

    return state.latest_frame



def render_detection_group(states: list[SourceState], frames_by_role: dict[str, Any], timestamp_ms: int) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    detections_by_role: dict[str, list[dict]] = {}
    rendered_by_role: dict[str, Any] = {}

    for state in states:
        role = state.assignment.role
        people, rendered = state.detector.detect(frames_by_role[role], timestamp_ms)
        detections_by_role[role] = people
        rendered_by_role[role] = rendered

    return detections_by_role, rendered_by_role



def run_preview_session(assignments: list[SourceAssignment], config: PipelineConfig, session_fps: float) -> list[SourceAssignment]:
    preview_config = replace(
        config,
        preview=True,
        record_output=False,
        render_output=config.preview,
        enable_hand_roi=False,
        hand_roi_fallback_to_full_frame=False,
    )
    states = open_source_states(assignments, preview_config)
    raw_writers: dict[str, VideoWriter] = {}
    sender = UDPSender(config.udp_ip, config.udp_port)
    fusion = MultiCameraFusion(preview_config, primary_role=assignments[0].role)
    started_at = time.perf_counter()
    preview_frame_index = 0
    preview_output_fps = max(min(config.preview_target_fps, session_fps), 1.0) if config.preview_target_fps is not None else max(session_fps, 1.0)
    last_preview_timestamp = -(1000.0 / preview_output_fps)
    latest_primary_render = None
    last_preview_log_at = 0.0

    try:
        for state in states:
            if isinstance(state.assignment.source, int):
                raw_writers[state.assignment.role] = VideoWriter(
                    output_dir=config.raw_capture_dir,
                    enabled=True,
                    fps=state.source_fps,
                    base_name=f"{state.assignment.role}_raw",
                    frame_size=get_target_frame_size(config),
                )

        preview_interval_ms = 1000.0 / preview_output_fps

        while True:
            frames_by_role = read_live_frames(states, config)

            for state in states:
                raw_writer = raw_writers.get(state.assignment.role)
                if raw_writer is not None:
                    raw_writer.write(frames_by_role[state.assignment.role])

            timestamp_ms = int((time.perf_counter() - started_at) * 1000.0)
            should_process = (timestamp_ms - last_preview_timestamp) >= preview_interval_ms

            if should_process:
                detections_by_role, rendered_by_role = render_detection_group(states, frames_by_role, timestamp_ms)
                fused_people = fusion.fuse_frame(detections_by_role)
                latest_primary_render = rendered_by_role[assignments[0].role]

                elapsed = max(time.perf_counter() - started_at, 0.001)
                runtime_fps = (preview_frame_index + 1) / elapsed
                draw_runtime_overlay(latest_primary_render, preview_frame_index, len(fused_people), runtime_fps, "PREVIEW")

                packet = build_packet(
                    persons=fused_people,
                    frame_index=preview_frame_index,
                    timestamp_ms=timestamp_ms,
                    source_fps=preview_output_fps,
                )
                sender.send(packet)
                now = time.perf_counter()
                if (now - last_preview_log_at) >= 1.0:
                    print(
                        f"[PREVIEW] fused | Frame {preview_frame_index:04d} | "
                        f"{len(fused_people)} person(s) | {len(packet)} bytes | {runtime_fps:05.1f} FPS"
                    )
                    last_preview_log_at = now

                preview_frame_index += 1
                last_preview_timestamp = timestamp_ms
            else:
                latest_primary_render = frames_by_role[assignments[0].role]

            if config.preview:
                cv2.imshow(f"{assignments[0].role.title()} Preview", latest_primary_render)
                for assignment in assignments[1:]:
                    cv2.imshow(f"{assignment.role.title()} Feed", frames_by_role[assignment.role])
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    except SessionEnded:
        pass
    finally:
        close_source_states(states)
        for writer in raw_writers.values():
            writer.close()
        sender.close()
        cv2.destroyAllWindows()

    return _build_final_assignments(assignments, raw_writers)



def _build_final_assignments(assignments: list[SourceAssignment], raw_writers: dict[str, VideoWriter]) -> list[SourceAssignment]:
    final_assignments: list[SourceAssignment] = []

    for assignment in assignments:
        raw_writer = raw_writers.get(assignment.role)
        final_source = str(raw_writer.output_path) if raw_writer is not None else assignment.source
        final_assignments.append(SourceAssignment(role=assignment.role, source=final_source, label=assignment.label))

    return final_assignments



def run_final_render(assignments: list[SourceAssignment], config: PipelineConfig, session_fps: float) -> None:
    final_config = replace(config, preview=config.preview, record_output=True, render_output=True)
    states = open_source_states(assignments, final_config)
    sender = UDPSender(config.udp_ip, config.udp_port)
    fusion = MultiCameraFusion(final_config, primary_role=assignments[0].role)
    writer = VideoWriter(
        output_dir=config.final_render_dir,
        enabled=config.record_output,
        fps=session_fps,
        base_name="fused_final",
        frame_size=get_target_frame_size(config),
    )
    output_frame_index = 0
    started_at = time.perf_counter()
    last_final_log_at = 0.0

    print(f"Using final render FPS: {session_fps:.3f}")
    target_size = get_target_frame_size(config)
    if target_size is not None:
        print(f"Using fixed resolution: {target_size[0]}x{target_size[1]}")
    else:
        print("Using recorded/native resolution for each source.")

    try:
        while True:
            timestamp_ms = int(output_frame_index * (1000.0 / max(session_fps, 0.001)))
            frames_by_role = {}

            for state in states:
                frame = read_frame_for_output(state, output_frame_index, session_fps, final_config)
                if frame is None:
                    raise SessionEnded()
                frames_by_role[state.assignment.role] = frame

            detections_by_role, rendered_by_role = render_detection_group(states, frames_by_role, timestamp_ms)
            fused_people = fusion.fuse_frame(detections_by_role)
            primary_render = rendered_by_role[assignments[0].role]

            elapsed = max(time.perf_counter() - started_at, 0.001)
            runtime_fps = (output_frame_index + 1) / elapsed
            draw_runtime_overlay(primary_render, output_frame_index, len(fused_people), runtime_fps, "FINAL")

            packet = build_packet(
                persons=fused_people,
                frame_index=output_frame_index,
                timestamp_ms=timestamp_ms,
                source_fps=session_fps,
            )
            sender.send(packet)
            writer.write(primary_render)

            now = time.perf_counter()
            if (now - last_final_log_at) >= 1.0:
                print(
                    f"[FINAL] fused_final | Frame {output_frame_index:04d} | "
                    f"{len(fused_people)} person(s) | {len(packet)} bytes | {runtime_fps:05.1f} FPS"
                )
                last_final_log_at = now

            if config.preview:
                cv2.imshow(f"{assignments[0].role.title()} Final Render", primary_render)
                for assignment in assignments[1:]:
                    cv2.imshow(f"{assignment.role.title()} Final Feed", rendered_by_role[assignment.role])
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            output_frame_index += 1
    except SessionEnded:
        pass
    finally:
        close_source_states(states)
        writer.close()
        sender.close()
        cv2.destroyAllWindows()



def run_session(assignments: list[SourceAssignment], config: PipelineConfig) -> None:
    profiles = collect_source_profiles(assignments, config)
    session_fps = determine_common_fps(profiles, config)

    print("Session source profile:")
    for profile in profiles:
        print(f"- {profile.role.title()}: {profile.width}x{profile.height} @ {profile.fps:.3f} FPS")
    print(f"- Common processing FPS: {session_fps:.3f}")

    final_sources = run_preview_session(assignments, config, session_fps)

    if not prompt_yes_no("Proceed with final render?", default=True):
        print("Capture session saved. Final render skipped.")
        return

    run_final_render(final_sources, config, session_fps)



def main() -> int:
    args = parse_args()

    try:
        config = build_config(args)
        assignments = build_assignments(args)
        run_session(assignments, config)
    except Exception as exc:
        print(f"Pipeline failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())







