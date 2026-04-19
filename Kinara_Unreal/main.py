from __future__ import annotations
import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast
import cv2
from camera.video_input import choose_video
from config import (
    IdentityProfile,
    MAX_OPTIONAL_CAMERAS,
    OPTIONAL_CAMERA_ROLES,
    PRIMARY_CAMERA_ROLE,
    PipelineConfig,
    ensure_runtime_directories,
    load_camera_calibrations,
)
from network.packet_builder import build_packet
from network.udp_sender import UDPSender
from pose_server.assisted_pose_detector import PoseDetector
from process.multi_camera_fusion import MultiCameraFusion
from utils.motion_export import MotionExporter
from utils.runtime_acceleration import detect_acceleration
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
    parser = argparse.ArgumentParser(description="Kinara Unreal v2 motion pipeline")
    parser.add_argument("--source", help="Single webcam index like 0, or a video path")
    parser.add_argument("--max-persons", type=int, help="Maximum number of people to track")
    parser.add_argument("--udp-ip", help="UDP target IP")
    parser.add_argument("--udp-port", type=int, help="UDP target port")
    parser.add_argument("--smoothing-alpha", type=float, help="Base smoothing alpha between 0 and 1")
    parser.add_argument("--preview-fps", type=float, help="Target FPS for the live block-animation preview")
    parser.add_argument("--fps-cap", type=float, help="Manual FPS cap for final processing")
    parser.add_argument("--width", type=int, help="Manual output width override")
    parser.add_argument("--height", type=int, help="Manual output height override")
    parser.add_argument("--calibration-file", help="Optional JSON file that defines per-role camera calibration")
    parser.add_argument("--no-preview", action="store_true", help="Disable live OpenCV preview")
    parser.add_argument("--no-record", action="store_true", help="Disable processed video recording")
    parser.add_argument("--no-motion-export", action="store_true", help="Disable fused motion export files")
    parser.add_argument("--no-json-export", action="store_true", help="Disable fused motion JSON export")
    parser.add_argument("--no-bvh-export", action="store_true", help="Disable BVH animation export")
    parser.add_argument("--no-fbx-export", action="store_true", help="Disable FBX animation export")
    parser.add_argument("--no-identity-memory", action="store_true", help="Disable color-based identity memory")
    parser.add_argument("--body-detection-confidence", type=float, help="Minimum MediaPipe body detection confidence")
    parser.add_argument("--body-presence-confidence", type=float, help="Minimum MediaPipe body presence confidence")
    parser.add_argument("--body-tracking-confidence", type=float, help="Minimum MediaPipe body tracking confidence")
    parser.add_argument("--hand-detection-confidence", type=float, help="Minimum MediaPipe hand detection confidence")
    parser.add_argument("--hand-presence-confidence", type=float, help="Minimum MediaPipe hand presence confidence")
    parser.add_argument("--hand-tracking-confidence", type=float, help="Minimum MediaPipe hand tracking confidence")
    parser.add_argument("--enable-yolo", nargs="?", const="yolov8x.pt", metavar="MODEL",
        help="Enable YOLO person assist during final render. "
             "Optionally pass a model name or path (default: yolov8x.pt). "
             "Example: --enable-yolo yolov8n.pt")
    parser.add_argument("--yolo-confidence", type=float, help="Minimum YOLO person confidence")
    parser.add_argument("--enable-rcnn", action="store_true",
        help="Enable Mask R-CNN refinement during final render (requires --enable-yolo)")
    parser.add_argument("--mask-rcnn-score", type=float, help="Minimum Mask R-CNN person score")
    parser.add_argument("--rcnn-confidence", type=float, help="Alias for --mask-rcnn-score")
    parser.add_argument("--cpu-only", action="store_true", help="Disable GPU acceleration and force CPU backends")
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
    default_hint = "y/n" if default else "Y/N"
    raw = input(f"{prompt} [{default_hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}



IDENTITY_COLOR_OPTIONS = ("orange", "yellow", "red", "green", "blue", "purple", "pink", "white", "black", "gray")
IDENTITY_REGION_OPTIONS = ("top", "torso", "full")


def prompt_choice(prompt: str, options: tuple[str, ...], default: str) -> str:
    option_map = {str(index): option for index, option in enumerate(options, start=1)}
    while True:
        print("=" * 50)
        for index, option in enumerate(options, start=1):
            print(f"{index}. {option}")
        print("=" * 50)
        raw = input(f"{prompt} [{default}]: ").strip().lower()
        if not raw:
            return default
        if raw in option_map:
            return option_map[raw]
        if raw in options:
            return raw
        print("Please choose one of the listed options.")


def prompt_identity_profiles(max_persons: int) -> list[IdentityProfile]:
    profiles: list[IdentityProfile] = []
    if max_persons <= 1:
        return profiles
    if not prompt_yes_no("Configure color-based identity memory for each person?", default=True):
        return profiles

    for person_index in range(max_persons):
        default_label = f"Person {person_index + 1}"
        label = input(f"Label for slot {person_index + 1} [{default_label}]: ").strip() or default_label
        color_name = prompt_choice(
            f"Which standout color best identifies {label}?",
            IDENTITY_COLOR_OPTIONS,
            default=IDENTITY_COLOR_OPTIONS[min(person_index, len(IDENTITY_COLOR_OPTIONS) - 1)],
        )
        region = prompt_choice(
            f"Where is that color easiest to see for {label}?",
            IDENTITY_REGION_OPTIONS,
            default="top",
        )
        profiles.append(IdentityProfile(slot_id=person_index + 1, label=label, color_name=color_name, region=region))
    return profiles


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
    setattr(config, "prefer_gpu", not args.cpu_only)

    if args.udp_ip:
        config.udp_ip = args.udp_ip
    if args.udp_port is not None:
        config.udp_port = args.udp_port
    if args.smoothing_alpha is not None:
        config.smoothing_alpha = args.smoothing_alpha
        config.body_smoothing_alpha = max(0.05, min(args.smoothing_alpha, 0.95))
        config.hand_smoothing_alpha = max(0.05, min(args.smoothing_alpha * 0.75, 0.95))
    if args.body_detection_confidence is not None and args.body_detection_confidence > 0:
        config.min_pose_detection_confidence = args.body_detection_confidence
    if args.body_presence_confidence is not None and args.body_presence_confidence > 0:
        config.min_pose_presence_confidence = args.body_presence_confidence
    if args.body_tracking_confidence is not None and args.body_tracking_confidence > 0:
        config.min_tracking_confidence = args.body_tracking_confidence
    if args.hand_detection_confidence is not None and args.hand_detection_confidence > 0:
        config.min_hand_detection_confidence = args.hand_detection_confidence
    if args.hand_presence_confidence is not None and args.hand_presence_confidence > 0:
        config.min_hand_presence_confidence = args.hand_presence_confidence
    if args.hand_tracking_confidence is not None and args.hand_tracking_confidence > 0:
        config.min_hand_tracking_confidence = args.hand_tracking_confidence
    if args.preview_fps is not None and args.preview_fps > 0:
        config.preview_target_fps = args.preview_fps
    config.enable_identity_memory = max_persons > 1 and not args.no_identity_memory
    config.identity_profiles = prompt_identity_profiles(max_persons) if config.enable_identity_memory else []

     # YOLO: only active when explicitly requested AND tracking >1 person
    yolo_requested = args.enable_yolo is not None   # None means flag was absent entirely
    config.enable_yolo_identity_assist = max_persons > 1 and yolo_requested
    if config.enable_yolo_identity_assist:
        config.yolo_model_name = args.enable_yolo   # already defaults to "yolov8x.pt" via const=
    if args.yolo_confidence is not None and args.yolo_confidence > 0:
        config.yolo_person_confidence = args.yolo_confidence

    # RCNN: only active when YOLO is also active AND explicitly requested
    config.enable_mask_rcnn_refinement = config.enable_yolo_identity_assist and args.enable_rcnn
    if args.mask_rcnn_score is not None and args.mask_rcnn_score > 0:
        config.mask_rcnn_score_threshold = args.mask_rcnn_score
    if args.rcnn_confidence is not None and args.rcnn_confidence > 0:
        config.mask_rcnn_score_threshold = args.rcnn_confidence
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

    config.enable_motion_export = not args.no_motion_export
    config.export_json = not args.no_json_export
    config.export_bvh = not args.no_bvh_export
    config.export_fbx = not args.no_fbx_export
    config.camera_calibration_path = args.calibration_file
    config.calibrations = load_camera_calibrations(config.camera_calibration_path)

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



def draw_runtime_overlay(frame, frame_index: int, people_count: int, runtime_fps: float, stage_label: str) -> None:
    cv2.putText(frame, f"Stage: {stage_label}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 255, 40), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {frame_index}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 255, 40), 2, cv2.LINE_AA)
    cv2.putText(frame, f"People: {people_count}", (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 255, 40), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {runtime_fps:.1f}", (12, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 255, 40), 2, cv2.LINE_AA)


def draw_identity_overlay(frame, people: list[dict]) -> None:
    frame_height, frame_width = frame.shape[:2]
    for person in people:
        bbox = person.get("_bbox")
        if bbox is None:
            continue

        identity = person.get("identity") or {}
        label = identity.get("label") or f"Person {person.get('id', '?')}"
        top_color = identity.get("top_color")
        if top_color:
            label = f"{label} | {top_color}"

        color = (0, 255, 255)
        cv2.rectangle(frame, (bbox["x0"], bbox["y0"]), (bbox["x1"], bbox["y1"]), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_x0 = max(0, min(int(bbox["x0"]), frame_width - text_width - 10))
        preferred_y1 = int(bbox["y0"]) - 6
        if preferred_y1 - text_height - baseline - 6 < 0:
            label_y0 = min(frame_height - text_height - baseline - 6, int(bbox["y0"]) + 6)
        else:
            label_y0 = preferred_y1 - text_height - baseline - 6
        label_y0 = max(0, label_y0)
        label_x1 = min(frame_width, label_x0 + text_width + 10)
        label_y1 = min(frame_height, label_y0 + text_height + baseline + 6)

        cv2.rectangle(frame, (label_x0, label_y0), (label_x1, label_y1), color, -1)
        cv2.putText(
            frame,
            label,
            (label_x0 + 5, label_y1 - baseline - 3),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

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
        enable_yolo_identity_assist=False,
        enable_mask_rcnn_refinement=False,
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
                fused_people = fusion.fuse_frame(detections_by_role, frame_index=preview_frame_index, timestamp_ms=timestamp_ms)
                latest_primary_render = rendered_by_role[assignments[0].role]

                elapsed = max(time.perf_counter() - started_at, 0.001)
                runtime_fps = (preview_frame_index + 1) / elapsed
                draw_runtime_overlay(latest_primary_render, preview_frame_index, len(fused_people), runtime_fps, "PREVIEW")
                draw_identity_overlay(latest_primary_render, fused_people)

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
    exporter = MotionExporter(
        output_dir=config.motion_export_dir,
        enabled=config.enable_motion_export,
        source_fps=session_fps,
        base_name="fused_motion",
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
    if config.camera_calibration_path:
        print(f"Using calibration file: {config.camera_calibration_path}")

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
            fused_people = fusion.fuse_frame(detections_by_role, frame_index=output_frame_index, timestamp_ms=timestamp_ms)
            primary_render = rendered_by_role[assignments[0].role]

            elapsed = max(time.perf_counter() - started_at, 0.001)
            runtime_fps = (output_frame_index + 1) / elapsed
            draw_runtime_overlay(primary_render, output_frame_index, len(fused_people), runtime_fps, "FINAL")
            draw_identity_overlay(primary_render, fused_people)

            packet = build_packet(
                persons=fused_people,
                frame_index=output_frame_index,
                timestamp_ms=timestamp_ms,
                source_fps=session_fps,
            )
            sender.send(packet)
            writer.write(primary_render)
            exporter.record_frame(
                frame_index=output_frame_index,
                timestamp_ms=timestamp_ms,
                persons=fused_people,
            )

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
        export_paths = exporter.close(export_json=config.export_json, export_bvh=config.export_bvh, export_fbx=config.export_fbx)
        sender.close()
        cv2.destroyAllWindows()
        if export_paths:
            print("Motion exports created:")
            for export_path in export_paths:
                print(f"- {export_path}")



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
        prefer_gpu = bool(getattr(cast(Any, config), "prefer_gpu", True))
        acceleration = detect_acceleration(prefer_gpu)
        print("Acceleration profile:")
        print(f"- Preferred backend: {'GPU' if prefer_gpu else 'CPU'}")
        print(f"- Active YOLO device: {acceleration.yolo_device}")
        print(f"- Active Torch device: {acceleration.torch_device}")
        if acceleration.gpu_name:
            cuda_label = acceleration.cuda_version or "detected"
            print(f"- NVIDIA GPU: {acceleration.gpu_name} | CUDA runtime: {cuda_label}")
        print(f"- cuDNN available: {'yes' if acceleration.cudnn_available else 'no'}")
        if acceleration.cudnn_version is not None:
            print(f"- cuDNN version: {acceleration.cudnn_version}")
        for note in acceleration.notes:
            print(f"- {note}")
        assignments = build_assignments(args)
        run_session(assignments, config)
    except Exception as exc:
        print(f"Pipeline failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
