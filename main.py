from __future__ import annotations

import argparse
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from camera.capture import VideoCaptureSession, VideoInputSource, VideoOutputWriter
from config import PipelineConfig
from inference.rtmpose import ONNXPoseHandRunner
from network.osc_sender import OSCSender
from pipeline.pipeline import PoseHandPipeline
from utils.exports import (
    build_joint_map,
    export_motion_fbx,
    export_motion_json,
    export_multi_person_fbx_bundle,
    export_multi_person_json,
)
from utils.fusion import estimate_joint_depths, fuse_body_views, fuse_hand_views, load_camera_calibrations
from utils.model_assets import DEFAULT_BODY_MODEL, HAND_MODEL_SPECS, ensure_body_model_file, ensure_model_file
from utils.multi_person import MultiPersonTracker, PersonTrack
from utils.smoothing import LandmarkSmoother


CAMERA_POSITION_OPTIONS = {
    "1": "BACK",
    "2": "LEFT",
    "3": "RIGHT",
}
DEFAULT_MULTI_SOURCE_LABELS = ("FRONT", "BACK", "LEFT", "RIGHT")


@dataclass(frozen=True, slots=True)
class InputAssignment:
    label: str
    source: int | Path


def parse_color(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Colors must be provided as B,G,R.")

    try:
        color_values = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Color values must be integers.") from exc

    color = (color_values[0], color_values[1], color_values[2])

    if any(channel < 0 or channel > 255 for channel in color):
        raise argparse.ArgumentTypeError("Each color channel must be between 0 and 255.")
    return color


def parse_identity_hint(value: str) -> tuple[str, tuple[str, ...]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Identity hints must look like 'person1=black,orange'.")
    label, colors_raw = value.split("=", 1)
    normalized_label = label.strip().lower()
    colors = tuple(color.strip().lower() for color in colors_raw.split(",") if color.strip())
    if not normalized_label or not colors:
        raise argparse.ArgumentTypeError("Identity hints must include a label and at least one color.")
    return normalized_label, colors


def prepare_model_assets(config: PipelineConfig) -> None:
    hand_spec = HAND_MODEL_SPECS[config.hand_model_variant]

    if config.body_model_path is None:
        config.body_model_path = config.body_model_variant or DEFAULT_BODY_MODEL
    config.body_model_path = ensure_body_model_file(config.project_root, str(config.body_model_path))

    if config.hand_model_path is None:
        print(f"Preparing hand model preset '{config.hand_model_variant}'...")
        config.hand_model_path = ensure_model_file(config.project_root, hand_spec)
        config.hand_input_size = hand_spec.input_size
        config.hand_input_name = hand_spec.input_name
        config.hand_input_dtype = hand_spec.input_dtype


def choose_video_gui(title: str = "Select Video File") -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ModuleNotFoundError:
        print("Error: tkinter is not available in this Python environment.")
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
            ("All Files", "*.*"),
        ],
    )
    root.destroy()
    return path or None


def sanitize_label(label: str) -> str:
    return label.strip().lower()


def resolve_output_basename(base_name: str | None, source: int | Path, label: str, multi_input: bool) -> str | None:
    if base_name is not None:
        cleaned = base_name.strip()
        return f"{cleaned}_{sanitize_label(label)}" if multi_input else cleaned

    if isinstance(source, Path):
        stem = source.stem
    else:
        stem = f"webcam_{source}"
    return f"{stem}_{sanitize_label(label)}" if multi_input else stem


def resolve_output_path(output_path: Path | None, label: str, multi_input: bool) -> Path | None:
    if output_path is None:
        return None
    if not multi_input:
        return output_path
    return output_path.with_name(f"{output_path.stem}_{sanitize_label(label)}{output_path.suffix or '.mp4'}")


def resolve_fused_output_basename(base_name: str | None, assignments: list[InputAssignment]) -> str | None:
    if base_name is not None:
        return f"{base_name.strip()}_fused"

    front_assignment = next((assignment for assignment in assignments if assignment.label == "FRONT"), assignments[0])
    if isinstance(front_assignment.source, Path):
        return f"{front_assignment.source.stem}_fused"
    return f"webcam_{front_assignment.source}_fused"


def resolve_fused_output_path(output_path: Path | None) -> Path | None:
    if output_path is None:
        return None
    return output_path.with_name(f"{output_path.stem}_fused{output_path.suffix or '.mp4'}")


def choose_camera_assignments_gui() -> list[InputAssignment]:
    print("How many cameras do you want to assign?")
    count_raw = input("Enter camera count [1-4]: ").strip()
    camera_count = int(count_raw) if count_raw.isdigit() else 1
    camera_count = max(1, min(4, camera_count))

    labels = ["FRONT"]
    available_positions = dict(CAMERA_POSITION_OPTIONS)

    while len(labels) < camera_count and available_positions:
        print("Assign the next camera position:")
        for option, name in available_positions.items():
            print(f"  {option}. {name}")
        selection = input("Choose position [1/2/3]: ").strip()
        selected_label = available_positions.get(selection)
        if selected_label is None:
            print("Invalid selection. Please choose 1, 2, or 3.")
            continue
        labels.append(selected_label)
        del available_positions[selection]

    assignments: list[InputAssignment] = []
    for label in labels:
        path = choose_video_gui(f"Select Video File for {label}")
        if not path:
            print(f"No file selected for {label}.")
            return []
        assignments.append(InputAssignment(label=label, source=Path(path)))

    return assignments


def resolve_sources(args: argparse.Namespace) -> list[InputAssignment]:
    # CLI: --source 0 or --source path/to/video.mp4 or repeated --source FRONT=path --source LEFT=path
    if args.source is not None:
        raw_sources = args.source if isinstance(args.source, list) else [args.source]
        assignments: list[InputAssignment] = []
        used_labels: set[str] = set()
        auto_labels = iter(DEFAULT_MULTI_SOURCE_LABELS)

        for raw_source in raw_sources:
            source_text = raw_source.strip()
            label: str | None = None
            value_text = source_text
            if "=" in source_text:
                raw_label, raw_value = source_text.split("=", 1)
                label = sanitize_label(raw_label).upper()
                value_text = raw_value.strip()

            if label is None:
                try:
                    label = next(auto_labels)
                except StopIteration:
                    print("Error: too many unlabeled sources. Use LABEL=path for additional inputs.")
                    return []

            if label in used_labels:
                print(f"Error: duplicate source label: {label}")
                return []
            used_labels.add(label)

            if value_text.isdigit():
                assignments.append(InputAssignment(label=label, source=int(value_text)))
                continue

            path = Path(value_text)
            if not path.exists():
                print(f"Error: file not found: {path}")
                return []
            assignments.append(InputAssignment(label=label, source=path))

        return assignments

    # Interactive prompt
    print("Select input source:")
    print("  1. Webcam")
    print("  2. Video file(s)")
    choice = input("Enter choice [1/2]: ").strip()

    if choice == "1":
        idx = input("Webcam index: ").strip()
        return [InputAssignment(label="FRONT", source=int(idx) if idx.isdigit() else 0)]

    if choice == "2":
        return choose_camera_assignments_gui()
    
    print("Invalid choice.")
    return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pose and Hand Landmark Pipeline")
    parser.add_argument(
        "--source",
        action="append",
        help="Webcam index (e.g. 0) or path to a video file. If omitted, an interactive prompt runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output video path. The final file will still be stacked as '<name> rendered-N.ext'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where rendered/video sibling outputs should be written.",
    )
    parser.add_argument(
        "--output-basename",
        help="Base filename prefix used for rendered/fbx/json sibling outputs.",
    )
    parser.add_argument(
        "--model",
        help="Body model filename or path, for example yolo11x-pose.pt or a local YOLO pose weights path.",
    )
    parser.add_argument(
        "--hand-model-variant",
        choices=("low", "mid", "high", "max"),
        default="max",
        help="Hand model preset. Supported variants: low, mid, high, max.",
    )
    parser.add_argument(
        "--hand-model",
        type=Path,
        help="Path to the ONNX hand model. Overrides the hand preset download.",
    )
    parser.add_argument(
        "--body-input-name",
        default="input",
        help="Reserved body input name setting. Kept in config for compatibility.",
    )
    parser.add_argument(
        "--hand-input-name",
        default="images",
        help="Input tensor name for the hand model.",
    )
    parser.add_argument(
        "--body-input-size",
        type=int,
        default=960,
        help="YOLO body model image size.",
    )
    parser.add_argument(
        "--hand-input-size",
        type=int,
        default=640,
        help="Square input size for the hand model crop.",
    )
    parser.add_argument(
        "--body-conf-threshold",
        type=float,
        default=0.30,
        help="Minimum confidence used to keep body landmarks.",
    )
    parser.add_argument(
        "--hand-det-threshold",
        type=float,
        default=0.15,
        help="Minimum hand detection score.",
    )
    parser.add_argument(
        "--hand-kp-threshold",
        type=float,
        default=0.20,
        help="Minimum hand keypoint confidence for drawing and live UDP output.",
    )
    parser.add_argument(
        "--hand-box-min-size",
        type=int,
        default=160,
        help="Minimum side length for the wrist-centered hand crop.",
    )
    parser.add_argument(
        "--hand-box-scale",
        type=float,
        default=2.0,
        help="Scale factor applied to the wrist-elbow based hand crop.",
    )
    parser.add_argument(
        "--body-iou-threshold",
        type=float,
        default=0.45,
        help="YOLO body NMS IoU threshold.",
    )
    parser.add_argument(
        "--max-people",
        type=int,
        default=1,
        help="Maximum number of people to detect and track in a single view.",
    )
    parser.add_argument(
        "--identity",
        dest="identity_hints",
        action="append",
        type=parse_identity_hint,
        help="Optional clothing color hint like --identity person1=black,orange.",
    )
    parser.add_argument(
        "--person-detector-scale",
        type=float,
        default=1.05,
        help="Deprecated compatibility setting. Kept in config but unused by the YOLO multi-person path.",
    )
    parser.add_argument(
        "--person-box-scale",
        type=float,
        default=1.15,
        help="Expand each detected person box before running pose and hand inference.",
    )
    parser.add_argument(
        "--person-track-hold-frames",
        type=int,
        default=10,
        help="How many frames to keep a person track alive when detections are briefly missing.",
    )
    parser.add_argument(
        "--person-match-threshold",
        type=float,
        default=0.15,
        help="Minimum association score when matching a detected person to an existing track.",
    )
    parser.add_argument(
        "--person-cross-wrist-ratio",
        type=float,
        default=0.90,
        help="Hand ownership switch ratio. Lower values are stricter during crossings.",
    )
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        help="Optional JSON file with per-camera fusion calibration overrides.",
    )
    parser.add_argument(
        "--fused-depth-scale",
        type=float,
        default=1.0,
        help="Depth scale multiplier used when estimating fused multi-camera joint depth.",
    )
    parser.add_argument(
        "--yolo-tracker",
        default="botsort.yaml",
        help="Ultralytics tracker config name for multi-person tracking.",
    )
    parser.add_argument(
        "--yolo-device",
        help="Optional Ultralytics device override such as 0, cpu, or cuda:0.",
    )
    parser.add_argument(
        "--provider",
        dest="providers",
        action="append",
        help="ONNX Runtime provider priority for the hand model, e.g. --provider CUDAExecutionProvider --provider CPUExecutionProvider.",
    )
    parser.add_argument(
        "--osc-host",
        default="127.0.0.1",
        help="Live UDP target host.",
    )
    parser.add_argument(
        "--osc-port",
        type=int,
        default=9000,
        help="Live UDP target port.",
    )
    parser.add_argument(
        "--osc-enabled",
        action="store_true",
        help="Enable live UDP sending.",
    )
    parser.add_argument(
        "--preview-title",
        default="Pose + Hand Landmarks",
        help="Window title for the live preview.",
    )
    parser.add_argument(
        "--fallback-fps",
        type=float,
        default=30.0,
        help="FPS to use when the source does not report one.",
    )
    parser.add_argument(
        "--output-fourcc",
        default="mp4v",
        help="FourCC codec for the output video writer.",
    )
    parser.add_argument(
        "--body-line-color",
        type=parse_color,
        default=parse_color("255,0,0"),
        help="Body line color as B,G,R.",
    )
    parser.add_argument(
        "--body-point-color",
        type=parse_color,
        default=parse_color("0,255,0"),
        help="Body landmark color as B,G,R.",
    )
    parser.add_argument(
        "--hand-box-color",
        type=parse_color,
        default=parse_color("80,80,255"),
        help="Hand box color as B,G,R.",
    )
    parser.add_argument(
        "--hand-line-color",
        type=parse_color,
        default=parse_color("0,255,255"),
        help="Hand skeleton color as B,G,R.",
    )
    parser.add_argument(
        "--hand-point-color",
        type=parse_color,
        default=parse_color("0,165,255"),
        help="Hand keypoint color as B,G,R.",
    )
    parser.add_argument(
        "--body-line-thickness",
        type=int,
        default=2,
        help="Thickness of body skeleton lines.",
    )
    parser.add_argument(
        "--body-point-radius",
        type=int,
        default=4,
        help="Radius of body landmark points.",
    )
    parser.add_argument(
        "--hand-box-thickness",
        type=int,
        default=1,
        help="Thickness of the hand crop box.",
    )
    parser.add_argument(
        "--hand-line-thickness",
        type=int,
        default=2,
        help="Thickness of hand skeleton lines.",
    )
    parser.add_argument(
        "--hand-point-radius",
        type=int,
        default=3,
        help="Radius of hand landmark points.",
    )
    parser.add_argument(
        "--body-smoothing-alpha",
        type=float,
        default=0.65,
        help="EMA smoothing factor for body landmarks. Higher follows new detections more closely.",
    )
    parser.add_argument(
        "--hand-smoothing-alpha",
        type=float,
        default=0.55,
        help="EMA smoothing factor for hand landmarks. Higher follows new detections more closely.",
    )
    parser.add_argument(
        "--body-hold-frames",
        type=int,
        default=8,
        help="How many frames to keep the last valid body landmark before dropping it.",
    )
    parser.add_argument(
        "--hand-hold-frames",
        type=int,
        default=6,
        help="How many frames to keep the last valid hand landmark before dropping it.",
    )
    parser.add_argument(
        "--hold-confidence-decay",
        type=float,
        default=0.85,
        help="Confidence multiplier applied each frame while reusing a held landmark.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the live OpenCV preview window.",
    )
    return parser


def validate_config(config: PipelineConfig) -> bool:
    missing_paths = [
        path for path in (config.hand_model_path,)
        if path is not None and not Path(path).exists()
    ]
    if missing_paths:
        for path in missing_paths:
            print(f"Error: model file not found: {path}")
        return False
    if config.osc_port < 1 or config.osc_port > 65535:
        print(f"Error: invalid OSC port: {config.osc_port}")
        return False
    if config.fallback_fps <= 0:
        print(f"Error: fallback FPS must be positive: {config.fallback_fps}")
        return False
    if len(config.output_fourcc) < 4:
        print(f"Error: output FourCC must have at least 4 characters: {config.output_fourcc}")
        return False
    if config.output_basename is not None and not config.output_basename.strip():
        print("Error: output basename must not be empty.")
        return False
    bounded_float_fields = {
        "body_smoothing_alpha": config.body_smoothing_alpha,
        "hand_smoothing_alpha": config.hand_smoothing_alpha,
        "hold_confidence_decay": config.hold_confidence_decay,
        "body_conf_threshold": config.body_conf_threshold,
        "body_iou_threshold": config.body_iou_threshold,
        "hand_det_threshold": config.hand_det_threshold,
        "hand_kp_threshold": config.hand_kp_threshold,
        "identity_min_score": config.identity_min_score,
        "person_cross_wrist_ratio": config.person_cross_wrist_ratio,
    }
    for field_name, value in bounded_float_fields.items():
        if value <= 0 or value > 1:
            print(f"Error: {field_name} must be in the range (0, 1]: {value}")
            return False
    positive_int_fields = {
        "body_input_size": config.body_input_size,
        "hand_input_size": config.hand_input_size,
        "hand_box_min_size": config.hand_box_min_size,
        "body_line_thickness": config.body_line_thickness,
        "body_point_radius": config.body_point_radius,
        "hand_box_thickness": config.hand_box_thickness,
        "hand_line_thickness": config.hand_line_thickness,
        "hand_point_radius": config.hand_point_radius,
        "body_hold_frames": config.body_hold_frames,
        "hand_hold_frames": config.hand_hold_frames,
        "max_people": config.max_people,
        "person_track_hold_frames": config.person_track_hold_frames,
    }
    for field_name, value in positive_int_fields.items():
        if value <= 0:
            print(f"Error: {field_name} must be positive: {value}")
            return False
    if config.person_detector_scale <= 1.0:
        print(f"Error: person_detector_scale must be greater than 1.0: {config.person_detector_scale}")
        return False
    if config.person_box_scale <= 0:
        print(f"Error: person_box_scale must be positive: {config.person_box_scale}")
        return False
    if config.person_match_threshold <= 0:
        print(f"Error: person_match_threshold must be positive: {config.person_match_threshold}")
        return False
    if config.fused_depth_scale <= 0:
        print(f"Error: fused_depth_scale must be positive: {config.fused_depth_scale}")
        return False
    return True


def export_motion_bundle(
    config: PipelineConfig,
    fps: float,
    frames: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    if not frames:
        return
    export_motion_json(config.json_output_path, fps, frames, metadata)
    export_motion_fbx(config.fbx_output_path, fps, frames)


def build_config_for_assignment(args: argparse.Namespace, assignment: InputAssignment, multi_input: bool) -> PipelineConfig:
    config = PipelineConfig(
        video_path=assignment.source,
        output_path=resolve_output_path(args.output, assignment.label, multi_input),
        output_directory=args.output_dir,
        output_basename=resolve_output_basename(args.output_basename, assignment.source, assignment.label, multi_input),
        body_model_path=args.model,
        hand_model_path=args.hand_model,
        body_model_variant=args.model or DEFAULT_BODY_MODEL,
        hand_model_variant=args.hand_model_variant,
        body_input_name=args.body_input_name,
        body_input_dtype="int32",
        hand_input_name=args.hand_input_name,
        hand_input_dtype="float32",
        body_input_size=args.body_input_size,
        hand_input_size=args.hand_input_size,
        body_conf_threshold=args.body_conf_threshold,
        body_iou_threshold=args.body_iou_threshold,
        hand_det_threshold=args.hand_det_threshold,
        hand_kp_threshold=args.hand_kp_threshold,
        hand_box_min_size=args.hand_box_min_size,
        hand_box_scale=args.hand_box_scale,
        enable_preview=not args.no_preview,
        provider_names=tuple(args.providers) if args.providers else ("CUDAExecutionProvider",),
        preview_window_title=f"{args.preview_title} - {assignment.label}" if multi_input else args.preview_title,
        osc_host=args.osc_host,
        osc_port=args.osc_port,
        osc_enabled=args.osc_enabled,
        fallback_fps=args.fallback_fps,
        output_fourcc=args.output_fourcc,
        body_line_color=args.body_line_color,
        body_point_color=args.body_point_color,
        hand_box_color=args.hand_box_color,
        hand_line_color=args.hand_line_color,
        hand_point_color=args.hand_point_color,
        body_line_thickness=args.body_line_thickness,
        body_point_radius=args.body_point_radius,
        hand_box_thickness=args.hand_box_thickness,
        hand_line_thickness=args.hand_line_thickness,
        hand_point_radius=args.hand_point_radius,
        body_smoothing_alpha=args.body_smoothing_alpha,
        hand_smoothing_alpha=args.hand_smoothing_alpha,
        body_hold_frames=args.body_hold_frames,
        hand_hold_frames=args.hand_hold_frames,
        hold_confidence_decay=args.hold_confidence_decay,
        max_people=args.max_people,
        person_detector_scale=args.person_detector_scale,
        person_box_scale=args.person_box_scale,
        person_track_hold_frames=args.person_track_hold_frames,
        person_match_threshold=args.person_match_threshold,
        person_cross_wrist_ratio=args.person_cross_wrist_ratio,
        camera_calibration_path=args.camera_calibration,
        fused_depth_scale=args.fused_depth_scale,
        yolo_tracker=args.yolo_tracker,
        yolo_device=args.yolo_device,
        identity_hints=dict(args.identity_hints or []),
    )
    return config


def build_fused_config(args: argparse.Namespace, assignments: list[InputAssignment]) -> PipelineConfig:
    reference_assignment = next((assignment for assignment in assignments if assignment.label == "FRONT"), assignments[0])
    return PipelineConfig(
        video_path=reference_assignment.source,
        output_path=resolve_fused_output_path(args.output),
        output_directory=args.output_dir,
        output_basename=resolve_fused_output_basename(args.output_basename, assignments),
        body_model_path=args.model,
        hand_model_path=args.hand_model,
        body_model_variant=args.model or DEFAULT_BODY_MODEL,
        hand_model_variant=args.hand_model_variant,
        body_input_name=args.body_input_name,
        body_input_dtype="int32",
        hand_input_name=args.hand_input_name,
        hand_input_dtype="float32",
        body_input_size=args.body_input_size,
        hand_input_size=args.hand_input_size,
        body_conf_threshold=args.body_conf_threshold,
        body_iou_threshold=args.body_iou_threshold,
        hand_det_threshold=args.hand_det_threshold,
        hand_kp_threshold=args.hand_kp_threshold,
        hand_box_min_size=args.hand_box_min_size,
        hand_box_scale=args.hand_box_scale,
        enable_preview=not args.no_preview,
        provider_names=tuple(args.providers) if args.providers else ("CUDAExecutionProvider",),
        preview_window_title=f"{args.preview_title} - FUSED",
        osc_host=args.osc_host,
        osc_port=args.osc_port,
        osc_enabled=args.osc_enabled,
        fallback_fps=args.fallback_fps,
        output_fourcc=args.output_fourcc,
        body_line_color=args.body_line_color,
        body_point_color=args.body_point_color,
        hand_box_color=args.hand_box_color,
        hand_line_color=args.hand_line_color,
        hand_point_color=args.hand_point_color,
        body_line_thickness=args.body_line_thickness,
        body_point_radius=args.body_point_radius,
        hand_box_thickness=args.hand_box_thickness,
        hand_line_thickness=args.hand_line_thickness,
        hand_point_radius=args.hand_point_radius,
        body_smoothing_alpha=args.body_smoothing_alpha,
        hand_smoothing_alpha=args.hand_smoothing_alpha,
        body_hold_frames=args.body_hold_frames,
        hand_hold_frames=args.hand_hold_frames,
        hold_confidence_decay=args.hold_confidence_decay,
        max_people=args.max_people,
        person_detector_scale=args.person_detector_scale,
        person_box_scale=args.person_box_scale,
        person_track_hold_frames=args.person_track_hold_frames,
        person_match_threshold=args.person_match_threshold,
        person_cross_wrist_ratio=args.person_cross_wrist_ratio,
        camera_calibration_path=args.camera_calibration,
        fused_depth_scale=args.fused_depth_scale,
        yolo_tracker=args.yolo_tracker,
        yolo_device=args.yolo_device,
        identity_hints=dict(args.identity_hints or []),
    )


def _color_similarity(profile_a: dict[str, float], profile_b: dict[str, float]) -> float:
    if not profile_a or not profile_b:
        return 0.0
    keys = set(profile_a) | set(profile_b)
    overlap = 0.0
    magnitude = 0.0
    for key in keys:
        overlap += min(profile_a.get(key, 0.0), profile_b.get(key, 0.0))
        magnitude += max(profile_a.get(key, 0.0), profile_b.get(key, 0.0))
    if magnitude <= 0.0:
        return 0.0
    return overlap / magnitude


def _track_sort_key(track: PersonTrack) -> tuple[int, float]:
    return (0 if track.label else 1, track.center[0])


def _person_key(track: PersonTrack, fallback_index: int) -> str:
    if track.label:
        return track.label
    if track.id > 0:
        return f"person{track.id}"
    return f"person{fallback_index + 1}"


def _align_people_across_cameras(
    camera_tracks: dict[str, list[PersonTrack]],
    reference_label: str,
) -> dict[str, dict[str, PersonTrack]]:
    grouped: dict[str, dict[str, PersonTrack]] = {}
    reference_tracks = sorted(camera_tracks.get(reference_label, []), key=_track_sort_key)
    reference_keys: list[str] = []

    for index, track in enumerate(reference_tracks):
        key = _person_key(track, index)
        reference_keys.append(key)
        grouped.setdefault(key, {})[reference_label] = track

    for camera_label, tracks in camera_tracks.items():
        if camera_label == reference_label:
            continue

        remaining_tracks = sorted(tracks, key=_track_sort_key)
        assigned_keys: set[str] = set()

        for track in list(remaining_tracks):
            if track.label and track.label in grouped:
                grouped.setdefault(track.label, {})[camera_label] = track
                assigned_keys.add(track.label)
                remaining_tracks.remove(track)

        open_reference_keys = [key for key in reference_keys if key not in assigned_keys]
        still_unmatched = list(remaining_tracks)
        while still_unmatched and open_reference_keys:
            scored_pairs: list[tuple[float, int, int]] = []
            for track_index, track in enumerate(still_unmatched):
                for key_index, key in enumerate(open_reference_keys):
                    reference_track = grouped.get(key, {}).get(reference_label)
                    if reference_track is None:
                        continue
                    score = _color_similarity(reference_track.color_signature, track.color_signature)
                    scored_pairs.append((score, track_index, key_index))
            if not scored_pairs:
                break
            _, track_index, key_index = max(scored_pairs)
            key = open_reference_keys.pop(key_index)
            track = still_unmatched.pop(track_index)
            grouped.setdefault(key, {})[camera_label] = track
            assigned_keys.add(key)

        for key, track in zip(open_reference_keys, still_unmatched):
            grouped.setdefault(key, {})[camera_label] = track
            assigned_keys.add(key)

        extra_index = 0
        for track in still_unmatched[len(open_reference_keys):]:
            while f"person{len(reference_keys) + extra_index + 1}" in grouped:
                extra_index += 1
            key = f"person{len(reference_keys) + extra_index + 1}"
            grouped.setdefault(key, {})[camera_label] = track
            extra_index += 1

    return grouped


def _build_person_payload(
    person_id: int,
    label: str,
    body_points: list[tuple[int, int, float]],
    hands_by_side: dict[str, dict],
    joint_depths: dict[str, float] | None = None,
    box: tuple[int, int, int, int] | None = None,
    score: float | None = None,
    camera_views: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": person_id,
        "label": label,
        "box": box,
        "score": score,
        "camera_views": camera_views or [],
        "body_points": body_points,
        "hands_by_side": hands_by_side,
        "joints": build_joint_map(body_points, hands_by_side, joint_depths=joint_depths),
    }


def _draw_person_overlay(frame, track_label: str, box: tuple[int, int, int, int], score: float | None = None) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
    label_text = track_label if score is None else f"{track_label} {score:.2f}"
    cv2.putText(frame, label_text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)


def _box_from_body_points(points: list[tuple[int, int, float]], threshold: float) -> tuple[int, int, int, int] | None:
    confident_points = [(x, y) for x, y, conf in points if conf > threshold]
    if not confident_points:
        return None
    xs = [point[0] for point in confident_points]
    ys = [point[1] for point in confident_points]
    return min(xs), min(ys), max(xs), max(ys)


def run_multi_person_assignment(config: PipelineConfig) -> None:
    try:
        prepare_model_assets(config)
    except Exception as exc:
        print(f"Error: failed to prepare model assets: {exc}")
        return
    if not validate_config(config):
        return

    assert config.output_path is not None
    session = VideoCaptureSession(
        config.video_path,
        config.output_path,
        fallback_fps=config.fallback_fps,
        output_fourcc=config.output_fourcc,
    )
    runner = ONNXPoseHandRunner(config)
    osc_sender = OSCSender(config.osc_host, config.osc_port, config.osc_enabled)
    tracker = MultiPersonTracker(config, runner)
    motion_frames: list[dict[str, object]] = []
    frame_index = 0

    try:
        while True:
            ok, frame = session.read()
            if not ok or frame is None:
                break

            people = tracker.update(frame)
            payload_people: list[dict[str, object]] = []
            for track in people:
                track.pipeline.render_pose(frame, track.body_points, track.hands_by_side, send_osc=False)
                label = track.label or f"person{track.id}"
                _draw_person_overlay(frame, label, track.box, track.detection_score)
                payload_people.append(
                    _build_person_payload(
                        person_id=track.id,
                        label=label,
                        box=track.box,
                        score=track.detection_score,
                        body_points=track.body_points,
                        hands_by_side=track.hands_by_side,
                        camera_views=["FRONT"],
                    )
                )

            osc_sender.send_people(
                payload_people,
                metadata={
                    "frame_index": frame_index,
                    "mode": "multi_person",
                    "source": str(config.video_path),
                },
            )
            motion_frames.append({"frame_index": frame_index, "people": payload_people})
            frame_index += 1
            session.write(frame)

            if config.enable_preview:
                cv2.imshow(config.preview_window_title, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        session.close()
        osc_sender.close()
        cv2.destroyAllWindows()

    export_multi_person_json(
        config.json_output_path,
        fps=session.fps,
        frames=motion_frames,
        metadata={
            "mode": "multi_person",
            "source": str(config.video_path),
            "max_people": config.max_people,
            "identity_hints": {key: list(value) for key, value in config.identity_hints.items()},
        },
    )
    exported_fbx_paths = export_multi_person_fbx_bundle(config.fbx_output_path, session.fps, motion_frames)
    print(f"Saved: {config.output_path}")
    print(f"Saved: {config.json_output_path}")
    for exported_path in exported_fbx_paths:
        print(f"Saved: {exported_path}")


def run_assignment(config: PipelineConfig) -> None:
    if config.max_people > 1:
        run_multi_person_assignment(config)
        return
    try:
        prepare_model_assets(config)
    except Exception as exc:
        print(f"Error: failed to prepare model assets: {exc}")
        return
    if not validate_config(config):
        return

    # ------------------------------------------------------------------ #
    # Build pipeline components                                           #
    # ------------------------------------------------------------------ #
    assert config.output_path is not None
    session = VideoCaptureSession(
        config.video_path,
        config.output_path,
        fallback_fps=config.fallback_fps,
        output_fourcc=config.output_fourcc,
    )
    runner = ONNXPoseHandRunner(config)
    smoother = LandmarkSmoother(config)
    osc_sender = OSCSender(config.osc_host, config.osc_port, config.osc_enabled)
    pipeline = PoseHandPipeline(config, runner, smoother, osc_sender)
    motion_frames: list[dict[str, object]] = []
    frame_index = 0

    # ------------------------------------------------------------------ #
    # Run                                                                 #
    # ------------------------------------------------------------------ #
    try:
        while True:
            ok, frame = session.read()
            if not ok or frame is None:
                break

            body_points, hands_by_side = pipeline.detect_pose(frame)
            joints = build_joint_map(body_points, hands_by_side)
            pipeline.render_pose(frame, body_points, hands_by_side, send_osc=False)
            osc_sender.send_pose(
                body_points,
                hands_by_side,
                joints=joints,
                metadata={
                    "frame_index": frame_index,
                    "mode": "single",
                    "source": str(config.video_path),
                },
            )
            motion_frames.append(
                {
                    "frame_index": frame_index,
                    "joints": joints,
                }
            )
            frame_index += 1
            rendered = frame
            session.write(rendered)

            if config.enable_preview:
                cv2.imshow(config.preview_window_title, rendered)
                if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
                    break
    finally:
        session.close()
        osc_sender.close()
        cv2.destroyAllWindows()

    export_motion_bundle(
        config,
        fps=session.fps,
        frames=motion_frames,
        metadata={
            "mode": "single",
            "source": str(config.video_path),
            "body_model_variant": config.body_model_variant,
            "hand_model_variant": config.hand_model_variant,
        },
    )
    print(f"Saved: {config.output_path}")
    print(f"Saved: {config.json_output_path}")
    print(f"Saved: {config.fbx_output_path}")


def run_fused_assignments(
    assignments: list[InputAssignment],
    args: argparse.Namespace,
) -> None:
    config = build_fused_config(args, assignments)
    try:
        prepare_model_assets(config)
    except Exception as exc:
        print(f"Error: failed to prepare model assets: {exc}")
        return
    if not validate_config(config):
        return

    try:
        calibrations = load_camera_calibrations(config.camera_calibration_path)
    except Exception as exc:
        print(f"Error: failed to load camera calibration: {exc}")
        return

    sources: dict[str, VideoInputSource] = {}
    osc_sender = OSCSender(config.osc_host, config.osc_port, config.osc_enabled)
    writer: VideoOutputWriter | None = None
    finished = False
    motion_frames: list[dict[str, object]] = []
    frame_index = 0
    export_fps = config.fallback_fps
    try:
        for assignment in assignments:
            sources[assignment.label] = VideoInputSource(assignment.source, fallback_fps=config.fallback_fps)

        reference_label = "FRONT" if "FRONT" in sources else next(iter(sources))
        reference_source = sources[reference_label]
        export_fps = reference_source.fps
        assert config.output_path is not None
        writer = VideoOutputWriter(
            config.output_path,
            frame_width=reference_source.frame_width,
            frame_height=reference_source.frame_height,
            fps=reference_source.fps,
            output_fourcc=config.output_fourcc,
        )

        runner = ONNXPoseHandRunner(config)
        single_view_pipelines = {
            label: PoseHandPipeline(config, runner, LandmarkSmoother(config), OSCSender(enabled=False))
            for label in sources
        }
        multi_person_trackers = {
            label: MultiPersonTracker(config, runner)
            for label in sources
        }
        fused_renderers: dict[str, PoseHandPipeline] = {}

        while True:
            frames_by_label: dict[str, Any] = {}
            for label, source in sources.items():
                ok, frame = source.read()
                if not ok or frame is None:
                    finished = True
                    break
                frames_by_label[label] = frame
            if finished:
                break

            canvas = frames_by_label[reference_label].copy()

            if config.max_people > 1:
                camera_tracks = {
                    label: multi_person_trackers[label].update(frame)
                    for label, frame in frames_by_label.items()
                }
                grouped_people = _align_people_across_cameras(camera_tracks, reference_label)
                payload_people: list[dict[str, object]] = []

                for person_index, (person_key, views) in enumerate(grouped_people.items(), start=1):
                    camera_bodies = {
                        label: track.body_points
                        for label, track in views.items()
                        if track.body_points
                    }
                    camera_hands = {
                        label: track.hands_by_side
                        for label, track in views.items()
                    }
                    if not camera_bodies:
                        continue

                    renderer = fused_renderers.setdefault(
                        person_key,
                        PoseHandPipeline(config, runner, LandmarkSmoother(config), OSCSender(enabled=False)),
                    )
                    fused_body = fuse_body_views(camera_bodies, config.body_conf_threshold, reference_label=reference_label)
                    if fused_body is not None:
                        fused_body = renderer.smoother.smooth_body(fused_body)
                    if fused_body is None:
                        continue

                    fused_hands: dict[str, dict] = {}
                    for side in ("left", "right"):
                        side_views = {
                            label: hands_by_side[side]
                            for label, hands_by_side in camera_hands.items()
                            if side in hands_by_side
                        }
                        fused_hand = fuse_hand_views(side_views, config.hand_kp_threshold, reference_label=reference_label)
                        if fused_hand is None:
                            continue
                        smoothed_points = renderer.smoother.smooth_hand(side, fused_hand["points"])
                        if smoothed_points is None:
                            continue
                        fused_hands[side] = {"box": fused_hand["box"], "points": smoothed_points}

                    renderer.render_pose(canvas, fused_body, fused_hands, send_osc=False)
                    label = next((track.label for track in views.values() if track.label), person_key)
                    box = _box_from_body_points(fused_body, config.body_conf_threshold)
                    if box is not None:
                        best_score = max(track.detection_score for track in views.values())
                        _draw_person_overlay(canvas, label, box, best_score)
                    joint_depths = estimate_joint_depths(
                        camera_bodies=camera_bodies,
                        camera_hands=camera_hands,
                        body_threshold=config.body_conf_threshold,
                        hand_threshold=config.hand_kp_threshold,
                        calibrations=calibrations,
                        depth_scale=config.fused_depth_scale,
                    )
                    payload_people.append(
                        _build_person_payload(
                            person_id=person_index,
                            label=label,
                            box=box,
                            score=max(track.detection_score for track in views.values()),
                            body_points=fused_body,
                            hands_by_side=fused_hands,
                            joint_depths=joint_depths,
                            camera_views=sorted(views),
                        )
                    )

                osc_sender.send_people(
                    payload_people,
                    metadata={
                        "frame_index": frame_index,
                        "mode": "fused_multi_person",
                        "camera_labels": list(frames_by_label),
                    },
                )
                motion_frames.append({"frame_index": frame_index, "people": payload_people})
            else:
                camera_bodies: dict[str, list[tuple[int, int, float]]] = {}
                camera_hands: dict[str, dict[str, dict]] = {}
                for label, frame in frames_by_label.items():
                    body_points, hands_by_side = single_view_pipelines[label].detect_pose(frame)
                    camera_bodies[label] = body_points
                    camera_hands[label] = hands_by_side

                renderer = fused_renderers.setdefault(
                    "single",
                    PoseHandPipeline(config, runner, LandmarkSmoother(config), OSCSender(enabled=False)),
                )
                fused_body = fuse_body_views(camera_bodies, config.body_conf_threshold, reference_label=reference_label)
                if fused_body is not None:
                    fused_body = renderer.smoother.smooth_body(fused_body)
                if fused_body is None:
                    fused_body = [(0, 0, 0.0) for _ in range(17)]

                fused_hands: dict[str, dict] = {}
                for side in ("left", "right"):
                    side_views = {
                        label: hands_by_side[side]
                        for label, hands_by_side in camera_hands.items()
                        if side in hands_by_side
                    }
                    fused_hand = fuse_hand_views(side_views, config.hand_kp_threshold, reference_label=reference_label)
                    if fused_hand is None:
                        continue
                    smoothed_points = renderer.smoother.smooth_hand(side, fused_hand["points"])
                    if smoothed_points is None:
                        continue
                    fused_hands[side] = {"box": fused_hand["box"], "points": smoothed_points}

                joint_depths = estimate_joint_depths(
                    camera_bodies=camera_bodies,
                    camera_hands=camera_hands,
                    body_threshold=config.body_conf_threshold,
                    hand_threshold=config.hand_kp_threshold,
                    calibrations=calibrations,
                    depth_scale=config.fused_depth_scale,
                )
                joints = build_joint_map(fused_body, fused_hands, joint_depths=joint_depths)
                renderer.render_pose(canvas, fused_body, fused_hands, send_osc=False)
                osc_sender.send_pose(
                    fused_body,
                    fused_hands,
                    joints=joints,
                    metadata={
                        "frame_index": frame_index,
                        "mode": "fused",
                        "camera_labels": list(frames_by_label),
                    },
                )
                motion_frames.append(
                    {
                        "frame_index": frame_index,
                        "joints": joints,
                    }
                )

            frame_index += 1
            writer.write(canvas)

            if config.enable_preview:
                cv2.imshow(config.preview_window_title, canvas)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        for source in sources.values():
            source.close()
        if writer is not None:
            writer.close()
        osc_sender.close()
        cv2.destroyAllWindows()

    if config.max_people > 1:
        export_multi_person_json(
            config.json_output_path,
            fps=export_fps,
            frames=motion_frames,
            metadata={
                "mode": "fused_multi_person",
                "camera_labels": [assignment.label for assignment in assignments],
                "sources": {assignment.label: str(assignment.source) for assignment in assignments},
                "body_model_variant": config.body_model_variant,
                "hand_model_variant": config.hand_model_variant,
                "camera_calibration_path": None if config.camera_calibration_path is None else str(config.camera_calibration_path),
            },
        )
        exported_fbx_paths = export_multi_person_fbx_bundle(config.fbx_output_path, export_fps, motion_frames)
        print(f"Saved: {config.output_path}")
        print(f"Saved: {config.json_output_path}")
        for exported_path in exported_fbx_paths:
            print(f"Saved: {exported_path}")
        return

    export_motion_bundle(
        config,
        fps=export_fps,
        frames=motion_frames,
        metadata={
            "mode": "fused",
            "camera_labels": [assignment.label for assignment in assignments],
            "sources": {assignment.label: str(assignment.source) for assignment in assignments},
            "body_model_variant": config.body_model_variant,
            "hand_model_variant": config.hand_model_variant,
            "camera_calibration_path": None if config.camera_calibration_path is None else str(config.camera_calibration_path),
        },
    )
    print(f"Saved: {config.output_path}")
    print(f"Saved: {config.json_output_path}")
    print(f"Saved: {config.fbx_output_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    assignments = resolve_sources(args)
    if not assignments:
        return

    if len(assignments) > 1:
        print("Running synchronized multi-camera fusion...")
        run_fused_assignments(assignments, args)
        return

    for assignment in assignments:
        print(f"Running pipeline for {assignment.label}...")
        config = build_config_for_assignment(args, assignment, False)
        run_assignment(config)


if __name__ == "__main__":
    main()
