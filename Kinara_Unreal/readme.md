# HMT3A Unreal Client Pipeline

Client-side motion pipeline that prepares fused tracking data for downstream animation systems.

---

## What is Implemented

- Webcam and video-file capture
- Multi-camera role assignment with optional calibration JSON
- Body and hand landmark detection
- Wrist-guided hand ROI processing
- Calibration-aware multi-camera fusion into a shared space
- Persistent multi-person IDs across frames
- Configurable color-based identity memory for multi-person sessions
- Final-render-only YOLO person assist
- Optional final-render-only Mask R-CNN overlap refinement
- Adaptive smoothing with stronger hand jitter reduction
- Hierarchical skeleton reconstruction for body and hands
- Joint rotation solving with quaternion and Euler output
- Joint-angle extraction for major limbs and finger curls
- Preview pass plus final offline render pass
- UDP packet streaming
- Reusable motion export to JSON, BVH, and FBX

---

## Preview vs Final Render

**Preview pass** is intentionally lightweight — MediaPipe pose and hand tracking only, no YOLO, no Mask R-CNN. Meant for fast checking and quick rejection of bad takes.

**Final render** can use the heavier assist stack:

- YOLO person detection with persistent detector-side track IDs
- Optional Mask R-CNN person masks to reduce cross-person landmark mixing
- Configurable color memory such as `Person 1 = orange top`
- Identity timestamps including `seen_since` and `last_seen`
- CLI-adjustable confidence values for body, hand, YOLO, and Mask R-CNN

---

## Run

```bash
python main.py
```

All options including source, FPS cap, resolution, export formats, confidence thresholds, YOLO model, identity memory, and calibration file can be set via CLI arguments. See **ARGS.md** for the full reference.

---

## Dependencies

Base pipeline:

```bash
pip install mediapipe opencv-python numpy
```

Full pipeline with final-render identity assist:

```bash
pip install ultralytics torch torchvision
```

---

## Identity Memory Flow

When tracking multiple people, the console prompts for identity hints:

- Label for each person slot
- Standout color to look for
- Where that color is easiest to see: `top`, `torso`, or `full`

```txt
Person 1 -> orange -> top
Person 2 -> blue   -> torso
```

These cues combine with detector tracking and temporal motion tracking to reduce ID swaps at crossings.

---

## Motion Outputs

During final render the pipeline writes to:

```txt
outputs/final_renders/     <- Processed video
outputs/motion_exports/    <- JSON, BVH, FBX per tracked person
```

---

## Calibration JSON

A calibration file is a JSON object keyed by camera role.

```json
{
  "front": {
    "rotation_deg": [0.0, 0.0, 0.0],
    "translation": [0.0, 0.0, 0.0],
    "scale": 1.0,
    "confidence_weight": 1.0
  },
  "right": {
    "rotation_deg": [0.0, 90.0, 0.0],
    "translation": [0.0, 0.0, 0.0],
    "scale": 1.0,
    "confidence_weight": 0.98
  }
}
```

Valid roles: `front`, `back`, `right`, `left`, `up`

---

## Packet Shape

Each streamed person contains:

```txt
identity
body
left_hand
right_hand
bones
angles
skeleton
rotations
```

Example identity block:

```json
{
  "identity": {
    "label": "Person 1",
    "profile_color": "orange",
    "profile_region": "top",
    "profile_score": 0.41,
    "top_color": "orange",
    "yolo_track_id": 4,
    "seen_since_timestamp_ms": 0,
    "last_seen_timestamp_ms": 3400
  }
}
```