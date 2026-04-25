# Human Motion Tracking Pipeline (Kinara)

Kinara is a local video and webcam motion-tracking pipeline built around a YOLO body pose model and an ONNX hand pose model. It supports single-person tracking, single-camera multi-person tracking, optional multi-camera fusion, live UDP output for Unreal-side receivers, and stack-safe recorded outputs.

This branch is currently optimized for practical local runtime use:
- YOLO body tracking for both single-person and multi-person flows
- ONNX hand tracking with fallback hands and anatomical cleanup
- clothing-hint assisted multi-person identity stability
- rendered video, JSON, and FBX outputs

---

# Requirements

## Hardware

- Webcam or video file input
- 16 GB RAM recommended
- NVIDIA GPU strongly recommended for YOLO pose inference
- Multi-camera setup optional

## Required Software

```txt
| Tool         | Version           | Notes                                 |
| ---          | ---               | ---                                   |
| Python       | 3.10+             | Pipeline runtime                      |
| PyTorch      | Recent CUDA build | Needed for Ultralytics YOLO           |
| Ultralytics  | Recent            | Body pose and multi-person tracking   |
| ONNX Runtime | Latest compatible | Hand pose inference                   |
| CUDA Toolkit | 12.x+             | Optional GPU acceleration             |
| cuDNN        | 9.x               | Required for ONNX Runtime CUDA        |
```

---

# Installation

## 1. Install Python

Download from:

[https://www.python.org/downloads/](https://www.python.org/downloads/)

Enable Python on `PATH` during installation if you want terminal launch support.

---

## 2. Install NVIDIA Runtime Stack (Optional but Recommended)

Install:

```txt
- recent NVIDIA driver
- CUDA 12.x or newer
- cuDNN 9.x
```

If ONNX Runtime reports a missing `cudnn64_9.dll`, add your cuDNN `bin` directory to `PATH`.

---

## 3. Install Python Dependencies

Example GPU setup:

```bash
pip install ultralytics torch torchvision numpy opencv-python onnxruntime-gpu
```

CPU-only hand inference fallback:

```bash
pip install ultralytics torch torchvision numpy opencv-python onnxruntime
```

---

## 4. Clone The Project

```bash
git clone https://github.com/HrithvikM23/Kinara.git
cd Kinara
```

---

# System Architecture

```txt
Webcam / Video File(s)
        ↓
YOLO Body Pose Detection
        ↓
Per-Person Hand Detection
        ↓
Temporal Smoothing + Hold + Fallback
        ↓
Identity Stabilization / Cross-Person Hand Guard
        ↓
Rendered Output / JSON / FBX / Live UDP
```

---

# Tracking System

## Body Tracking

Body tracking uses an Ultralytics YOLO pose model.

Default body model:

```txt
yolo11x-pose.pt
```

You can replace it with any compatible YOLO pose weights file through `--model`.

If you pass a known YOLO filename such as `yolo11x-pose.pt`, `yolo11l-pose.pt`, `yolo11m-pose.pt`, `yolo11s-pose.pt`, or `yolo11n-pose.pt` and it is missing, Kinara downloads it directly into:

```txt
models/body/
```

## Hand Tracking

Hand tracking uses a YOLO26 hand-pose ONNX model.

Preset mapping:
- `low`, `mid` -> FP16 variant
- `high`, `max` -> FP32 variant

Downloaded hand models are stored in:

```txt
models/hand/
```

Each hand outputs 21 landmarks.

## Hand Robustness Layer

The hand pipeline includes:

```txt
- wrist-directed crop generation
- temporal hold when a hand briefly disappears
- wrist-attached default hand fallback
- anatomical cleanup and distance constraints
- cross-person hand ownership rejection in multi-person mode
```

---

# Output Types

## Single-Person

Single-person runs currently write:

```txt
- rendered tracking video
- motion JSON export
- FBX export
```

## Multi-Person

Single-camera multi-person runs currently write:

```txt
- rendered tracking video
- multi-person JSON export
- live UDP packets
```

Multi-person FBX export is not the default output path yet.

---

# Multi-Person Tracking

Single-camera multi-person mode is currently the supported path.

It uses:

```txt
- YOLO pose detections and tracker IDs
- box continuity
- optional clothing color hints
- wrist ownership checks to reduce hand stealing during crossings
```

Example identity hints:

```bash
py main.py --source ".\two_people.mp4" --max-people 2 --identity person1=black,orange --identity person2=gray,silver
```

Important limitation:

Multi-camera plus multi-person at the same time is not the current default workflow.

---

# Multi-Camera Support

Multi-camera mode currently supports:

```txt
- FRONT
- BACK
- LEFT
- RIGHT
```

The current fusion path is still shared-reference 2D fusion, not full calibration-aware 3D reconstruction.

---

# Model Management

Repo-managed model downloads go directly into the local `models/` tree:

```txt
models/body/
models/hand/
```

That means body and hand weights stay inside the project instead of being left in external caches.

---

# Live UDP Output

Kinara can stream live UDP packets for downstream receivers such as Unreal-side runtime tools.

Current packet content includes:

```txt
- person IDs
- person labels
- body landmarks
- hand landmarks
- hand boxes
```

See [args.md](https://github.com/HrithvikM23/Kinara/blob/kinara/cortex/args.md) for host/port flags.

---

# Output Naming

Rendered and export outputs are stack-safe and never overwrite previous runs.

Example:

```txt
outputs/dance rendered-1.mp4
outputs/dance json-1.json
outputs/dance fbx-1.fbx
```

The next run becomes `-2`, then `-3`, and so on.

---

# How To Run

## Interactive Mode

```bash
cd [drive]:\[path]\Kinara
py main.py
```

Program flow:

```txt
Select input source
1 -> Webcam
2 -> Video file(s)
If video mode:
  Enter number of cameras
  Assign FRONT/BACK/LEFT/RIGHT roles
  Pick one video per assigned role
Pipeline starts
```

Press `ESC` to close preview windows.

## Example Commands

Single-person:

```bash
py main.py --source ".\video.mp4" --model yolo11x-pose.pt
```

Single-person with CPU hand fallback:

```bash
py main.py --source ".\video.mp4" --model yolo11x-pose.pt --provider CPUExecutionProvider
```

Two-person tracking:

```bash
py main.py --source ".\two_people.mp4" --model yolo11x-pose.pt --max-people 2
```

Two-person tracking with identity hints:

```bash
py main.py --source ".\two_people.mp4" --model yolo11x-pose.pt --max-people 2 --identity person1=black,orange --identity person2=gray,silver
```

All CLI arguments are documented in [args.md](https://github.com/HrithvikM23/Kinara/blob/kinara/cortex/args.md).

---

# Current Development Status

```txt
| Component                           | Status   |
| ---                                 | ---      |
| Webcam input                        | Complete |
| Video file input                    | Complete |
| Interactive source selection        | Complete |
| Camera role selection               | Complete |
| Multi-camera synchronized input     | Complete |
| YOLO single-person body tracking    | Complete |
| YOLO multi-person body tracking     | Working  |
| ONNX hand tracking                  | Complete |
| Clothing color identity hints       | Working  |
| Cross-person hand guard             | Working  |
| Automatic model download to models/ | Complete |
| Stack-safe output naming            | Complete |
| Temporal smoothing                  | Complete |
| Rendered output video               | Complete |
| JSON export generation              | Complete |
| FBX export generation               | Complete |
| Live UDP streaming                  | Working  |
| Multi-person FBX export             | Planned  |
| Calibration-aware 3D fusion         | Planned  |
| Multi-camera + multi-person         | Planned  |
```

---

# Roadmap

## Multi-Person Export

Add stable per-person FBX export once the current identity and hand ownership path is fully validated.

## 3D Fusion

Add calibration-aware multi-camera fusion with true cross-view geometry instead of shared 2D reference-space blending.

## Runtime Streaming

Keep the current UDP live-motion path and add an Unreal-side receiver/parser workflow once the packet schema stabilizes.

---

# Technology Stack

## Inference

- Ultralytics YOLO pose
- ONNX Runtime
- YOLO26 hand pose ONNX

## Computer Vision

- OpenCV
- NumPy
- Python

## Stabilization

- Exponential landmark smoothing
- Landmark hold / decay
- Default hand fallback
- Cross-person hand ownership filtering

---

# License

See [LICENSE.md](https://github.com/HrithvikM23/Kinara/blob/kinara/cortex/LICENSE.md).
