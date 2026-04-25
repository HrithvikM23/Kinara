# Human Motion Tracking Pipeline (Kinara)

An ONNX Runtime based human motion tracking pipeline that captures body pose and hand landmarks from webcam or video input, supports optional multi-camera fusion, and produces rendered tracking outputs for downstream animation or motion processing workflows.

This branch is the ONNX-based tracking path. It focuses on configurable local inference, stack-safe output generation, automatic model setup, and synchronized multi-camera fusion in a shared 2D reference space.

---

# Requirements

## Hardware

- Webcam or video file input
- 16 GB RAM recommended
- NVIDIA GPU recommended for faster inference
- Multi-camera setup optional

## Required Software

| Tool | Version | Notes |
| --- | --- | --- |
| Python | 3.10+ | Pipeline runtime |
| ONNX Runtime | Latest compatible | CPU or GPU execution |
| CUDA Toolkit | 12.x | Optional GPU acceleration |
| cuDNN | 9.x | Required for ONNX Runtime CUDA provider |

---

# Installation

Follow these steps in order.

## 1. Install Python

Download from:

[https://www.python.org/downloads/](https://www.python.org/downloads/)

During installation, enable Python on `PATH` if you want to launch it directly from terminal.

---

## 2. Install CUDA Toolkit (Optional)

Download from:

[https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

Use CUDA 12.x if you want GPU inference through ONNX Runtime.

---

## 3. Install cuDNN (Optional, Required for CUDAExecutionProvider)

1. Download cuDNN 9.x for CUDA 12.x from NVIDIA
2. Install or extract it
3. Make sure the cuDNN `bin` directory is available on `PATH`

Typical example:

```txt
C:\Program Files\NVIDIA\CUDNN\v9.21\bin\12.9\x64
```

If `cudnn64_9.dll` is missing, ONNX Runtime CUDA will fail and you should either fix `PATH` or run with CPU provider.

---

## 4. Install Python Dependencies

GPU path:

```bash
pip install numpy opencv-python onnxruntime-gpu
```

CPU-only path:

```bash
pip install numpy opencv-python onnxruntime
```

---

## 5. Set Up the Project

```bash
cd D:\IDT
git clone https://github.com/HrithvikM23/Kinara.git
cd Kinara
```

The pipeline downloads the default ONNX presets automatically on first run if they are missing.

---

# System Architecture

```txt
Webcam / Video File(s)
        ↓
Per-View ONNX Body Detection
        ↓
Per-View Hand Detection
        ↓
Temporal Smoothing + Landmark Hold
        ↓
Multi-Camera Fusion (optional)
        ↓
Rendered Tracking Output
```

---

# Tracking System

### Body Tracking

Body tracking uses an ONNX MoveNet preset:

- `low`, `mid` -> MoveNet Lightning
- `high`, `max` -> MoveNet Thunder

The current pipeline uses the single-person body keypoint structure expected by the ONNX model loaded at runtime.

---

### Hand Tracking

Hand tracking uses a YOLO26 hand-pose ONNX preset:

- `low`, `mid` -> FP16 variant
- `high`, `max` -> FP32 variant

Each hand outputs 21 landmarks.

---

# Landmark Output

Current runtime output includes:

```txt
Body      : 17 keypoints
LeftHand  : 21 keypoints
RightHand : 21 keypoints
```

Each run now writes:

- rendered tracking video
- motion JSON export
- BVH export
- FBX export

---

# Multi-Camera Support

Default primary camera: `FRONT`

Optional additional roles:

- `BACK`
- `LEFT`
- `RIGHT`

Each camera runs detection independently. During fusion, the system maps joints into a shared front-reference space and combines them with confidence weighting.

Important note:

This is currently 2D multi-view fusion, not calibration-aware 3D reconstruction. It improves robustness when one view loses a joint, but it is not full motion-capture triangulation.

---

# Model Management

Default model files:

```txt
models/body/movenet_thunder.onnx
models/hand/yolo26_hand_pose_fp32.onnx
```

If they are missing, the pipeline downloads them automatically on first run.

You can also override the presets with explicit paths or choose different presets through CLI arguments. See [args.md](D:\IDT\Kinara\args.md).

---

# Video Output System

Rendered outputs are stack-safe and never overwrite previous runs.

Example:

```txt
outputs/dance rendered-1.mp4
outputs/dance json-1.json
outputs/dance bvh-1.bvh
outputs/dance fbx-1.fbx
```

The next run becomes `-2`, then `-3`, and so on.

For fused multi-camera runs, the pipeline automatically uses a fused basename:

```txt
outputs/dance_fused rendered-N.mp4
```

---

# FPS and Resolution

- Single source: uses that source FPS
- Multiple sources: fused output currently uses the reference camera FPS
- Fallback FPS can be supplied for sources that report invalid FPS
- Output writer codec can be changed via CLI

---

# Project Structure

```txt
Kinara/
|-- main.py
|-- config.py
|-- args.md
|-- camera/
|-- inference/
|-- network/
|-- pipeline/
|-- utils/
|-- outputs/
`-- models/
```

---

# How to Run

## Interactive Mode

```bash
cd D:\IDT\Kinara
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

Single video:

```bash
py main.py --source "C:\path\to\video.mp4"
```

CPU-only fallback:

```bash
py main.py --source "C:\path\to\video.mp4" --provider CPUExecutionProvider
```

Explicit provider priority:

```bash
py main.py --source 0 --provider CUDAExecutionProvider --provider CPUExecutionProvider
```

Alternate model presets:

```bash
py main.py --source 0 --model movenet=max --model hand=high
```

All CLI arguments are documented in [args.md](D:\IDT\Kinara\args.md).

---

# Current Development Status

| Component | Status |
| --- | --- |
| Webcam input | Complete |
| Video file input | Complete |
| Interactive source selection | Complete |
| Camera role selection | Complete |
| Multi-camera synchronized input | Complete |
| ONNX body tracking | Complete |
| ONNX hand tracking | Complete |
| Automatic model download | Complete |
| Stack-safe output naming | Complete |
| Temporal smoothing | Complete |
| 2D confidence-based fusion | Complete |
| Rendered output video | Complete |
| JSON export generation | Complete |
| BVH export generation | Complete |
| FBX export generation | Complete |
| OSC payload output | Placeholder |
| Calibration-aware 3D fusion | Planned |
| Multi-person tracking | Planned |

---

# Roadmap

### 3D Fusion

Add calibration-aware multi-camera fusion with true cross-view geometry instead of shared 2D reference-space blending.

### Motion Export

Improve the current JSON, BVH, and FBX exports with richer skeleton metadata, rotation solving, and downstream animation compatibility.

### Runtime Streaming

Replace the current OSC placeholder with a structured live motion streaming layer for downstream animation tools.

---

# Technology Stack

### Inference

- ONNX Runtime
- MoveNet ONNX
- YOLO26 hand pose ONNX

### Computer Vision

- OpenCV
- NumPy
- Python

### Fusion and Stabilization

- Confidence-weighted multi-view fusion
- Exponential landmark smoothing
- Short-term landmark persistence

---

# Project Phase

Alpha Development

Current focus:

```txt
Multi-Camera Fusion Stability
Motion Export Foundation
Provider Robustness
Type-Safe Runtime Cleanup
```

---

# License

See [LICENSE.md](D:\IDT\Kinara\LICENSE.md).
