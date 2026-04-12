# Human Motion to Animation Pipeline (HMT3A)

A human motion capture pipeline that converts human movement from a webcam or video into 3D character animation inside Unreal Engine 5.

The system tracks body and hands together, supports multi-camera input, streams fused joint data over UDP, records raw and processed outputs, and uses a 2-stage workflow with a fast preview pass first and an optional heavier final render pass after confirmation.

---

# Requirements

## Hardware

* Webcam or video file input
* Minimum 16GB RAM recommended
* NVIDIA GPU recommended for heavier workloads
* Multi-camera setup optional

## Required Software

| Tool          | Version | Notes                    |
| ------------- | ------- | ------------------------ |
| Unreal Engine | 5.4     | Target animation runtime |
| Python        | 3.11.9  | Pipeline scripting       |
| CUDA Toolkit  | 13.2    | Optional acceleration    |
| cuDNN         | 9.x     | Optional backend support |

---

# Installation

Follow these steps in order.

## 1. Install Python 3.11.9

Download from:

https://www.python.org/downloads/release/python-3119/

During installation, check **Add Python to PATH**.

---

## 2. Install CUDA Toolkit 13.2

Download from:

https://developer.nvidia.com/cuda-downloads

Select: `Windows → x86_64 → exe (local)` and run with Express Install.

---

## 3. Install cuDNN

1. Go to https://developer.nvidia.com/cudnn
2. Download cuDNN for CUDA 13.x
3. Extract contents into:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\
```

---

## 4. Install Python Dependencies

Base pipeline:

```bash
pip install mediapipe opencv-python numpy
```

Full pipeline including final-render identity assist:

```bash
pip install ultralytics torch torchvision
```

The identity stack enables YOLO person detection and tracking, optional Mask R-CNN refinement, and configurable color-based identity memory.

---

## 5. Set Up the Project

```bash
cd D:\IDT
git clone https://github.com/HrithvikM23/HMT3A.git

cd HMT3A\HMT3A_Unreal
```

Initialize Python packages if needed:

```bash
type nul > camera\__init__.py
type nul > pose_server\__init__.py
type nul > network\__init__.py
type nul > process\__init__.py
type nul > utils\__init__.py
```

---

# System Architecture

```txt
Camera / Video File(s)
        ->
Preview Pass:
MediaPipe Pose + Hand Tracking
        ->
Per-Camera Landmark Detection
        ->
Multi-Camera Fusion (optional)
        ->
UDP Streaming / Preview Output

Final Render Pass (optional heavier path):
YOLO Person Detection / Tracking (optional)
        ->
Mask R-CNN Person Refinement (optional)
        ->
Per-Person MediaPipe Pose + Hand Tracking
        ->
Color-Based Identity Memory (optional)
        ->
Multi-Camera Fusion
        ->
Unified Landmark Packet + Motion Export
```

---

# Tracking System

### Body Tracking

Based on MediaPipe BlazePose. Head landmarks are removed to avoid conflicts with Unreal MetaHuman face systems.

Remaining joints use indices 11 → 32, totalling 22 landmarks.

---

### Hand Tracking

MediaPipe Hand Landmarker provides 21 landmarks per hand:

```txt
wrist
thumb_cmc -> thumb_tip
index_mcp -> index_tip
middle_mcp -> middle_tip
ring_mcp -> ring_tip
pinky_mcp -> pinky_tip
```

---

### Multi-Person Identity Assist

When enabled for the final render, the pipeline combines YOLO person boxes with persistent track IDs, optional Mask R-CNN masks to reduce cross-person landmark contamination, and configurable appearance rules such as `Person 1 = orange top`. Specifically designed to reduce ID swaps when people cross or briefly overlap.

---

# Landmark Output

The UDP packet contains:

```txt
Body      : 22 joints
LeftHand  : 21 joints
RightHand : 21 joints
Total     : 64 landmarks per person per frame
```

Each landmark includes `x`, `y`, `z`, and `visibility`. Multi-person packets also carry optional identity metadata including label, color cue, YOLO track ID, and seen/last-seen timestamps.

---

# Multi-Camera Support

Default primary camera: `front`

Optional additional roles: `back`, `right`, `left`, `up`

Each camera runs detection independently. The primary camera is the main source while backup cameras help when joints are missing or weak.

---

# Packet Streaming

```txt
IP:     127.0.0.1
Port:   7000
Format: JSON (UTF-8)
```

---

# Video Output System

### Stage 1: Preview Pass

Runs lightweight detection for a quick look at capture quality. Streams data over UDP and records raw webcam input. Heavy detector assists are kept off for speed.

### Stage 2: Final Render Pass

Starts only after user confirmation. Enables YOLO, Mask R-CNN, and identity memory if configured. Produces the final processed video and animation exports.

```txt
outputs/raw_captures/
outputs/final_renders/
outputs/motion_exports/
```

---

# FPS and Resolution

* Single source: uses that source FPS
* Multiple sources: uses the lowest FPS across all sources
* Manual FPS cap and fixed resolution can be applied via CLI arguments

---

# Project Structure

```txt
HMT3A/
|-- HMT3A_Blender/
|-- HMT3A_Unreal/
|   |-- main.py
|   |-- config.py
|   |-- camera/
|   |-- pose_server/
|   |-- network/
|   |-- process/
|   |-- utils/
|   |-- outputs/
|   `-- models/
```

---

# How to Run

## Step 1 - Launch Unreal Engine

1. Open the Unreal project
2. Load the animation receiver scene
3. Press Play

---

## Step 2 - Run the Python Pipeline

```bash
cd D:\IDT\HMT3A\HMT3A_Unreal
python main.py
```

Program flow:

```txt
Enter number of people to track
Select input source
1 -> Webcam
2 -> Video file
Enter number of cameras
Assign camera roles
Optionally configure color identity profiles
Preview pass starts
Proceed with final render? [Y/n]
```

Press ESC to exit preview windows.

Input source, person count, FPS cap, resolution, export formats, UDP target, confidence thresholds, YOLO model, identity memory, and camera calibration can all be configured via CLI arguments. See **ARGS.md** for the full reference.

---

# Current Development Status

| Component                             | Status   |
| ------------------------------------- | -------- |
| Camera input                          | Complete |
| Video file input                      | Complete |
| Multi-source input selector           | Complete |
| Camera role selection                 | Complete |
| Multi-camera role system              | Complete |
| BlazePose integration                 | Complete |
| Hand tracking (21 landmarks)          | Complete |
| Head landmark filtering               | Complete |
| Packet builder                        | Complete |
| UDP streaming                         | Complete |
| Raw capture recording                 | Complete |
| Final render output                   | Complete |
| Variable FPS configuration            | Complete |
| Manual FPS cap                        | Complete |
| Manual resolution override            | Complete |
| Preview-first workflow                | Complete |
| Calibration-aware multi-camera fusion | Complete |
| Persistent multi-person tracking      | Working  |
| Color-based identity memory           | Working  |
| YOLO-assisted final-render tracking   | Working  |
| Optional Mask R-CNN refinement        | Complete |
| Landmark smoothing                    | Complete |
| Skeleton + bone reconstruction        | Working  |
| Quaternion conversion                 | Working  |
| Motion export (JSON / BVH / FBX)      | Complete |
| Unreal UDP receiver                   | Planned  |
| Unreal JSON packet parser             | Planned  |
| Unreal Control Rig mapping            | Planned  |
| MetaHuman skeleton mapping            | Planned  |
| Motion playback system                | Planned  |

---

# Roadmap

### Unreal Receiver
Build the Unreal-side system that converts UDP packets into live animation input.

### Control Rig and MetaHuman Mapping
Map the skeleton, rotations, finger data, and multi-person identity metadata onto the Unreal rig.

### Playback and Cleanup
Add offline playback, review, and cleanup tools for exported motion packages.

---

# Technology Stack

### Pose Estimation
* MediaPipe BlazePose
* MediaPipe Hand Landmarker

### Detection and Identity Assist
* YOLOv8 person detection and tracking
* Mask R-CNN person segmentation refinement
* HSV color-region identity memory

### Computer Vision
* OpenCV
* NumPy
* Python 3.11

### Networking
* UDP sockets
* JSON serialization

### Game Engine
* Unreal Engine 5.4
* MetaHuman framework

---

# Project Phase

Alpha Development

Current focus:

```txt
Identity-Stable Multi-Person Tracking
Packet Optimization
Skeleton Reconstruction
Unreal Animation Mapping
```