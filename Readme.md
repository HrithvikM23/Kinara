# Human Motion to Animation Pipeline (HMT3A)

A human motion capture pipeline that converts human movement from a webcam or video into 3D character animation inside Unreal Engine 5.

The system now tracks body + hands together, supports multi-camera input, streams fused joint data over UDP, records raw and processed outputs, and uses a 2-stage workflow with a preview pass first and an optional final render pass after confirmation.

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

During installation:

Check Add Python to PATH

---

## 2. Install CUDA Toolkit 13.2

Download from:

[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

Select:

Windows -> x86_64 -> exe (local)

Run installer with Express Install.

---

## 3. Install cuDNN

1. Go to [cuDNN](https://developer.nvidia.com/cudnn)
2. Download cuDNN for CUDA 13.x
3. Extract contents into:

```txt
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\
```

---

## 4. Install Python Dependencies

```bash
pip install mediapipe opencv-python numpy
```

---

## 5. Set Up the Project

```bash
cd D:\IDT
git clone https://github.com/HrithvikM23/HMT3A.git

cd HMT3A\HMT3A_Unreal
```

Initialize Python packages:

```bash
type nul > camera\__init__.py
type nul > pose_server\__init__.py
type nul > network\__init__.py
type nul > process\__init__.py
type nul > utils\__init__.py
```

---

# Project Goal

Build a motion capture pipeline capable of:

* Tracking human body movement
* Tracking finger motion
* Extracting 3D joint coordinates
* Supporting single-camera and multi-camera setups
* Streaming motion data into Unreal Engine
* Driving MetaHuman characters
* Recording raw and processed tracking video
* Producing reusable animation data
* Providing a fast preview pass before final rendering

---

# System Architecture

```txt
Camera / Video File(s)
        ???
MediaPipe Pose + Hand Tracking
        ???
Body Landmarks + Hand Landmarks
        ???
Per-Camera Landmark Detection
        ???
Multi-Camera Fusion (optional)
        ???
Unified Landmark Packet
        ???
UDP Streaming (JSON)
        ???
Unreal Engine Receiver
        ???
Skeleton Mapping
        ???
Character Animation
```

---

# Tracking System

The pipeline now uses a multimodal tracking approach.

### Body Tracking

Based on MediaPipe BlazePose.

Head landmarks are removed to avoid conflicts with Unreal MetaHuman face systems.

Remaining body joints:

```txt
Indices 11 -> 32
```

Total body joints used:

```txt
22 landmarks
```

---

### Hand Tracking

MediaPipe Hand Landmarker provides:

```txt
21 landmarks per hand
```

Joint structure:

```txt
wrist
thumb_cmc -> thumb_tip
index_mcp -> index_tip
middle_mcp -> middle_tip
ring_mcp -> ring_tip
pinky_mcp -> pinky_tip
```

Total landmarks:

```txt
21 left hand
21 right hand
```

---

# Landmark Output Structure

The UDP packet contains:

```txt
Body      : 22 joints
LeftHand  : 21 joints
RightHand : 21 joints
```

Total possible joints per frame:

```txt
64 landmarks
```

Coordinates include:

```txt
x
y
z
visibility
```

Hand confidence is also included when available.

---

# Multi-Camera Support

The pipeline supports multiple camera roles.

Default primary camera:

```txt
front
```

Optional additional roles:

```txt
back
right
left
up
```

The user selects:

* Number of cameras
* Which role each camera should use
* Number of people to track

Each camera runs detection independently, and the system can fuse joint data into one final skeleton. The primary camera remains the default source, while backup cameras help when joints are missing or weak.

---

# Packet Streaming

Data is streamed via UDP.

```txt
IP:   127.0.0.1
Port: 7000
Format: JSON (UTF-8)
```

Example packet:

```json
{
  "frame": 102,
  "timestamp_ms": 3400,
  "source_fps": 30.0,
  "count": 1,
  "persons": [
    {
      "id": 0,
      "body": {},
      "left_hand": {},
      "right_hand": {},
      "bones": {},
      "angles": {}
    }
  ]
}
```

The packet contains fused per-person landmark data ready for Unreal-side parsing.

---

# Video Output System

The pipeline now uses a 2-stage workflow.

### Stage 1: Preview Pass

* Generates a rough preview animation
* Streams preview data over UDP
* Records raw webcam input if live cameras are used
* Lets the user quickly judge if the capture is usable

### Stage 2: Final Render Pass

* Starts only after user confirmation
* Uses the recorded or uploaded source media
* Produces the final processed animation output
* Saves results to dedicated output folders

Output folders:

```txt
outputs/raw_captures/
outputs/final_renders/
outputs/motion_exports/
```

These outputs are useful for:

* debugging
* dataset generation
* reprocessing
* animation review
* offline reuse in DCC tools

---

# FPS and Resolution Behavior

The system now uses source media settings by default.

### FPS

* Single source: uses that source FPS
* Multiple sources: uses the lowest FPS across all sources
* Manual FPS cap can be applied if needed

Examples:

```txt
120 FPS video  -> 120 FPS processing
30 FPS video   -> 30 FPS processing
60 + 30 + 24   -> 24 FPS processing
```

### Resolution

* Native recorded resolution is used by default
* Manual width and height override can be applied if needed

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

1. Open Unreal project
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
Preview pass starts
Proceed with final render? [Y/n]
```

Press ESC to exit preview windows.

Optional examples:

```bash
python main.py --source 0 --max-persons 1
python main.py --source "C:\path\to\video.mp4" --no-preview
python main.py --source 0 --fps-cap 30 --width 1280 --height 720
python main.py --source 0 --calibration-file "C:\path\to\camera_calibration.json"
python main.py --source 0 --no-fbx-export
```

---

# Current Development Status

| Component                          | Status   |
|-----------------------------------|----------|
| Camera input                       | Complete |
| Video file input                   | Complete |
| Multi-source input selector        | Complete |
| Camera role selection              | Complete |
| Multi-camera role system           | Complete |
| BlazePose integration              | Complete |
| Hand tracking (21 landmarks)       | Complete |
| Head landmark filtering            | Complete |
| Packet builder                     | Complete |
| Dynamic landmark packet map        | Complete |
| UDP streaming                      | Complete |
| Configurable UDP settings          | Complete |
| Raw capture recording              | Complete |
| Final render output                | Complete |
| Variable FPS configuration         | Complete |
| Native/lowest FPS selection        | Complete |
| Manual FPS cap                     | Complete |
| Manual resolution override         | Complete |
| Frame timestamp system             | Complete |
| Preview-first workflow             | Complete |
| Calibration-aware multi-camera fusion | Complete |
| Persistent multi-person tracking   | Complete |
| Landmark smoothing                 | Complete |
| Wrist-guided hand ROI logic        | Complete |
| Skeleton builder                   | Complete |
| Bone reconstruction                | Complete |
| Bone rotation solver               | Complete |
| Quaternion conversion              | Complete |
| Hand jitter reduction              | Complete |
| Motion export (JSON/BVH/FBX)       | Complete |
| Unreal UDP receiver                | Planned  |
| Unreal JSON packet parser          | Planned  |
| Unreal Control Rig mapping         | Planned  |
| MetaHuman skeleton mapping         | Planned  |
| Motion playback system             | Planned  |

---

# Roadmap

Next development goals:

### Unreal Receiver

Build the Unreal-side system that converts UDP packets into live animation input.

### Control Rig and MetaHuman Mapping

Map the client-side skeleton, rotations, and finger data onto the Unreal rig.

### Playback and Cleanup

Add richer offline playback, review, and cleanup tools for exported motion packages.

---

# Technology Stack

### Pose Estimation

* MediaPipe BlazePose
* MediaPipe Hand Landmarker

### Computer Vision

* OpenCV
* NumPy
* Python 3.11

### Networking

* UDP sockets
* JSON serialization

### Game Engine

* Unreal Engine 5.4
* Blueprint animation system
* MetaHuman framework

---

# Project Phase

Alpha Development

Current focus:

```txt
Multi-Camera Fusion
Packet Optimization
Skeleton Reconstruction
Unreal Animation Mapping
```

