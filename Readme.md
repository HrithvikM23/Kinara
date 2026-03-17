# Human Motion to Animation Pipeline (HMT3A)

A **real-time human motion capture pipeline** that converts human movement from a webcam or video into **3D character animation inside Unreal Engine 5**.

The system now tracks **body + hands simultaneously**, streams joint data over UDP, and records the processed video output for debugging and dataset generation.

---

# Requirements

## Hardware

* NVIDIA GPU (CUDA-capable, recommended)
* Webcam or video file input
* Minimum **16GB RAM** recommended

## Required Software

| Tool          | Version | Notes                    |
| ------------- | ------- | ------------------------ |
| Unreal Engine | 5.4     | Target animation runtime |
| Python        | 3.11.9  | Pipeline scripting       |
| CUDA Toolkit  | 13.2    | GPU acceleration         |
| cuDNN         | 9.x     | Deep learning backend    | 


---

# Why MediaPipe over OpenPose

OpenPose was originally chosen but was abandoned due to incompatibilities between its bundled **Caffe binaries (compiled for sm_35)** and modern NVIDIA GPUs (RTX 50-series Blackwell architecture).

MediaPipe BlazePose was adopted because it:

* Installs with **one pip command**
* Works with **modern NVIDIA GPUs**
* Outputs **33 landmarks with full 3D coordinates**
* Is **actively maintained by Google**
* Runs significantly **faster on modern hardware**

---

# Installation

Follow these steps in order.

## 1. Install Python 3.11.9

Download from:

https://www.python.org/downloads/release/python-3119/

During installation:

✔ Check **Add Python to PATH**

---

## 2. Install CUDA Toolkit 13.2

Download from:

[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

Select:

Windows → x86_64 → exe (local)

Run installer with **Express Install**.

---

## 3. Install cuDNN

1. Go to ([cuDNN](https://developer.nvidia.com/cudnn))
2. Download **cuDNN for CUDA 13.x**
3. Extract contents into:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\
```

---

## 4. Install Python Dependencies

```
pip install mediapipe opencv-python numpy
```

---

## 5. Set Up the Project

```
cd D:\IDT
git clone https://github.com/HrithvikM23/HMT3A.git

cd HMT3A\HMT3A_Unreal
```

Initialize Python packages:

```
type nul > camera\__init__.py
type nul > pose_server\__init__.py
type nul > network\__init__.py
type nul > process\__init__.py
type nul > utils\__init__.py
```

---

# Project Goal

Build a **real-time motion capture pipeline** capable of:

* Tracking **human body movement**
* Tracking **finger motion**
* Extracting **3D joint coordinates**
* Streaming motion data into **Unreal Engine**
* Driving **MetaHuman characters**
* Recording **processed tracking video**
* Producing reusable **animation data**

---

# System Architecture

```
Camera / Video File
        ↓
MediaPipe Pose + Hand Tracking
        ↓
Body Landmarks (22) + Hand Landmarks (21x2)
        ↓
Unified Landmark Packet
        ↓
UDP Streaming (JSON)
        ↓
Unreal Engine Receiver
        ↓
Skeleton Mapping
        ↓
Real-time Character Animation
```

---

# Tracking System

The pipeline now uses a **multimodal tracking approach**.

### Body Tracking

Based on **MediaPipe BlazePose**.

Head landmarks are removed to avoid conflicts with Unreal's MetaHuman face tracking.

Remaining body joints:

```
Indices 11 → 32
```

Total body joints used:

```
22 landmarks
```

---

### Hand Tracking

MediaPipe **Hand Landmarker** provides:

```
21 landmarks per hand
```

Joint structure:

```
wrist
thumb_cmc → thumb_tip
index_mcp → index_tip
middle_mcp → middle_tip
ring_mcp → ring_tip
pinky_mcp → pinky_tip
```

Total landmarks:

```
21 left hand
21 right hand
```

---

# Landmark Output Structure

The UDP packet contains:

```
Body      : 22 joints
LeftHand  : 21 joints
RightHand : 21 joints
```

Total possible joints per frame:

```
64 landmarks
```

All coordinates include:

```
x
y
z
visibility
```

---

# Packet Streaming

Data is streamed via **UDP**.

```
IP:   127.0.0.1
Port: 7000
Format: JSON (UTF-8)
```

Example packet:

```
{
  "frame": 102,
  "count": 1,
  "body": {...},
  "left_hand": {...},
  "right_hand": {...}
}
```

Average streaming rate:

```
~30 FPS
≈33 ms latency
```

---

# Video Output System

The pipeline records the **processed tracking frames**.

Files are saved automatically in:

```
outputs/
```

Naming scheme:

```
video.mp4
video1.mp4
video2.mp4
...
```

These videos contain:

* body skeleton overlay
* hand skeleton overlay
* real tracking results

This is used for **debugging and dataset generation**.

---

# Project Structure

```
HMT3A/
├── HMT3A_Blender/
├── HMT3A_Unreal/
│
│   main.py
│   config.py
│
├── camera/
│   video_input.py
│   webcam_input.py
│
├── pose_server/
│   pose_detector.py
│
├── network/
│   packet_builder.py
│   udp_sender.py
│
├── process/
│   skeleton_builder.py
│   angle_calculator.py
│
└── utils/
    video_output.py
    smoothing.py
    math_utils.py
```

---

# How to Run

## Step 1 — Launch Unreal Engine

1. Open Unreal project
2. Load the animation receiver scene
3. Press **Play**

---

## Step 2 — Run the Python Pipeline

```
cd D:\IDT\HMT3A\HMT3A_Unreal
python main.py
```

Program menu:

```
Enter number of people to track:

Select input source
1 → Webcam
2 → Video file
```

Press **ESC** to exit.


# Current Development Status
---

| Component                    | Status         |
|------------------------------|----------------|
| Camera input                 | Complete       |
| Video file input             | Complete       |
| Multi-source input selector  | Complete       |
| BlazePose integration        | Complete       |
| Hand tracking (21 landmarks) | Complete       |
| Head landmark filtering      | Complete       |
| Packet builder               | Complete       |
| Dynamic landmark packet map  | Complete       |
| UDP streaming                | Complete       |
| Configurable UDP settings    | Complete       |
| Video recording              | Complete       |
| Auto video file indexing     | Complete       |
| Variable FPS configuration   | In Development |
| Frame timestamp system       | Complete       |
| Async inference timestamps   | Complete       |
| Skeleton builder             | In Development |
| Bone reconstruction          | Planned        |
| Bone rotation solver         | Planned        |
| Quaternion conversion        | Planned        |
| Landmark smoothing filter    | Planned        |
| Hand jitter reduction        | Planned        |
| Multi-person tracking        | Planned        |
| Unreal UDP receiver          | Planned        |
| Unreal JSON packet parser    | Planned        |
| Unreal Control Rig mapping   | Planned        |
| MetaHuman skeleton mapping   | Planned        |
| Animation recording system   | Planned        |
| Motion playback system       | Planned        |
| Animation export (FBX)       | Planned        |

---

# Roadmap

Next development goals:

### Skeleton Builder

Convert raw landmark positions into **hierarchical skeleton data**.

### Bone Rotation Solver

Compute **quaternion rotations** from joint vectors.

### Landmark Smoothing

Implement **One-Euro filter** to remove jitter.

### Unreal Receiver

Blueprint system that converts UDP packets into **Control Rig animation**.

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

**Alpha Development**

Current focus:

```
Data Fusion
Packet Optimization
Real-time Animation Streaming
```
