# Human Motion to Animation Pipeline

A **real-time human motion capture pipeline** that converts human movement from a webcam or video into **3D character animation inside Unreal Engine 5**.

The system tracks a person using MediaPipe BlazePose, extracts 33 body landmarks with full 3D world coordinates, streams joint data over UDP, and drives a real-time character animation system in Unreal Engine.

---

# Requirements

## Hardware

* NVIDIA GPU (CUDA-capable, recommended)
* Webcam or video file input
* Minimum 16GB RAM recommended

## Required Software

| Tool | Version | Notes |
|------|---------|-------|
| Unreal Engine | 5.4 | Target animation runtime |
| Python | 3.11.9 | Pipeline scripting |
| CUDA Toolkit | 13.2 | GPU acceleration |
| cuDNN | 9.x | Deep learning backend |

---

# Why MediaPipe over OpenPose

OpenPose was originally chosen for this pipeline but was abandoned due to fundamental incompatibilities between its bundled Caffe binaries (compiled for sm_35, a 2013-era GPU architecture) and modern NVIDIA GPUs (RTX 5060 Ti, sm_120 Blackwell). The Caffe dependency cannot be recompiled without a full from-source build involving dozens of additional patches.

MediaPipe BlazePose was chosen as a replacement for the following reasons:

* Installs in a single pip command with no build step
* Natively supports modern NVIDIA GPUs including Blackwell (sm_120)
* Outputs 33 landmarks vs OpenPose's 25, including wrists, ankles, and facial points
* Provides full 3D world coordinates (x, y, z) per joint, not just screen pixels
* Actively maintained by Google with Python 3.11 support
* Faster inference on modern hardware than OpenPose

---

# Installation

Follow these steps in order.

## 1. Install Python 3.11.9

1. Go to https://www.python.org/downloads/release/python-3119/
2. Download **Windows installer (64-bit)**
3. Run installer, check **"Add Python to PATH"**

## 2. Install CUDA Toolkit 13.2

1. Go to https://developer.nvidia.com/cuda-downloads
2. Select Windows → x86_64 → exe (local)
3. Download and run, choose **Express Install**

## 3. Install cuDNN

1. Go to https://developer.nvidia.com/cudnn (free NVIDIA account required)
2. Download cuDNN for CUDA 13.x — Windows zip
3. Extract and copy contents into `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\`

## 4. Install Python dependencies
```cmd
pip install mediapipe opencv-python numpy
```

## 5. Set up the project
```cmd
cd D:\IDT
git clone https://github.com/YOUR_REPO/HMT3A.git
cd HMT3A\HMT3A_Unreal

type nul > camera\__init__.py
type nul > pose_server\__init__.py
type nul > network\__init__.py
type nul > process\__init__.py
type nul > utils\__init__.py
```

---

# Project Goal

Create a full motion-capture pipeline capable of:

* Tracking **human body motion from a camera or video file**
* Detecting **33 BlazePose landmarks** with full 3D world coordinates
* Streaming pose data to Unreal Engine in real time via UDP
* Mapping joints to a **humanoid animation skeleton**
* Driving characters such as **MetaHumans**
* Recording motion as reusable animation data
* Exporting animation for further editing or cinematic use

---

# System Architecture
```
Camera / Video File
       ↓
MediaPipe BlazePose (33 landmarks, x/y/z world coordinates)
       ↓
Joint Coordinates + Visibility scores
       ↓
Skeleton Construction
       ↓
Angle Calculation
       ↓
UDP Streaming (JSON packets → 127.0.0.1:7000)
       ↓
Unreal Engine Receiver
       ↓
Humanoid Skeleton Mapping
       ↓
Real-time Character Animation
```

---

# BlazePose Landmark Map

MediaPipe outputs 33 landmarks. Key joints used for body animation:

| Index | Landmark | UE5 Bone |
|-------|----------|----------|
| 11 | Left Shoulder | clavicle_l |
| 12 | Right Shoulder | clavicle_r |
| 13 | Left Elbow | lowerarm_l |
| 14 | Right Elbow | lowerarm_r |
| 15 | Left Wrist | hand_l |
| 16 | Right Wrist | hand_r |
| 23 | Left Hip | thigh_l |
| 24 | Right Hip | thigh_r |
| 25 | Left Knee | calf_l |
| 26 | Right Knee | calf_r |
| 27 | Left Ankle | foot_l |
| 28 | Right Ankle | foot_r |

---

# Project Structure
```
HMT3A/
├── HMT3A_Blender/
├── HMT3A_Unreal/
│   ├── main.py                  ← Entry point
│   ├── config.py                ← Paths and UDP settings
│   ├── camera/
│   │   ├── video_input.py       ← Video file input (tkinter file picker)
│   │   └── webcam_input.py      ← Webcam input (OpenCV)
│   ├── pose_server/
│   │   └── pose_detector.py     ← MediaPipe BlazePose wrapper
│   ├── network/
│   │   ├── packet_builder.py    ← Landmarks → JSON packets
│   │   └── udp_sender.py        ← UDP socket to Unreal
│   ├── process/
│   │   ├── skeleton_builder.py  ← Joint hierarchy construction
│   │   └── angle_calculator.py  ← Bone rotation calculation
│   └── utils/
│       ├── smoothing.py         ← Landmark smoothing / filtering
│       └── math_utils.py        ← Vector / rotation math helpers
```

---

# How to Run

## Step 1 — Launch Unreal Engine

1. Open the Unreal project
2. Start the scene containing the UDP animation receiver
3. Press Play to begin listening for pose data

## Step 2 — Start the Python Pipeline
```cmd
cd D:\IDT\HMT3A\HMT3A_Unreal
python main.py
```

Select input:
```
1 → Webcam (live)
2 → Video file (file picker opens)
```

---

# Current Development Status

| Component | Status |
|-----------|--------|
| Camera input (video + webcam) | ✅ Done |
| MediaPipe BlazePose integration | ✅ Done |
| 33 landmark extraction | ✅ Done |
| UDP packet builder | ✅ Done |
| UDP sender | ✅ Done |
| Skeleton builder | Planned |
| Angle calculator | Planned |
| Smoothing / filtering | Planned |
| Unreal UDP receiver | Planned |
| MetaHuman skeleton mapping | Planned |
| Animation recording | Planned |

---

# Planned Features

* Full BlazePose → MetaHuman bone mapping
* Multi-person tracking
* Real-time landmark smoothing and filtering
* Inverse Kinematics support
* Animation recording and playback
* FBX export pipeline
* MetaHuman animation support

---

# Technology Stack

**Pose Estimation**
* MediaPipe BlazePose — 33 landmarks, 3D world coordinates
* Google ML Kit backend

**Computer Vision**
* OpenCV
* NumPy
* Python 3.11

**Networking**
* UDP sockets (JSON)

**Game Engine**
* Unreal Engine 5.4
* Blueprint / C++ Animation System
* MetaHuman framework