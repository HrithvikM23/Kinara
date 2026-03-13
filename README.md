# Human Motion to Animation Pipeline

A **real-time human motion capture pipeline** that converts human movement from a webcam or video into **3D character animation inside Unreal Engine 5**.

The system tracks a person using pose estimation, streams body joint data over UDP, maps it to a humanoid skeleton, and drives a real-time character animation system.

---

# Requirements

## Hardware

* NVIDIA GPU (recommended)
* Webcam or camera device
* Minimum 16GB RAM recommended

## Required Applications

* Unreal Engine 5.4
* Visual Studio Code
* Python 3.9/3.10
* Visual Studio Build Tools
* CUDA Toolkit 12.4
* cuDNN 9.x
* Cmake 3.26

---

# Project Goal

Create a full motion-capture pipeline capable of:

* Tracking **human body motion from a camera or video**
* Detecting **multi-person pose data**
* Streaming pose information to Unreal Engine in real time
* Mapping joints to a **humanoid animation skeleton**
* Driving characters such as **MetaHumans**
* Recording motion as reusable animation data
* Exporting animation for further editing or cinematic use

---

# System Architecture

```
Camera / Video
       ↓
Pose Detection (OpenPose / AI Pose Model)
       ↓
Joint Coordinates
       ↓
Skeleton Construction
       ↓
Angle Calculation
       ↓
UDP Streaming
       ↓
Unreal Engine Receiver
       ↓
Humanoid Skeleton Mapping
       ↓
Real-time Character Animation
```

---

# Pipeline Overview

### Motion Capture

Human motion is captured using a webcam or video input.

### Pose Detection

The AI pose system extracts body joints such as:

* shoulders
* elbows
* wrists
* hips
* knees
* ankles

### Pose Processing

Joint coordinates are converted into animation-ready bone rotations.

### Data Streaming

Pose data is transmitted via **UDP** to the Unreal Engine application.

### Unreal Animation System

The Unreal receiver interprets incoming pose data and applies it to a humanoid skeleton.

---

# How to Run (Current Workflow)

## Step 1 — Launch Unreal Engine

1. Open the Unreal project
2. Start the scene containing the animation receiver
3. Run the project to begin listening for pose data

## Step 2 — Start the Pose Server

From the project directory:

```
cd pose_server
python main.py
```

Select input source:

```
1 → Webcam
2 → Video file
```

The server will begin tracking motion and streaming pose data.

---

# Current Development Status

| Component                   | Status         |
| --------------------------- | -------------- |
| Pose detection pipeline     | In development |
| Multi-person tracking       | Planned        |
| UDP streaming system        | Designed       |
| Unreal data receiver        | Planned        |
| Character skeleton mapping  | Planned        |
| Real-time animation preview | Planned        |
| Animation recording system  | Planned        |

---

# Planned Features

* Full humanoid bone mapping
* Multi-person tracking
* Real-time smoothing and filtering
* Inverse Kinematics support
* Animation recording and playback
* FBX export pipeline
* MetaHuman animation support

---

# Technology Stack

**Computer Vision**

* Python
* OpenPose
* OpenCV
* NumPy

**Game Engine**

* Unreal Engine 5
* C++ / Blueprint Animation System

