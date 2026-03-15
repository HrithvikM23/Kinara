# Human Motion to Animation Pipeline

A **real-time human motion capture pipeline** that converts human movement from a webcam or video into **3D character animation inside Unreal Engine 5**.

The system tracks a person using OpenPose pose estimation, extracts BODY_25 keypoints, streams joint data over UDP, and drives a real-time character animation system in Unreal Engine.

---

# Requirements

## Hardware

* NVIDIA GPU (CUDA-capable, required)
* Webcam or video file input
* Minimum 16GB RAM recommended

## Required Software

| Tool | Version | Notes |
|------|---------|-------|
| Unreal Engine | 5.4 | Target animation runtime |
| Python | 3.11.9 | Pipeline scripting |
| Visual Studio | 2025 Community (v18) | Required to build OpenPose |
| CUDA Toolkit | 12.4 | GPU acceleration |
| cuDNN | 9.x | Deep learning backend |
| CMake | 4.x | Build system |
| OpenPose | Latest (CMU) | Pose estimation engine |

---

# Installation

Follow these steps in order. Each step must complete successfully before moving to the next.

---

## 1. Install Visual Studio 2025 Community

1. Go to https://visualstudio.microsoft.com/downloads/
2. Download **Visual Studio 2025 Community** (free)
3. Run the installer
4. When prompted for workloads, check **"Desktop development with C++"**
5. Click **Install**

Verify:
```cmd
dir "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC"
```
You should see a version folder like `14.50.xxxxx`.

---

## 2. Install CUDA Toolkit 12.4

1. Go to https://developer.nvidia.com/cuda-12-4-0-download-archive
2. Select: Windows → x86_64 → your Windows version → exe (local)
3. Download and run the installer
4. Use **Express** install

Verify:
```cmd
nvcc --version
```
Should show `release 12.4`.

---

## 3. Install cuDNN 9.x

1. Go to https://developer.nvidia.com/cudnn (requires free NVIDIA account)
2. Download **cuDNN 9.x for CUDA 12.x** — Windows zip
3. Extract the zip
4. Copy the contents into your CUDA install folder:
   - `bin\` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\`
   - `include\` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include\`
   - `lib\` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\`

---

## 4. Install CMake 4.x

1. Go to https://cmake.org/download/
2. Download the latest **Windows x64 Installer** `.msi`
3. Run the installer
4. When asked about PATH, select **"Add CMake to the system PATH for all users"**

Verify:
```cmd
cmake --version
```
Should show `4.x.x`.

---

## 5. Install Python 3.11.9

1. Go to https://www.python.org/downloads/release/python-3119/
2. Download **Windows installer (64-bit)**
3. Run installer
4. Check **"Add Python to PATH"** at the bottom before clicking Install

Verify:
```cmd
python --version
```
Should show `Python 3.11.9`.

Install required Python packages:
```cmd
pip install opencv-python numpy
```

---

## 6. Clone and Build OpenPose

### 6a. Clone
```cmd
cd D:\IDT
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
cd openpose
git submodule update --init --recursive
```

### 6b. Download models
```cmd
cd D:\IDT\openpose\models
getModels.bat
```

Wait for download to complete. The BODY_25 model is ~200MB.

Verify:
```cmd
dir D:\IDT\openpose\models\pose\body_25
```
You should see `pose_iter_584000.caffemodel`.

### 6c. Patch CMakeLists.txt for CMake 4.x compatibility
```cmd
powershell -Command "(Get-Content D:\IDT\openpose\CMakeLists.txt) -replace 'find_package\(CUDA\)', 'find_package(CUDAToolkit)' | Set-Content D:\IDT\openpose\CMakeLists.txt"
```

### 6d. Configure
```cmd
cd D:\IDT\openpose
mkdir build
cd build

cmake .. -G "Visual Studio 18 2026" -A x64 ^
  -DBUILD_PYTHON=ON ^
  -DPYTHON_EXECUTABLE="C:\Users\YOUR_USERNAME\AppData\Local\Programs\Python\Python311\python.exe" ^
  -DGPU_MODE=CUDA ^
  -DCUDA_TOOLKIT_ROOT_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
```

> Replace `YOUR_USERNAME` with your Windows username. Find it with `whoami`.

### 6e. Build
```cmd
cmake --build . --config Release
```

This takes 10–30 minutes. Let it run.

### 6f. Verify build output
```cmd
dir D:\IDT\openpose\build\bin\OpenPoseDemo.exe
dir D:\IDT\openpose\build\python\openpose\Release\
```

You should see `OpenPoseDemo.exe` and a `.pyd` file.

### 6g. Test the executable
```cmd
cd D:\IDT\openpose
build\bin\OpenPoseDemo.exe --video "D:\IDT\test.mp4" --model_folder models --net_resolution 368x368
```

A window should open showing the video with skeleton overlay drawn on the person.

---

## 7. Set Up the Python Project

### 7a. Clone the project
```cmd
cd D:\IDT
git clone https://github.com/YOUR_REPO/HMT3A.git
cd HMT3A\HMT3A_Unreal
```

### 7b. Create missing `__init__.py` files
```cmd
type nul > camera\__init__.py
type nul > pose_server\__init__.py
type nul > network\__init__.py
type nul > process\__init__.py
type nul > utils\__init__.py
```

### 7c. Verify pyopenpose is importable
```cmd
python -c "import sys; sys.path.append(r'D:\IDT\openpose\build\python\openpose\Release'); import pyopenpose; print('pyopenpose OK')"
```

Should print `pyopenpose OK`.

---

# Project Goal

Create a full motion-capture pipeline capable of:

* Tracking **human body motion from a camera or video file**
* Detecting **BODY_25 keypoints** (25 joints per person)
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
OpenPose (BODY_25 — 25 keypoints)
       ↓
Joint Coordinates (x, y, confidence)
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
│   │   └── pose_detector.py     ← pyopenpose wrapper, BODY_25 extraction
│   ├── network/
│   │   ├── packet_builder.py    ← Keypoints → JSON packets
│   │   └── udp_sender.py        ← UDP socket to Unreal
│   ├── process/
│   │   ├── skeleton_builder.py  ← Joint hierarchy construction
│   │   └── angle_calculator.py  ← Bone rotation calculation
│   └── utils/
│       ├── smoothing.py         ← Keypoint smoothing / filtering
│       └── math_utils.py        ← Vector / rotation math helpers
└── openpose/                    ← CMU OpenPose (built separately)
    ├── build/
    │   ├── bin/                 ← OpenPoseDemo.exe + DLLs
    │   └── python/openpose/     ← pyopenpose.pyd
    └── models/                  ← BODY_25 caffemodel
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
| OpenPose build + pyopenpose | 🔧 In progress |
| BODY_25 keypoint extraction | 🔧 In progress |
| UDP packet builder | Designed |
| UDP sender | Designed |
| Skeleton builder | Planned |
| Angle calculator | Planned |
| Smoothing / filtering | Planned |
| Unreal UDP receiver | Planned |
| MetaHuman skeleton mapping | Planned |
| Animation recording | Planned |

---

# Planned Features

* Full BODY_25 → humanoid bone mapping
* Multi-person tracking
* Real-time keypoint smoothing and filtering
* Inverse Kinematics support
* Animation recording and playback
* FBX export pipeline
* MetaHuman animation support

---

# Technology Stack

**Pose Estimation**
* OpenPose (CMU) — BODY_25 model
* pyopenpose — Python bindings

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