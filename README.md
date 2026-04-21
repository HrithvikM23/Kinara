# Real-Time Pose + Hand Tracking (ONNX Runtime, CUDA)

## Status

This project is **alpha-stage**.

Expect rough edges:
- paths are hardcoded
- model names are hardcoded
- thresholds are hardcoded
- input/output settings are hardcoded
- error handling is minimal
- tracking stability is still being refined

This is currently a working prototype, not a polished package.

## Requirements

- Windows
- Python 3.10+
- NVIDIA GPU with CUDA support
- ONNX Runtime GPU
- OpenCV
- NumPy
- MoveNet ONNX body model
- Hand pose ONNX model

## Installation

1. Create a virtual environment:

```powershell
python -m venv venv
```
Activate it:
```powershell
.\venv\Scripts\Activate.ps1
```
Install dependencies:
```powershell
pip install numpy opencv-python onnxruntime-gpu
```
Make sure ONNX Runtime CUDA dependencies are available:
```txt
CUDA 12.x
cuDNN 9.x
cuDNN bin directory on PATH
```
Example session-only PATH setup:

$env:Path = "C:\Program Files\NVIDIA\CUDNN\v9.21\bin\12.9\x64;" + $env:Path
Put the ONNX models in the project folder:
movenet.onnx
hand_pose.onnx
Usage
Edit the script and set your video path directly in code.

Example:

VIDEO_PATH = r"C:\path\to\your\video.mp4"
Then run:

python .\main.py
Annotated output is written to a hardcoded output file in the working directory.

Current Limitations
many values are hardcoded directly in the script
no config file
no CLI arguments
no automatic model download
no packaging
no robust multi-person support
hand landmarks may disappear at difficult angles
temporary persistence is used instead of true prediction/interpolation
designed around the current local setup only
Warning
Do not treat this as production-ready.

Before wider use, this should be cleaned up into:

configurable paths
configurable thresholds
proper logging
dependency checks
reusable modules
better hand tracking logic
documentation for model files and environment validation