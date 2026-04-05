# HMT3A Unreal Client Pipeline

This folder contains the client-side motion pipeline that prepares fused tracking data for downstream animation systems.

## What is implemented

- Webcam and video-file capture
- Multi-camera role assignment with optional calibration JSON
- Body and hand landmark detection
- Wrist-guided hand ROI processing
- Calibration-aware multi-camera fusion into a shared space
- Persistent multi-person IDs across frames
- Adaptive smoothing with stronger hand jitter reduction
- Hierarchical skeleton reconstruction for body and hands
- Joint rotation solving with quaternion and Euler output
- Joint-angle extraction for major limbs and finger curls
- Preview pass plus final offline render pass
- UDP packet streaming
- Reusable motion export to JSON, BVH, and FBX

## Run

From this folder:

```bash
python main.py
```

Optional examples:

```bash
python main.py --source 0 --max-persons 1
python main.py --source "C:\path\to\video.mp4" --no-preview
python main.py --source 0 --fps-cap 30 --width 1280 --height 720
python main.py --source 0 --calibration-file "C:\path\to\camera_calibration.json"
python main.py --source 0 --no-fbx-export
```

## New CLI flags

- `--calibration-file` loads per-role extrinsic calibration data from JSON
- `--no-motion-export` disables all fused motion file export
- `--no-json-export` disables JSON motion export only
- `--no-bvh-export` disables BVH animation export only
- `--no-fbx-export` disables FBX animation export only

## Motion outputs

During final render the pipeline now writes:

- processed preview/final videos into `outputs/final_renders/`
- reusable motion data into `outputs/motion_exports/`

Motion export files include:

- fused frame JSON
- one BVH file per tracked person
- one FBX file per tracked person

FBX export runs through Blender in background when Blender is available on the machine.

## Calibration JSON shape

A calibration file is a JSON object keyed by role name.

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

## Packet shape

Each streamed person now contains:

- `body`
- `left_hand`
- `right_hand`
- `bones`
- `angles`
- `skeleton`
- `rotations`

This makes the client output suitable for live preview, offline cleanup, and non-Unreal animation export.
