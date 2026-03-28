# HMT3A Unreal v2

This folder contains a reworked version of the Unreal-side Python pipeline.

## What changed

- Keeps the original project untouched.
- Uses a fixed JSON packet schema for every frame.
- Separates `body`, `left_hand`, and `right_hand` data per person.
- Assigns hands to people using MediaPipe handedness plus wrist proximity.
- Adds smoothing to reduce landmark jitter.
- Adds bone vectors and joint angles to each packet for easier Unreal mapping.
- Uses safer input handling and cleanup in `main.py`.
- Stores downloaded MediaPipe task models in a local `models/` folder.
- Adds a 2-stage workflow with a preview pass first and an optional final render pass after confirmation.
- Supports both live webcam input and uploaded video files.
- Prompts for the number of people to track.
- Prompts for camera count and camera roles, with `front` as the default primary camera.
- Supports extra camera roles such as `back`, `right`, `left`, and `up`.
- Records raw live camera feeds during preview for later final rendering.
- Writes final offline renders into a separate `final_renders/` folder.
- Adds multi-camera fusion so one fused skeleton is streamed instead of using only the front camera.
- Uses config-driven rendering so only the required camera feed draws overlays.
- Adds wrist-guided hand ROI logic with support for fallback handling when needed.
- Supports manual FPS cap and manual resolution overrides.
- Uses source FPS by default, or the lowest FPS when multiple sources are used together.
- Uses native recorded resolution by default unless a manual resolution override is provided.
- Reduces preview overhead by removing duplicate smoothing and throttling console logging.

## Run

From this folder:

```bash
python main.py
```

Optional CLI flags:

```bash
python main.py --source 0 --max-persons 1
python main.py --source "C:\path\to\video.mp4" --no-preview
python main.py --source 0 --fps-cap 30 --width 1280 --height 720
```

## Packet shape

Each frame is streamed as JSON over UDP:

```json
{
  "frame": 12,
  "timestamp_ms": 410,
  "source_fps": 30.0,
  "count": 1,
  "persons": [
    {
      "id": 0,
      "body": {
        "present": true,
        "joints": {
          "left_shoulder": { "x": 0.0, "y": 0.0, "z": 0.0, "visibility": 1.0 }
        }
      },
      "left_hand": {
        "present": true,
        "confidence": 0.98,
        "joints": {
          "wrist": { "x": 0.0, "y": 0.0, "z": 0.0 }
        }
      },
      "right_hand": {
        "present": false,
        "confidence": null,
        "joints": {
          "wrist": null
        }
      },
      "bones": {},
      "angles": {}
    }
  ]
}
```

This makes it easier for an Unreal receiver to consume the data consistently.
