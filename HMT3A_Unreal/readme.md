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

## Run

From this folder:

```bash
python main.py
```

Optional CLI flags:

```bash
python main.py --source 0 --max-persons 1
python main.py --source "C:\path\to\video.mp4" --no-preview
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
