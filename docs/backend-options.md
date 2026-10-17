# Landmark Backend Options

Kinara supports four body tracking backends and three hand tracking backends, allowing users to balance speed, hardware support, and multi-person capability:

### Body Backends

- `rtmpose`: RTMPose through `rtmlib`. This is the preferred CUDA body path for RTX systems (including RTX 50-series / `sm_120`), offering state-of-the-art real-time multi-person tracking and high COCO AP.
- `rtmpose-wholebody`: RTMPose WholeBody through `rtmlib`. A unified model that predicts both body and 21-point hand landmarks in a single pass; selecting this pairs body and hand detection automatically.
- `yolo`: Ultralytics YOLO pose models (`yolo11n-pose` through `yolo11x-pose`). Kept for baseline comparisons and multi-person ByteTrack/BoT-SORT integration.
- `mediapipe`: MediaPipe Pose. Suitable for local single-person tracking on CPU or systems without CUDA, with optional relative depth estimation.

### Hand Backends

- `onnx`: Dedicated ONNX-based YOLO26 hand-pose model (variants: `low`, `mid`, `high`, `max`). Runs on wrist-directed crops with retry scaling, palm validation, and physiological knuckle/lane constraints.
- `rtmpose-wholebody`: Unified WholeBody hand extraction with zero additional crop latency.
- `mediapipe`: MediaPipe Hands solution for 21-point hand tracking.

---

## Recommended Configurations

For RTX / CUDA RTMPose runs (preferred high-performance setup):

```powershell
kinara --landmark-backend rtmpose --rtmpose-device cuda --rtmpose-mode balanced --benchmark-frames 120 --no-preview
```

To let RTMPose WholeBody own both body and hands simultaneously:

```powershell
kinara --landmark-backend rtmpose-wholebody --rtmpose-device cuda --rtmpose-mode balanced --benchmark-frames 120 --no-preview
```

In the launcher, choosing `rtmpose-wholebody` disables separate body and hand backend controls because WholeBody is an integrated body+hand model.

For CPU-only fallback or diagnostic testing:

```powershell
kinara --landmark-backend rtmpose --rtmpose-device cpu --rtmpose-mode lightweight --benchmark-frames 30 --no-preview
```

For legacy YOLO pose comparisons:

```powershell
kinara --landmark-backend yolo --yolo-fast-preset nano --benchmark-frames 120 --no-preview
```

### Legacy YOLO Tradeoffs

- `--yolo-fast-preset nano`: Fastest, lowest body accuracy.
- `--yolo-fast-preset small`: Light, balanced speed/stability.
- `--yolo-fast-preset medium`: Balanced offline or GPU mode.
- `--yolo-fast-preset xlarge`: Highest YOLO accuracy, slowest.

### Hand Model Presets (`--hand-model-variant`)

- `low` / `mid`: YOLO26 hand-pose FP16 (~512px input), optimized for high frame-rate processing.
- `high` / `max`: YOLO26 hand-pose FP32 (~640px/768px input), highest landmark precision and finger articulation.

---

## Implementation Details

1. `body_backend="rtmpose"` uses `rtmlib` and returns Kinara `BodyDetection` records.
2. `body_backend="rtmpose-wholebody"` maps WholeBody output into standard Kinara body and hand payloads.
3. Existing Kinara smoothing, anatomical constraints, distance-invariant scale normalization, identity hints, multi-camera fusion, and export layers remain completely backend-agnostic.
4. Benchmark your pipeline speed using `--benchmark-frames N` and `--fps-log-interval S`.
