# Pose and Hand Landmark Pipeline Arguments

## General

`--source`  
Function: Selects the input source.  
Accepted values: webcam index like `0`, `1`, `2` or a path to a video file.

`--output`  
Function: Sets the output video name template.  
Accepted values: any writable video path.  
Notes: The actual saved file is stacked as `"<name> rendered-N.<ext>"`.

`--output-dir`  
Function: Sets the directory where rendered, fbx, and json outputs are written.  
Accepted values: any writable directory path.

`--output-basename`  
Function: Sets the shared filename prefix for rendered, fbx, and json outputs.  
Accepted values: any non-empty text string.

## Model Selection

`--model`  
Function: Selects the body pose model file used by Ultralytics YOLO.  
Accepted values: a model filename like `yolo11x-pose.pt`, `yolo11l-pose.pt`, `yolo11m-pose.pt`, `yolo11s-pose.pt`, `yolo11n-pose.pt`, or a custom local path.  
Default: `yolo11x-pose.pt`  
Notes: If you pass one of the known YOLO filenames and it is missing, it is downloaded into `models/body/`.

`--hand-model-variant`  
Function: Selects the hand model preset and auto-download target.  
Accepted variants: `low`, `mid`, `high`, `max`.  
Current mapping:  
`low`, `mid` -> YOLO26 hand pose FP16  
`high`, `max` -> YOLO26 hand pose FP32  
Default: `max`

`--hand-model`  
Function: Uses a specific ONNX hand model file instead of the preset downloader.  
Accepted values: path to an ONNX file.

## Model Runtime Settings

`--body-input-name`  
Function: Reserved body input setting kept in config for compatibility.  
Accepted values: any text string.  
Default: `input`

`--hand-input-name`  
Function: Sets the ONNX input tensor name for the hand model.  
Accepted values: any valid ONNX input name string.  
Default: `images`

`--body-input-size`  
Function: Sets the YOLO body model image size.  
Accepted range: integer `> 0`.  
Default: `960`

`--hand-input-size`  
Function: Sets the square resize dimension for the hand crop input.  
Accepted range: integer `> 0`.  
Default: `640`

## Detection Thresholds

`--body-conf-threshold`  
Function: Minimum YOLO body confidence used to keep body landmarks.  
Accepted range: float in `(0, 1]`.  
Default: `0.30`

`--body-iou-threshold`  
Function: YOLO body NMS IoU threshold.  
Accepted range: float in `(0, 1]`.  
Default: `0.45`

`--hand-det-threshold`  
Function: Minimum confidence needed to keep a hand detection candidate.  
Accepted range: float in `(0, 1]`.  
Default: `0.15`

`--hand-kp-threshold`  
Function: Minimum confidence needed to draw hand keypoints and hand skeleton links.  
Accepted range: float in `(0, 1]`.  
Default: `0.20`

## Hand Crop Settings

`--hand-box-min-size`  
Function: Minimum side length of the wrist-centered hand crop box.  
Accepted range: integer `> 0`.  
Default: `160`

`--hand-box-scale`  
Function: Scales the wrist-elbow distance to form the hand crop size.  
Accepted range: float `> 0`.  
Default: `2.0`

## Hand Model Providers

`--provider`  
Function: Adds an ONNX Runtime execution provider in priority order for the hand model.  
Accepted values: ONNX Runtime provider names like `CUDAExecutionProvider` or `CPUExecutionProvider`.  
Usage: repeat the argument to set fallback order.  
Default: `CUDAExecutionProvider`

## Multi-Person Tracking

`--max-people`  
Function: Enables multi-person tracking for a single camera or video source and sets the maximum number of tracked people.  
Accepted range: integer `> 0`.  
Default: `1`

`--identity personN=color1,color2`  
Function: Provides optional clothing color hints to keep person IDs stable when people cross or overlap.  
Accepted values: labels like `person1`, `person2` with one or more color names such as `black`, `orange`, `blue`, `gray`, `silver`, `red`, `green`, `yellow`, `purple`, `pink`, `brown`, `white`.  
Usage: repeat for multiple people.

`--person-detector-scale`  
Function: Deprecated compatibility setting kept in config. It is currently unused by the YOLO multi-person path.  
Accepted range: float `> 1.0`.  
Default: `1.05`

`--person-box-scale`  
Function: Expands each detected person box before running hand inference so limbs near the box edges are less likely to be cut off.  
Accepted range: float `> 0`.  
Default: `1.15`

`--person-track-hold-frames`  
Function: Keeps a person track alive briefly when detections disappear for a few frames during crossings or occlusions.  
Accepted range: integer `> 0`.  
Default: `10`

`--person-match-threshold`  
Function: Minimum association score when matching a detected person to an existing track.  
Accepted range: float `> 0`.  
Default: `0.15`

`--person-cross-wrist-ratio`  
Function: Hand ownership switch ratio during overlaps. Lower values are stricter and more likely to reject a stolen hand.  
Accepted range: float in `(0, 1]`.  
Default: `0.90`

`--yolo-tracker`  
Function: Sets the Ultralytics tracker config used for multi-person tracking.  
Accepted values: tracker config names like `botsort.yaml` or `bytetrack.yaml`.  
Default: `botsort.yaml`

`--yolo-device`  
Function: Overrides the Ultralytics device selection.  
Accepted values: values like `0`, `cpu`, `cuda:0`.  
Default: automatic

## Live UDP Output

`--osc-host`  
Function: Sets the live UDP target host.  
Accepted values: hostname or IP address.  
Default: `127.0.0.1`

`--osc-port`  
Function: Sets the live UDP target port.  
Accepted range: integer from `1` to `65535`.  
Default: `9000`

`--osc-enabled`  
Function: Enables live UDP sending.  
Accepted values: flag only.  
Default: disabled

## Preview and Video Writer

`--preview-title`  
Function: Sets the preview window title.  
Accepted values: any text string.  
Default: `Pose + Hand Landmarks`

`--fallback-fps`  
Function: FPS used when the video source reports `0` or invalid FPS.  
Accepted range: float `> 0`.  
Default: `30.0`

`--output-fourcc`  
Function: Sets the video writer codec code.  
Accepted values: text string with at least 4 characters. First 4 are used.  
Default: `mp4v`

## Drawing Colors

`--body-line-color`  
Function: Sets body skeleton line color.  
Accepted values: `B,G,R` integers.  
Accepted range: each channel `0` to `255`.  
Default: `255,0,0`

`--body-point-color`  
Function: Sets body landmark point color.  
Accepted values: `B,G,R` integers.  
Accepted range: each channel `0` to `255`.  
Default: `0,255,0`

`--hand-box-color`  
Function: Sets hand crop rectangle color.  
Accepted values: `B,G,R` integers.  
Accepted range: each channel `0` to `255`.  
Default: `80,80,255`

`--hand-line-color`  
Function: Sets hand skeleton line color.  
Accepted values: `B,G,R` integers.  
Accepted range: each channel `0` to `255`.  
Default: `0,255,255`

`--hand-point-color`  
Function: Sets hand landmark point color.  
Accepted values: `B,G,R` integers.  
Accepted range: each channel `0` to `255`.  
Default: `0,165,255`

## Drawing Sizes

`--body-line-thickness`  
Function: Sets body skeleton line thickness.  
Accepted range: integer `> 0`.  
Default: `2`

`--body-point-radius`  
Function: Sets body landmark point radius.  
Accepted range: integer `> 0`.  
Default: `4`

`--hand-box-thickness`  
Function: Sets hand crop rectangle thickness.  
Accepted range: integer `> 0`.  
Default: `1`

`--hand-line-thickness`  
Function: Sets hand skeleton line thickness.  
Accepted range: integer `> 0`.  
Default: `2`

`--hand-point-radius`  
Function: Sets hand landmark point radius.  
Accepted range: integer `> 0`.  
Default: `3`

## Temporal Stability

`--body-smoothing-alpha`  
Function: EMA smoothing factor for body landmarks. Higher values follow fresh detections more closely, lower values make motion steadier.  
Accepted range: float in `(0, 1]`.  
Default: `0.65`

`--hand-smoothing-alpha`  
Function: EMA smoothing factor for hand landmarks. Higher values follow fresh detections more closely, lower values make motion steadier.  
Accepted range: float in `(0, 1]`.  
Default: `0.55`

`--body-hold-frames`  
Function: Number of frames to keep the last valid body landmark before dropping it when detection confidence disappears.  
Accepted range: integer `> 0`.  
Default: `8`

`--hand-hold-frames`  
Function: Number of frames to keep the last valid hand landmark before dropping it when detection confidence disappears.  
Accepted range: integer `> 0`.  
Default: `6`

`--hold-confidence-decay`  
Function: Confidence multiplier applied each frame while a held landmark is being reused. Lower values fade held joints out faster.  
Accepted range: float in `(0, 1]`.  
Default: `0.85`

## Preview Toggle

`--no-preview`  
Function: Disables the live OpenCV preview window.  
Accepted values: flag only.  
Default: preview enabled
