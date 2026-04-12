KINARA PIPELINE — CLI ARGUMENTS REFERENCE
==========================================

INPUT
-----
--source                  Webcam index (e.g. 0) or path to a video file.
                          If omitted, the interactive source prompt runs.

--max-persons             Integer >= 1.

NETWORK
-------
--udp-ip                  Any valid IP address string.
--udp-port                Integer, valid port number.

SMOOTHING
---------
--smoothing-alpha         Float, 0.0 – 1.0.
                          Higher = smoother but laggier.
                          Body alpha clamped to [0.05, 0.95].
                          Hand alpha set to 75% of this, also clamped to [0.05, 0.95].

MEDIAPIPE CONFIDENCE
--------------------
--body-detection-confidence    Float, 0.0 – 1.0. Default: 0.7
--body-presence-confidence     Float, 0.0 – 1.0. Default: 0.7
--body-tracking-confidence     Float, 0.0 – 1.0. Default: 0.8
--hand-detection-confidence    Float, 0.0 – 1.0. Default: 0.7
--hand-presence-confidence     Float, 0.0 – 1.0. Default: 0.7
--hand-tracking-confidence     Float, 0.0 – 1.0. Default: 0.7

YOLO
----
--no-yolo                 Flag. No value required.
                          Automatically disabled when max-persons is 1.

--yolo-model              Model name or local file path. Default: yolov8x.pt

                          yolov8n.pt   Nano    — fastest, least accurate, low VRAM
                          yolov8s.pt   Small   — faster, moderate accuracy
                          yolov8m.pt   Medium  — balanced speed and accuracy
                          yolov8l.pt   Large   — accurate, slower
                          yolov8x.pt   XLarge  — most accurate, highest VRAM (default)

--yolo-confidence         Float, 0.0 – 1.0. Default: 0.35
                          Lower = detects more people but more false positives.
                          Higher = stricter, may miss partially visible people.

MASK R-CNN
----------
--no-mask-rcnn            Flag. No value required. Requires YOLO to be active.
--mask-rcnn-score         Float, 0.0 – 1.0. Default: 0.5
--rcnn-confidence         Alias for --mask-rcnn-score.

IDENTITY MEMORY
---------------
--no-identity-memory      Flag. No value required.
                          Automatically disabled when max-persons is 1.

OUTPUT CONTROL
--------------
--no-preview              Flag. No value required.
--no-record               Flag. No value required.
--no-motion-export        Flag. Disables all of JSON, BVH, and FBX export.
--no-json-export          Flag. No value required.
--no-bvh-export           Flag. No value required.
--no-fbx-export           Flag. No value required.

PERFORMANCE
-----------
--fps-cap                 Float, >= 1.0.
                          If omitted, the interactive FPS prompt runs.

--preview-fps             Float, >= 1.0.
                          Capped internally to the session source FPS.
                          Does not affect final render FPS.

RESOLUTION
----------
--width                   Integer >= 1. Must be paired with --height.
--height                  Integer >= 1. Must be paired with --width.
                          If omitted, the interactive resolution prompt runs.

CALIBRATION
-----------
--calibration-file        Path to a JSON file defining per-role camera calibration.
                          Roles: front, back, left, right, up.
                          Each entry accepts: rotation_deg [x,y,z], translation [x,y,z],
                          scale (float), confidence_weight (float, 0.0 – 1.0).