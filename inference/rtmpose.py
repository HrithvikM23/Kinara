from __future__ import annotations

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "onnxruntime is not installed. Install `onnxruntime-gpu` for CUDA inference "
        "or `onnxruntime` for CPU-only inference, then run the app again."
    ) from exc


class ONNXPoseHandRunner:
    def __init__(self, config):
        self.config = config
        self.body_session = ort.InferenceSession(
            str(config.body_model_path),
            providers=list(config.provider_names),
        )
        self.hand_session = ort.InferenceSession(
            str(config.hand_model_path),
            providers=list(config.provider_names),
        )

    def detect_body(self, frame_bgr):
        frame_height, frame_width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self.config.body_input_size, self.config.body_input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        body_input = np.expand_dims(resized, axis=0).astype(np.dtype(self.config.body_input_dtype))
        outputs = self.body_session.run(None, {self.config.body_input_name: body_input})
        keypoints = np.asarray(outputs[0], dtype=np.float32)[0, 0]

        points = []
        for y, x, conf in keypoints:
            px = int(x * frame_width)
            py = int(y * frame_height)
            points.append((px, py, float(conf)))
        return points

    def detect_hand(self, frame_bgr, box):
        x1, y1, x2, y2 = box
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            crop_rgb,
            (self.config.hand_input_size, self.config.hand_input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        hand_input = resized.astype(np.float32) / 255.0
        hand_input = np.transpose(hand_input, (2, 0, 1))
        hand_input = np.expand_dims(hand_input, axis=0)

        outputs = self.hand_session.run(None, {self.config.hand_input_name: hand_input})
        detections = np.asarray(outputs[0], dtype=np.float32)[0]

        best = detections[np.argmax(detections[:, 4])]
        if float(best[4]) <= self.config.hand_det_threshold:
            return None

        crop_w = x2 - x1
        crop_h = y2 - y1
        points = []
        for i in range(21):
            base = 6 + i * 3
            x = float(best[base])
            y = float(best[base + 1])
            conf = float(best[base + 2])

            px = x1 + int((x / self.config.hand_input_size) * crop_w)
            py = y1 + int((y / self.config.hand_input_size) * crop_h)
            points.append((px, py, conf))

        return points
