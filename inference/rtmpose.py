from __future__ import annotations

import cv2
import numpy as np
from typing import Any, TypedDict, cast

try:
    import onnxruntime as ort
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "onnxruntime is not installed. Install `onnxruntime-gpu` for CUDA inference "
        "or `onnxruntime` for CPU-only inference, then run the app again."
    ) from exc

try:
    from ultralytics import YOLO
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "ultralytics is not installed. Install `ultralytics` and a CUDA-enabled PyTorch build, then run the app again."
    ) from exc


class BodyDetection(TypedDict):
    id: int | None
    score: float
    box: tuple[int, int, int, int]
    body_points: list[tuple[int, int, float]]


class ONNXPoseHandRunner:
    def __init__(self, config):
        self.config = config
        self.body_model = YOLO(str(config.body_model_path))
        self.hand_session = ort.InferenceSession(
            str(config.hand_model_path),
            providers=list(config.provider_names),
        )

    def detect_body(self, frame_bgr):
        detections = self.detect_bodies(frame_bgr, max_people=1, track=False)
        if not detections:
            return [(0, 0, 0.0) for _ in range(17)]
        return cast(list[tuple[int, int, float]], detections[0]["body_points"])

    @staticmethod
    def _to_numpy(value: Any, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray[Any, Any]:
        if value is None:
            return np.empty(shape, dtype=dtype)
        tensor_like = cast(Any, value)
        if hasattr(tensor_like, "cpu"):
            tensor_like = tensor_like.cpu()
        if hasattr(tensor_like, "numpy"):
            tensor_like = tensor_like.numpy()
        return np.asarray(tensor_like, dtype=dtype)

    def detect_bodies(self, frame_bgr, max_people: int, track: bool):
        if track:
            results = self.body_model.track(
                frame_bgr,
                conf=self.config.body_conf_threshold,
                iou=self.config.body_iou_threshold,
                imgsz=self.config.body_input_size,
                max_det=max_people,
                persist=True,
                verbose=False,
                tracker=self.config.yolo_tracker,
                device=self.config.yolo_device,
            )
        else:
            results = self.body_model.predict(
                frame_bgr,
                conf=self.config.body_conf_threshold,
                iou=self.config.body_iou_threshold,
                imgsz=self.config.body_input_size,
                max_det=max_people,
                verbose=False,
                device=self.config.yolo_device,
            )

        if not results:
            return []

        result = results[0]
        if result.boxes is None or result.keypoints is None:
            return []

        boxes_xyxy = self._to_numpy(result.boxes.xyxy, (0, 4), np.dtype(np.float32))
        boxes_conf = self._to_numpy(result.boxes.conf, (0,), np.dtype(np.float32))
        boxes_id = None if result.boxes.id is None else self._to_numpy(result.boxes.id, (0,), np.dtype(np.float32))
        keypoints_data = self._to_numpy(result.keypoints.data, (0, 17, 3), np.dtype(np.float32))

        detections: list[BodyDetection] = []
        for index in range(min(len(boxes_xyxy), len(keypoints_data))):
            box = boxes_xyxy[index]
            keypoints = keypoints_data[index]
            detection_id = None if boxes_id is None else int(float(boxes_id[index]))
            detection_score = float(boxes_conf[index])
            body_points: list[tuple[int, int, float]] = []
            for point in keypoints:
                point_x = float(point[0])
                point_y = float(point[1])
                point_conf = float(point[2])
                body_points.append((int(round(point_x)), int(round(point_y)), point_conf))
            detections.append(
                {
                    "id": detection_id,
                    "score": detection_score,
                    "box": (
                        int(round(float(box[0]))),
                        int(round(float(box[1]))),
                        int(round(float(box[2]))),
                        int(round(float(box[3]))),
                    ),
                    "body_points": body_points,
                }
            )
        detections.sort(key=lambda item: item["score"], reverse=True)
        return detections[:max_people]

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
