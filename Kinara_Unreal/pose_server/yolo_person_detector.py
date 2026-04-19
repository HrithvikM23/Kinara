from __future__ import annotations

from pathlib import Path

from process.identity_memory import build_bbox, clamp_bbox

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise RuntimeError("Ultralytics is not installed. Run 'pip install ultralytics'.") from exc


class YOLOPersonDetector:
    def __init__(self, model_path: str = "yolov8x.pt", confidence: float = 0.35, device: str = "cpu"):
        self.model: YOLO = YOLO(str(Path(model_path)))
        self.device = device
        self.model.to(self.device)
        self.confidence = float(confidence)

    def detect(self, frame, max_people: int) -> list[dict]:
        results = self.model.track(
            source=frame,
            device=self.device,
            conf=self.confidence,
            classes=[0],
            persist=True,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []

        frame_height, frame_width = frame.shape[:2]
        xyxy_items = boxes.xyxy.tolist()
        confidence_items = boxes.conf.tolist() if boxes.conf is not None else [1.0] * len(xyxy_items)
        raw_ids = boxes.id.tolist() if boxes.id is not None else []

        detections = []
        for index, coords in enumerate(xyxy_items):
            bbox = clamp_bbox(
                build_bbox(
                    int(coords[0]),
                    int(coords[1]),
                    int(coords[2]),
                    int(coords[3]),
                ),
                frame_width,
                frame_height,
            )
            if bbox is None:
                continue

            raw_id = raw_ids[index] if index < len(raw_ids) else None
            track_id = int(raw_id) if raw_id is not None else None

            detections.append(
                {
                    "bbox": bbox,
                    "confidence": round(float(confidence_items[index]), 4),
                    "track_id": track_id,
                }
            )

        detections.sort(key=lambda item: (item["confidence"], item["bbox"]["height"] * item["bbox"]["width"]), reverse=True)
        return detections[: max(int(max_people), 1)]
