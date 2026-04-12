from __future__ import annotations

import cv2
import numpy as np

from process.identity_memory import build_bbox, clamp_bbox


class MaskRCNNPersonSegmenter:
    def __init__(self, score_threshold: float = 0.5):
        try:
            import torch
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
        except ImportError as exc:
            raise RuntimeError("torch and torchvision are required for Mask R-CNN refinement.") from exc

        self.torch = torch
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn(weights=weights)
        self.model.eval()
        self.score_threshold = float(score_threshold)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def detect(self, frame) -> list[dict]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb_frame).permute(2, 0, 1).float().div(255.0).to(self.device)
        with self.torch.inference_mode():
            prediction = self.model([tensor])[0]

        frame_height, frame_width = frame.shape[:2]
        boxes = prediction.get("boxes")
        labels = prediction.get("labels")
        scores = prediction.get("scores")
        masks = prediction.get("masks")
        if boxes is None or labels is None or scores is None or masks is None:
            return []

        detections = []
        for box, label, score, mask in zip(boxes, labels, scores, masks):
            if int(label.item()) != 1:
                continue
            score_value = float(score.item())
            if score_value < self.score_threshold:
                continue

            coords = box.detach().cpu().tolist()
            bbox = clamp_bbox(
                build_bbox(int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])),
                frame_width,
                frame_height,
            )
            if bbox is None:
                continue

            mask_array = mask[0].detach().cpu().numpy()
            detections.append(
                {
                    "bbox": bbox,
                    "score": round(score_value, 4),
                    "mask": mask_array >= 0.5,
                }
            )

        detections.sort(key=lambda item: item["score"], reverse=True)
        return detections

