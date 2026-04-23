from __future__ import annotations

import cv2

from config import BODY_EDGES, BODY_KEYPOINTS, HAND_EDGES, WRIST_TO_ELBOW
from utils.normalize import build_hand_box


class PoseHandPipeline:
    def __init__(self, config, runner, smoother, osc_sender):
        self.config = config
        self.runner = runner
        self.smoother = smoother
        self.osc_sender = osc_sender

    def process_frame(self, frame):
        frame_height, frame_width = frame.shape[:2]
        body_points = self.smoother.smooth_body(self.runner.detect_body(frame))
        hands_by_side = {}

        for wrist_idx, elbow_idx in WRIST_TO_ELBOW.items():
            wrist_point = body_points[wrist_idx]
            elbow_point = body_points[elbow_idx]

            if wrist_point[2] <= self.config.body_conf_threshold or elbow_point[2] <= self.config.body_conf_threshold:
                continue

            side = "left" if wrist_idx == 9 else "right"
            box = build_hand_box(
                wrist_point,
                elbow_point,
                frame_width,
                frame_height,
                self.config.hand_box_min_size,
                self.config.hand_box_scale,
            )

            hand_points = self.runner.detect_hand(frame, box)
            hand_points = self.smoother.smooth_hand(side, hand_points)
            if hand_points is not None:
                hands_by_side[side] = {"box": box, "points": hand_points}

        self._draw_body(frame, body_points)
        self._draw_hands(frame, hands_by_side)
        self.osc_sender.send_pose(body_points, hands_by_side)
        return frame

    def _draw_body(self, frame, body_points) -> None:
        for start_idx, end_idx in BODY_EDGES:
            x1, y1, c1 = body_points[start_idx]
            x2, y2, c2 = body_points[end_idx]
            if c1 > self.config.body_conf_threshold and c2 > self.config.body_conf_threshold:
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        for idx, (px, py, conf) in enumerate(body_points):
            if idx in BODY_KEYPOINTS and conf > self.config.body_conf_threshold:
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

    def _draw_hands(self, frame, hands_by_side) -> None:
        for hand_payload in hands_by_side.values():
            x1, y1, x2, y2 = hand_payload["box"]
            hand_points = hand_payload["points"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 255), 1)

            for start_idx, end_idx in HAND_EDGES:
                x1p, y1p, c1 = hand_points[start_idx]
                x2p, y2p, c2 = hand_points[end_idx]
                if c1 > self.config.hand_kp_threshold and c2 > self.config.hand_kp_threshold:
                    cv2.line(frame, (x1p, y1p), (x2p, y2p), (0, 255, 255), 2)

            for px, py, conf in hand_points:
                if conf > self.config.hand_kp_threshold:
                    cv2.circle(frame, (px, py), 3, (0, 165, 255), -1)
