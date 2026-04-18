from __future__ import annotations

from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, PoseLandmarker, PoseLandmarkerOptions, RunningMode

from config import HAND_INDEX_BY_NAME, HAND_MODEL_PATH, LEFT_ELBOW_INDEX, LEFT_HIP_INDEX, LEFT_SHOULDER_INDEX, LEFT_WRIST_INDEX, POSE_MODEL_PATH, RIGHT_ELBOW_INDEX, RIGHT_HIP_INDEX, RIGHT_SHOULDER_INDEX, RIGHT_WRIST_INDEX
from pose_server.maskrcnn_segmenter import MaskRCNNPersonSegmenter
from pose_server.pose_detector import HAND_COLORS, PERSON_COLORS, PoseDetector as BasePoseDetector
from pose_server.yolo_person_detector import YOLOPersonDetector
from process.identity_memory import bbox_iou, estimate_pose_bbox, expand_bbox, extract_identity_features
from utils.math_utils import average, distance_2d


class PoseDetector(BasePoseDetector):
    def __init__(self, config):
        super().__init__(config)
        self._hand_side_memory: dict[int, dict] = {}
        self.pose_landmarker_image = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=config.min_pose_detection_confidence,
                min_pose_presence_confidence=config.min_pose_presence_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
            )
        )
        self.hand_landmarker_image = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
                running_mode=RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=config.min_hand_detection_confidence,
                min_hand_presence_confidence=config.min_hand_presence_confidence,
                min_tracking_confidence=config.min_hand_tracking_confidence,
            )
        )

        self.yolo_detector: YOLOPersonDetector | None = None
        self.mask_segmenter: MaskRCNNPersonSegmenter | None = None

        if self.config.enable_yolo_identity_assist:
            try:
                yolo_model_path = str(self.config.model_dir / self.config.yolo_model_name)
                self.yolo_detector = YOLOPersonDetector(yolo_model_path, self.config.yolo_person_confidence)
                print(f"YOLO assist enabled: {yolo_model_path}")
            except Exception as exc:
                print(f"YOLO assist disabled: {exc}")

        if self.yolo_detector is not None and self.config.enable_mask_rcnn_refinement:
            try:
                self.mask_segmenter = MaskRCNNPersonSegmenter(self.config.mask_rcnn_score_threshold)
                print("Mask R-CNN refinement enabled.")
            except Exception as exc:
                print(f"Mask R-CNN refinement disabled: {exc}")

    def detect(self, frame, timestamp_ms: int):
        if self.yolo_detector is not None:
            assisted = self._detect_with_person_assists(frame)
            if assisted is not None and assisted[0]:
                return assisted

        rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = self._detect_hands(rgb_frame, mp_image, pose_result, timestamp_ms)
        people = self._combine_results_with_metadata(
            pose_result,
            hand_result,
            frame,
            [{} for _ in list(pose_result.pose_landmarks or [])[: self.config.max_persons]],
        )
        rendered = frame.copy() if self.config.render_output else frame
        if self.config.render_output:
            self._draw_results(rendered, pose_result, hand_result)
        return people, rendered

    def _detect_with_person_assists(self, frame):
        if self.yolo_detector is None:
            return None

        detections = self.yolo_detector.detect(frame, max_people=self.config.max_persons)
        if not detections:
            return None

        frame_height, frame_width = frame.shape[:2]
        mask_candidates = self.mask_segmenter.detect(frame) if self.mask_segmenter is not None else []
        people = []
        draw_pose_images = []
        draw_hand_images = []
        draw_handedness = []

        for detection in detections:
            matched_mask = self._match_mask(mask_candidates, detection["bbox"])
            crop_bbox = expand_bbox(detection["bbox"], frame_width, frame_height, scale=0.12)
            crop = frame[crop_bbox["y0"]:crop_bbox["y1"], crop_bbox["x0"]:crop_bbox["x1"]].copy()
            if crop.size == 0:
                continue

            if matched_mask is not None:
                crop_mask = matched_mask[crop_bbox["y0"]:crop_bbox["y1"], crop_bbox["x0"]:crop_bbox["x1"]]
                if crop_mask.size:
                    crop[~crop_mask] = 0

            crop_rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crop_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
            crop_pose_result = self.pose_landmarker_image.detect(crop_mp_image)
            if not list(crop_pose_result.pose_landmarks or []):
                continue

            crop_hand_result = self._detect_hands_image(crop_rgb, crop_mp_image, crop_pose_result)
            mapped_pose_result = self._map_pose_result(crop_pose_result, crop_bbox, frame_width, frame_height)
            mapped_hand_result = self._map_hand_result(crop_hand_result, crop_bbox, frame_width, frame_height)
            batch = self._combine_results_with_metadata(
                mapped_pose_result,
                mapped_hand_result,
                frame,
                [{
                    "_bbox": detection["bbox"],
                    "_yolo_track_id": detection.get("track_id"),
                    "_detector_confidence": detection.get("confidence"),
                }],
            )
            if not batch:
                continue

            people.extend(batch[:1])
            draw_pose_images.extend(list(mapped_pose_result.pose_landmarks or []))
            draw_hand_images.extend(list(mapped_hand_result.hand_landmarks or []))
            draw_handedness.extend(list(mapped_hand_result.handedness or []))

        rendered = frame.copy() if self.config.render_output else frame
        if self.config.render_output:
            self._draw_results(
                rendered,
                SimpleNamespace(pose_landmarks=draw_pose_images),
                SimpleNamespace(hand_landmarks=draw_hand_images, handedness=draw_handedness),
            )

        return people, rendered

    def _match_mask(self, mask_candidates: list[dict], bbox: dict):
        best_index = None
        best_iou = 0.0
        for index, candidate in enumerate(mask_candidates):
            iou = bbox_iou(candidate.get("bbox"), bbox)
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is None or best_iou < 0.15:
            return None
        return mask_candidates.pop(best_index).get("mask")

    def _detect_hands_image(self, rgb_frame, full_mp_image, pose_result):
        if not self.config.enable_hand_roi:
            return self.hand_landmarker_image.detect(full_mp_image)

        frame_height, frame_width = rgb_frame.shape[:2]
        roi_records = self._build_hand_rois(pose_result, frame_width, frame_height)
        roi_results = []

        for roi_record in roi_records:
            x0, y0, x1, y1 = roi_record["bounds"]
            crop = np.ascontiguousarray(rgb_frame[y0:y1, x0:x1])
            if crop.size == 0:
                continue

            roi_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
            roi_result = self.hand_landmarker_roi.detect(roi_mp_image)
            selected = self._select_roi_detection(roi_result, roi_record["side"])
            if selected is None:
                continue

            roi_results.append(
                self._map_roi_detection(selected, x0, y0, x1 - x0, y1 - y0, frame_width, frame_height)
            )

        expected_count = len(roi_records)
        if not self.config.hand_roi_fallback_to_full_frame:
            return self._build_hand_result(roi_results)

        if expected_count == 0 or len(roi_results) < expected_count:
            full_result = self.hand_landmarker_image.detect(full_mp_image)
            return self._merge_hand_results(self._build_hand_result(roi_results), full_result)

        return self._build_hand_result(roi_results)

    def _map_pose_result(self, pose_result, bbox: dict, frame_width: int, frame_height: int):
        mapped_pose_images = []
        for pose_image in list(pose_result.pose_landmarks or [])[:1]:
            mapped_pose = []
            for landmark in pose_image:
                mapped_pose.append(
                    SimpleNamespace(
                        x=(bbox["x0"] + (float(landmark.x) * bbox["width"])) / frame_width,
                        y=(bbox["y0"] + (float(landmark.y) * bbox["height"])) / frame_height,
                        z=float(landmark.z),
                        visibility=float(getattr(landmark, "visibility", 1.0)),
                        presence=float(getattr(landmark, "presence", 1.0)),
                    )
                )
            mapped_pose_images.append(mapped_pose)

        return SimpleNamespace(
            pose_landmarks=mapped_pose_images,
            pose_world_landmarks=list(pose_result.pose_world_landmarks or [])[: len(mapped_pose_images)],
        )

    def _map_hand_result(self, hand_result, bbox: dict, frame_width: int, frame_height: int):
        mapped_entries = []
        hand_images = list(hand_result.hand_landmarks or [])
        hand_worlds = list(hand_result.hand_world_landmarks or [])
        handedness_sets = list(hand_result.handedness or [])

        for hand_index, hand_landmarks in enumerate(hand_images):
            mapped_hand = []
            for landmark in hand_landmarks:
                mapped_hand.append(
                    SimpleNamespace(
                        x=(bbox["x0"] + (float(landmark.x) * bbox["width"])) / frame_width,
                        y=(bbox["y0"] + (float(landmark.y) * bbox["height"])) / frame_height,
                        z=float(landmark.z),
                    )
                )
            mapped_entries.append(
                {
                    "hand_landmarks": mapped_hand,
                    "hand_world_landmarks": hand_worlds[hand_index] if hand_index < len(hand_worlds) else None,
                    "handedness": handedness_sets[hand_index] if hand_index < len(handedness_sets) else [],
                }
            )

        return self._build_hand_result(mapped_entries)

    def _combine_results_with_metadata(self, pose_result, hand_result, frame, person_metadata: list[dict]):
        pose_images = list(pose_result.pose_landmarks or [])
        pose_worlds = list(pose_result.pose_world_landmarks or [])
        frame_height, frame_width = frame.shape[:2]
        people = []

        for pose_index, pose_image in enumerate(pose_images[: self.config.max_persons]):
            metadata = person_metadata[pose_index] if pose_index < len(person_metadata) else {}
            pose_world = pose_worlds[pose_index] if pose_index < len(pose_worlds) else None
            bbox = metadata.get("_bbox") or estimate_pose_bbox(pose_image, frame_width, frame_height)
            appearance = extract_identity_features(frame, bbox, self.config.identity_profiles)
            people.append(
                {
                    "id": pose_index,
                    "body": self._extract_body(pose_world, pose_image),
                    "left_hand": self._empty_joint_map(HAND_INDEX_BY_NAME),
                    "right_hand": self._empty_joint_map(HAND_INDEX_BY_NAME),
                    "left_hand_confidence": None,
                    "right_hand_confidence": None,
                    "_pose_image": pose_image,
                    "_sort_x": self._pose_sort_key(pose_image),
                    "_left_distance": float("inf"),
                    "_right_distance": float("inf"),
                    "_bbox": bbox,
                    "_appearance": appearance,
                    "_yolo_track_id": metadata.get("_yolo_track_id"),
                    "_detector_confidence": metadata.get("_detector_confidence"),
                }
            )

        people.sort(key=lambda person: person["_sort_x"])
        for person_index, person in enumerate(people):
            person["id"] = person_index

        if people:
            self._attach_hands(people, hand_result)

        return [
            {
                "id": person["id"],
                "body": person["body"],
                "left_hand": person["left_hand"],
                "right_hand": person["right_hand"],
                "left_hand_confidence": person["left_hand_confidence"],
                "right_hand_confidence": person["right_hand_confidence"],
                "_bbox": person.get("_bbox"),
                "_appearance": person.get("_appearance"),
                "_yolo_track_id": person.get("_yolo_track_id"),
                "_detector_confidence": person.get("_detector_confidence"),
            }
            for person in people
        ]

    def _draw_boxes(self, frame, people: list[dict]) -> None:
        for person in people:
            bbox = person.get("_bbox")
            if bbox is None:
                continue
            color = PERSON_COLORS[int(person.get("id", 0)) % len(PERSON_COLORS)]
            cv2.rectangle(frame, (bbox["x0"], bbox["y0"]), (bbox["x1"], bbox["y1"]), color, 2)
            label_parts = [f"P{int(person.get('id', 0))}"]
            if person.get("_yolo_track_id") is not None:
                label_parts.append(f"Y{person['_yolo_track_id']}")
            top_color = ((((person.get("_appearance") or {}).get("regions") or {}).get("top") or {}).get("color"))
            if top_color:
                label_parts.append(top_color)
            cv2.putText(frame, " | ".join(label_parts), (bbox["x0"], max(18, bbox["y0"] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    def close(self) -> None:
        super().close()
        self.pose_landmarker_image.close()
        self.hand_landmarker_image.close()