from __future__ import annotations

from dataclasses import dataclass, field

from config import BODY_LANDMARKS, HAND_LANDMARKS, PRIMARY_CAMERA_ROLE
from process.person_tracker import PersonTracker
from utils.math_utils import average_points, distance_3d, transform_point
from utils.smoothing import LandmarkSmoother


TORSO_JOINTS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


@dataclass(slots=True)
class Observation:
    role: str
    body: dict
    left_hand: dict
    right_hand: dict
    left_hand_confidence: float | None
    right_hand_confidence: float | None
    anchor: dict | None
    confidence: float


@dataclass(slots=True)
class ObservationCluster:
    observations: list[Observation] = field(default_factory=list)
    anchor: dict | None = None
    roles: set[str] = field(default_factory=set)


class MultiCameraFusion:
    def __init__(self, config, primary_role: str = PRIMARY_CAMERA_ROLE):
        self.config = config
        self.primary_role = primary_role
        self.smoother = LandmarkSmoother(config)
        self.tracker = PersonTracker(
            max_match_distance=config.track_match_distance,
            max_missed_frames=config.max_track_missed_frames,
        )

    def fuse_frame(self, detections_by_role: dict[str, list[dict]]) -> list[dict]:
        observations = []
        for role, people in detections_by_role.items():
            observations.extend(self._prepare_observations(role, people))

        clusters = self._cluster_observations(observations)
        fused_people = []
        for cluster in clusters:
            fused_person = self._fuse_cluster(cluster)
            if fused_person is not None:
                fused_people.append(fused_person)

        tracked_people = self.tracker.update(fused_people)
        smoothed_people = self.smoother.smooth_people(tracked_people)
        for person in smoothed_people:
            person.pop("_anchor", None)
        return smoothed_people

    def _prepare_observations(self, role: str, people: list[dict]) -> list[Observation]:
        observations = []
        for person in people:
            transformed_body = self._transform_section(role, person.get("body", {}))
            transformed_left = self._align_hand_to_body(
                transformed_body.get("left_wrist"),
                self._transform_section(role, person.get("left_hand", {})),
            )
            transformed_right = self._align_hand_to_body(
                transformed_body.get("right_wrist"),
                self._transform_section(role, person.get("right_hand", {})),
            )
            anchor = self._build_anchor(transformed_body)
            observations.append(
                Observation(
                    role=role,
                    body=transformed_body,
                    left_hand=transformed_left,
                    right_hand=transformed_right,
                    left_hand_confidence=self._optional_float(person.get("left_hand_confidence")),
                    right_hand_confidence=self._optional_float(person.get("right_hand_confidence")),
                    anchor=anchor,
                    confidence=self._body_confidence(transformed_body, role),
                )
            )
        return observations

    def _cluster_observations(self, observations: list[Observation]) -> list[ObservationCluster]:
        sorted_observations = sorted(observations, key=lambda observation: observation.confidence, reverse=True)
        clusters: list[ObservationCluster] = []

        for observation in sorted_observations:
            if observation.anchor is None:
                clusters.append(ObservationCluster(observations=[observation], anchor=None, roles={observation.role}))
                continue

            best_cluster = None
            best_distance = None
            for cluster in clusters:
                if cluster.anchor is None:
                    continue
                if observation.role in cluster.roles:
                    continue
                distance = distance_3d(observation.anchor, cluster.anchor)
                if distance > float(self.config.cluster_match_distance):
                    continue
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_cluster = cluster

            if best_cluster is None:
                clusters.append(
                    ObservationCluster(
                        observations=[observation],
                        anchor=dict(observation.anchor),
                        roles={observation.role},
                    )
                )
                continue

            best_cluster.observations.append(observation)
            best_cluster.roles.add(observation.role)
            best_cluster.anchor = average_points(
                [candidate.anchor for candidate in best_cluster.observations if candidate.anchor is not None]
            )

        return clusters

    def _fuse_cluster(self, cluster: ObservationCluster):
        if not cluster.observations:
            return None

        fused_body = {
            joint_name: self._fuse_joint(
                [
                    self._candidate(observation, observation.body.get(joint_name), observation.role, "body")
                    for observation in cluster.observations
                ]
            )
            for joint_name in BODY_LANDMARKS
        }

        fused_left = {
            joint_name: self._fuse_joint(
                [
                    self._candidate(observation, observation.left_hand.get(joint_name), observation.role, "left_hand")
                    for observation in cluster.observations
                ]
            )
            for joint_name in HAND_LANDMARKS
        }
        fused_right = {
            joint_name: self._fuse_joint(
                [
                    self._candidate(observation, observation.right_hand.get(joint_name), observation.role, "right_hand")
                    for observation in cluster.observations
                ]
            )
            for joint_name in HAND_LANDMARKS
        }

        anchor = self._build_anchor(fused_body)
        if anchor is None:
            anchor = cluster.anchor

        return {
            "id": -1,
            "body": fused_body,
            "left_hand": fused_left,
            "right_hand": fused_right,
            "left_hand_confidence": self._average_optional([observation.left_hand_confidence for observation in cluster.observations]),
            "right_hand_confidence": self._average_optional([observation.right_hand_confidence for observation in cluster.observations]),
            "_anchor": anchor,
        }

    def _candidate(self, observation: Observation, joint, role: str, section_name: str):
        if joint is None:
            return None

        calibration = self.config.calibrations.get(role)
        role_weight = float(calibration.confidence_weight) if calibration is not None else 1.0
        if role == self.primary_role:
            role_weight *= 1.05

        if section_name == "body":
            base_weight = float(joint.get("visibility", 0.5) or 0.0)
            weight = max(base_weight, 0.05) * role_weight
        elif section_name == "left_hand":
            weight = max(float(observation.left_hand_confidence or 0.0), 0.12) * role_weight
        else:
            weight = max(float(observation.right_hand_confidence or 0.0), 0.12) * role_weight

        return {"joint": joint, "weight": weight}

    def _fuse_joint(self, candidates):
        candidates = [candidate for candidate in candidates if candidate is not None]
        if not candidates:
            return None

        points = [candidate["joint"] for candidate in candidates]
        weights = [candidate["weight"] for candidate in candidates]
        fused = average_points(points, weights=weights)
        if fused is None:
            return None

        visibility_pairs = [
            (float(candidate["joint"].get("visibility", 0.0)), float(candidate["weight"]))
            for candidate in candidates
            if "visibility" in candidate["joint"]
        ]
        if visibility_pairs:
            visibility_weight_total = sum(weight for _, weight in visibility_pairs)
            fused["visibility"] = round(
                sum(value * weight for value, weight in visibility_pairs) / max(visibility_weight_total, 1e-8),
                6,
            )
        return fused

    def _transform_section(self, role: str, joints: dict) -> dict:
        transformed = {}
        for joint_name, joint in (joints or {}).items():
            transformed[joint_name] = self._transform_joint(role, joint)
        return transformed

    def _transform_joint(self, role: str, joint):
        if joint is None:
            return None

        calibration = self.config.calibrations.get(role)
        if calibration is None:
            return dict(joint)

        transformed = transform_point(
            joint,
            rotation_deg=calibration.rotation_deg,
            translation=calibration.translation,
            scale=calibration.scale,
        )
        for key, value in joint.items():
            if key not in transformed:
                transformed[key] = value
        return transformed

    def _align_hand_to_body(self, wrist_joint, hand_section: dict) -> dict:
        aligned = {}
        hand_wrist = hand_section.get("wrist")
        for joint_name, joint in (hand_section or {}).items():
            if joint is None:
                aligned[joint_name] = None
                continue
            if joint_name == "wrist":
                aligned[joint_name] = wrist_joint if wrist_joint is not None else joint
                continue
            if wrist_joint is None or hand_wrist is None:
                aligned[joint_name] = joint
                continue

            aligned[joint_name] = {
                "x": round(float(wrist_joint["x"]) + (float(joint["x"]) - float(hand_wrist["x"])), 6),
                "y": round(float(wrist_joint["y"]) + (float(joint["y"]) - float(hand_wrist["y"])), 6),
                "z": round(float(wrist_joint["z"]) + (float(joint["z"]) - float(hand_wrist["z"])), 6),
            }
        return aligned

    def _build_anchor(self, body: dict):
        points = [body.get(joint_name) for joint_name in TORSO_JOINTS if body.get(joint_name) is not None]
        if not points:
            return None
        return average_points(points)

    def _body_confidence(self, body: dict, role: str) -> float:
        visibility_values = [float(body.get(joint_name, {}).get("visibility", 0.0)) for joint_name in TORSO_JOINTS if body.get(joint_name) is not None]
        base_confidence = sum(visibility_values) / len(visibility_values) if visibility_values else 0.0
        calibration = self.config.calibrations.get(role)
        if calibration is not None:
            base_confidence *= float(calibration.confidence_weight)
        if role == self.primary_role:
            base_confidence *= 1.05
        return base_confidence

    def _average_optional(self, values):
        values = [float(value) for value in values if value is not None]
        if not values:
            return None
        return round(sum(values) / len(values), 6)

    def _optional_float(self, value):
        if value is None:
            return None
        return float(value)

