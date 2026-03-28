from __future__ import annotations

from dataclasses import dataclass

from config import BODY_LANDMARKS, HAND_LANDMARKS, PRIMARY_CAMERA_ROLE
from utils.math_utils import average
from utils.smoothing import LandmarkSmoother


BODY_KEEP_THRESHOLD = 0.55
HAND_KEEP_THRESHOLD = 0.45
TORSO_JOINTS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
ROLE_PRIORITY = {
    "front": 0,
    "back": 1,
    "right": 2,
    "left": 3,
    "up": 4,
}


@dataclass(slots=True)
class JointCandidate:
    role: str
    score: float
    joint: dict


class MultiCameraFusion:
    def __init__(self, config, primary_role: str = PRIMARY_CAMERA_ROLE):
        self.config = config
        self.primary_role = primary_role
        self.smoother = LandmarkSmoother(alpha=config.smoothing_alpha)

    def fuse_frame(self, detections_by_role: dict[str, list[dict]]) -> list[dict]:
        prepared_people: dict[str, list[dict]] = {}
        for role, people in detections_by_role.items():
            prepared_people[role] = self._prepare_people(role, people)

        person_count = max((len(people) for people in prepared_people.values()), default=0)
        fused_people = []

        for person_index in range(person_count):
            role_candidates = {
                role: people[person_index]
                for role, people in prepared_people.items()
                if person_index < len(people)
            }
            if not role_candidates:
                continue

            fused_body = self._fuse_body(role_candidates)
            left_hand, left_confidence = self._fuse_hand(role_candidates, "left_hand")
            right_hand, right_confidence = self._fuse_hand(role_candidates, "right_hand")

            fused_people.append(
                {
                    "id": person_index,
                    "body": fused_body,
                    "left_hand": left_hand,
                    "right_hand": right_hand,
                    "left_hand_confidence": left_confidence,
                    "right_hand_confidence": right_confidence,
                }
            )

        return self.smoother.smooth_people(fused_people)

    def _prepare_people(self, role: str, people: list[dict]) -> list[dict]:
        prepared = []
        for person in people:
            transformed_person = {
                "id": person.get("id", 0),
                "body": self._transform_section(role, person.get("body", {})),
                "left_hand": self._transform_section(role, person.get("left_hand", {})),
                "right_hand": self._transform_section(role, person.get("right_hand", {})),
                "left_hand_confidence": float(person.get("left_hand_confidence") or 0.0),
                "right_hand_confidence": float(person.get("right_hand_confidence") or 0.0),
            }
            transformed_person["_sort_key"] = self._person_sort_key(transformed_person)
            prepared.append(transformed_person)

        prepared.sort(key=lambda person: person["_sort_key"])
        return prepared

    def _transform_section(self, role: str, joints: dict) -> dict:
        transformed = {}
        for joint_name, joint in (joints or {}).items():
            transformed[joint_name] = self._transform_joint(role, joint)
        return transformed

    def _transform_joint(self, role: str, joint):
        if joint is None:
            return None

        x, y, z = self._rotate_to_primary(role, float(joint["x"]), float(joint["y"]), float(joint["z"]))
        transformed = {
            "x": round(x, 6),
            "y": round(y, 6),
            "z": round(z, 6),
        }

        for key, value in joint.items():
            if key not in transformed:
                transformed[key] = value

        return transformed

    def _rotate_to_primary(self, role: str, x: float, y: float, z: float) -> tuple[float, float, float]:
        if role == self.primary_role or role == "front":
            return x, y, z
        if role == "back":
            return -x, y, -z
        if role == "right":
            return z, y, -x
        if role == "left":
            return -z, y, x
        if role == "up":
            return x, -z, y
        return x, y, z

    def _person_sort_key(self, person: dict) -> tuple[float, float]:
        x_values = []
        z_values = []
        for joint_name in TORSO_JOINTS:
            joint = person.get("body", {}).get(joint_name)
            if joint is None:
                continue
            x_values.append(float(joint["x"]))
            z_values.append(float(joint["z"]))

        if not x_values:
            for joint in person.get("body", {}).values():
                if joint is None:
                    continue
                x_values.append(float(joint["x"]))
                z_values.append(float(joint["z"]))

        return (average(x_values), average(z_values))

    def _fuse_body(self, role_candidates: dict[str, dict]) -> dict:
        fused_body = {}
        for joint_name in BODY_LANDMARKS:
            candidates = []
            for role, person in role_candidates.items():
                joint = person.get("body", {}).get(joint_name)
                if joint is None:
                    continue
                score = float(joint.get("visibility", 0.0))
                candidates.append(JointCandidate(role=role, score=score, joint=joint))
            fused_body[joint_name] = self._choose_joint(candidates, BODY_KEEP_THRESHOLD)
        return fused_body

    def _fuse_hand(self, role_candidates: dict[str, dict], side_key: str) -> tuple[dict, float | None]:
        fused_hand = {}
        best_confidence = None
        confidence_key = f"{side_key}_confidence"

        for joint_name in HAND_LANDMARKS:
            candidates = []
            for role, person in role_candidates.items():
                joint = person.get(side_key, {}).get(joint_name)
                if joint is None:
                    continue
                score = float(person.get(confidence_key) or 0.0)
                candidates.append(JointCandidate(role=role, score=score, joint=joint))
            selected = self._choose_joint(candidates, HAND_KEEP_THRESHOLD)
            fused_hand[joint_name] = selected
            if selected is not None:
                selected_confidence = max((candidate.score for candidate in candidates if candidate.joint == selected), default=None)
                if selected_confidence is not None:
                    best_confidence = max(best_confidence or 0.0, selected_confidence)

        return fused_hand, best_confidence

    def _choose_joint(self, candidates: list[JointCandidate], keep_threshold: float):
        if not candidates:
            return None

        primary_candidate = next((candidate for candidate in candidates if candidate.role == self.primary_role), None)
        if primary_candidate is not None and primary_candidate.score >= keep_threshold:
            return dict(primary_candidate.joint)

        best_candidate = max(candidates, key=self._candidate_sort_key)
        return dict(best_candidate.joint)

    def _candidate_sort_key(self, candidate: JointCandidate) -> tuple[float, float]:
        priority = ROLE_PRIORITY.get(candidate.role, len(ROLE_PRIORITY))
        primary_bonus = 1.0 if candidate.role == self.primary_role else 0.0
        return (candidate.score, primary_bonus - (priority * 0.001))
