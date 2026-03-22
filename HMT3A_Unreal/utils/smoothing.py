from __future__ import annotations

from utils.math_utils import clamp


class LandmarkSmoother:
    def __init__(self, alpha: float = 0.65, max_stale_frames: int = 12):
        self.alpha = clamp(float(alpha), 0.0, 1.0)
        self.max_stale_frames = max_stale_frames
        self.frame_index = 0
        self.tracks = {}

    def smooth_people(self, people: list) -> list:
        self.frame_index += 1
        smoothed_people = []

        for person in people:
            track_key = f"person_{person['id']}"
            track = self.tracks.setdefault(
                track_key,
                {
                    "last_seen": self.frame_index,
                    "sections": {},
                },
            )
            track["last_seen"] = self.frame_index

            smoothed_people.append(
                {
                    "id": person["id"],
                    "body": self._smooth_section(track, "body", person.get("body", {})),
                    "left_hand": self._smooth_section(track, "left_hand", person.get("left_hand", {})),
                    "right_hand": self._smooth_section(track, "right_hand", person.get("right_hand", {})),
                    "left_hand_confidence": person.get("left_hand_confidence"),
                    "right_hand_confidence": person.get("right_hand_confidence"),
                }
            )

        self._prune_tracks()
        return smoothed_people

    def _smooth_section(self, track: dict, section_name: str, joints: dict) -> dict:
        section_state = track["sections"].setdefault(section_name, {})
        result = {}

        for joint_name, joint in joints.items():
            result[joint_name] = self._smooth_joint(section_state, joint_name, joint)

        return result

    def _smooth_joint(self, section_state: dict, joint_name: str, joint):
        if joint is None:
            return None

        previous = section_state.get(joint_name)
        if previous is None:
            smoothed = dict(joint)
        else:
            smoothed = {}
            for axis in ("x", "y", "z"):
                smoothed[axis] = round(
                    (self.alpha * float(joint[axis])) + ((1.0 - self.alpha) * float(previous[axis])),
                    6,
                )

            if "visibility" in joint:
                previous_visibility = float(previous.get("visibility", joint["visibility"]))
                smoothed["visibility"] = round(
                    (self.alpha * float(joint["visibility"])) + ((1.0 - self.alpha) * previous_visibility),
                    6,
                )

            for key, value in joint.items():
                if key not in smoothed:
                    smoothed[key] = value

        section_state[joint_name] = smoothed
        return smoothed

    def _prune_tracks(self) -> None:
        stale_keys = [
            key
            for key, track in self.tracks.items()
            if (self.frame_index - track["last_seen"]) > self.max_stale_frames
        ]
        for key in stale_keys:
            del self.tracks[key]
