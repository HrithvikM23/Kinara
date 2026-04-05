from __future__ import annotations

from utils.math_utils import clamp, vector_length, vector_subtract


SECTION_SETTINGS = {
    "body": {
        "alpha_key": "body_smoothing_alpha",
        "deadband_key": "joint_deadband",
        "max_jump_key": "max_position_jump",
    },
    "left_hand": {
        "alpha_key": "hand_smoothing_alpha",
        "deadband_key": "hand_joint_deadband",
        "max_jump_key": "max_hand_position_jump",
    },
    "right_hand": {
        "alpha_key": "hand_smoothing_alpha",
        "deadband_key": "hand_joint_deadband",
        "max_jump_key": "max_hand_position_jump",
    },
}


class LandmarkSmoother:
    def __init__(self, config, max_stale_frames: int = 12):
        self.config = config
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

            smoothed_person = {
                "id": person["id"],
                "body": self._smooth_section(track, "body", person.get("body", {})),
                "left_hand": self._smooth_section(track, "left_hand", person.get("left_hand", {})),
                "right_hand": self._smooth_section(track, "right_hand", person.get("right_hand", {})),
                "left_hand_confidence": person.get("left_hand_confidence"),
                "right_hand_confidence": person.get("right_hand_confidence"),
            }
            if "_anchor" in person:
                smoothed_person["_anchor"] = person["_anchor"]
            smoothed_people.append(smoothed_person)

        self._prune_tracks()
        return smoothed_people

    def _smooth_section(self, track: dict, section_name: str, joints: dict) -> dict:
        section_state = track["sections"].setdefault(section_name, {})
        result = {}

        for joint_name, joint in joints.items():
            result[joint_name] = self._smooth_joint(section_state, section_name, joint_name, joint)

        return result

    def _smooth_joint(self, section_state: dict, section_name: str, joint_name: str, joint):
        state = section_state.setdefault(
            joint_name,
            {
                "value": None,
                "velocity": (0.0, 0.0, 0.0),
                "missing": 0,
            },
        )

        if joint is None:
            state["missing"] += 1
            if state["value"] is not None and state["missing"] <= int(self.config.missing_joint_hold_frames):
                return dict(state["value"])
            state["value"] = None
            state["velocity"] = (0.0, 0.0, 0.0)
            return None

        previous = state["value"]
        state["missing"] = 0
        if previous is None:
            smoothed = dict(joint)
            state["value"] = smoothed
            return smoothed

        settings = SECTION_SETTINGS[section_name]
        base_alpha = float(getattr(self.config, settings["alpha_key"]))
        deadband = float(getattr(self.config, settings["deadband_key"]))
        max_jump = float(getattr(self.config, settings["max_jump_key"]))

        delta = vector_subtract(
            (float(joint["x"]), float(joint["y"]), float(joint["z"])),
            (float(previous["x"]), float(previous["y"]), float(previous["z"])),
        )
        distance = vector_length(delta)

        if distance <= deadband:
            smoothed = dict(previous)
            for key, value in joint.items():
                if key not in smoothed:
                    smoothed[key] = value
            state["value"] = smoothed
            return smoothed

        if distance > max_jump > 0.0:
            ratio = max_jump / max(distance, 1e-8)
            joint = {
                **joint,
                "x": float(previous["x"]) + (delta[0] * ratio),
                "y": float(previous["y"]) + (delta[1] * ratio),
                "z": float(previous["z"]) + (delta[2] * ratio),
            }
            delta = vector_subtract(
                (float(joint["x"]), float(joint["y"]), float(joint["z"])),
                (float(previous["x"]), float(previous["y"]), float(previous["z"])),
            )
            distance = vector_length(delta)

        adaptive_alpha = clamp(base_alpha + min(distance * 0.9, 0.35), 0.08, 0.92)
        predicted = [float(previous[axis]) + (0.45 * state["velocity"][index]) for index, axis in enumerate(("x", "y", "z"))]

        smoothed = {}
        for index, axis in enumerate(("x", "y", "z")):
            current_value = float(joint[axis])
            smoothed_value = (adaptive_alpha * current_value) + ((1.0 - adaptive_alpha) * predicted[index])
            smoothed[axis] = round(smoothed_value, 6)

        if "visibility" in joint:
            previous_visibility = float(previous.get("visibility", joint["visibility"]))
            smoothed["visibility"] = round(
                (adaptive_alpha * float(joint["visibility"])) + ((1.0 - adaptive_alpha) * previous_visibility),
                6,
            )

        for key, value in joint.items():
            if key not in smoothed:
                smoothed[key] = value

        state["velocity"] = tuple(
            smoothed[axis] - float(previous[axis])
            for axis in ("x", "y", "z")
        )
        state["value"] = smoothed
        return smoothed

    def _prune_tracks(self) -> None:
        stale_keys = [
            key
            for key, track in self.tracks.items()
            if (self.frame_index - track["last_seen"]) > self.max_stale_frames
        ]
        for key in stale_keys:
            del self.tracks[key]
