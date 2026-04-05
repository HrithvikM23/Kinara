from __future__ import annotations

from dataclasses import dataclass

from utils.math_utils import average_points, distance_3d, vector_add, vector_scale, vector_subtract


@dataclass(slots=True)
class TrackState:
    track_id: int
    anchor: dict
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    missed_frames: int = 0
    age: int = 0


class PersonTracker:
    def __init__(self, max_match_distance: float = 0.36, max_missed_frames: int = 18):
        self.max_match_distance = float(max_match_distance)
        self.max_missed_frames = int(max_missed_frames)
        self.next_track_id = 0
        self.tracks: dict[int, TrackState] = {}

    def update(self, persons: list[dict]) -> list[dict]:
        remaining_persons = list(persons)
        track_candidates = []

        for track in self.tracks.values():
            predicted_anchor = self._predict_anchor(track)
            best_index = None
            best_distance = None

            for person_index, person in enumerate(remaining_persons):
                anchor = person.get("_anchor")
                if anchor is None:
                    continue
                distance = distance_3d(anchor, predicted_anchor)
                if distance > self.max_match_distance:
                    continue
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_index = person_index

            if best_index is not None:
                person = remaining_persons.pop(best_index)
                self._update_track(track, person)
                track_candidates.append((track.track_id, person))
            else:
                track.missed_frames += 1

        for person in remaining_persons:
            track = self._create_track(person)
            track_candidates.append((track.track_id, person))

        self._prune_tracks()

        updated_people = []
        for track_id, person in sorted(track_candidates, key=lambda item: item[0]):
            updated_person = dict(person)
            updated_person["id"] = track_id
            updated_people.append(updated_person)
        return updated_people

    def _predict_anchor(self, track: TrackState) -> dict:
        predicted = average_points([track.anchor])
        if predicted is None:
            return {"x": 0.0, "y": 0.0, "z": 0.0}

        if track.missed_frames <= 0:
            return predicted

        extrapolated = vector_add(
            (predicted["x"], predicted["y"], predicted["z"]),
            vector_scale(track.velocity, min(track.missed_frames, 3)),
        )
        return {"x": extrapolated[0], "y": extrapolated[1], "z": extrapolated[2]}

    def _create_track(self, person: dict) -> TrackState:
        anchor = person.get("_anchor") or {"x": 0.0, "y": 0.0, "z": 0.0}
        track = TrackState(track_id=self.next_track_id, anchor=dict(anchor), age=1)
        self.tracks[track.track_id] = track
        self.next_track_id += 1
        return track

    def _update_track(self, track: TrackState, person: dict) -> None:
        anchor = person.get("_anchor") or track.anchor
        delta = vector_subtract(
            (float(anchor["x"]), float(anchor["y"]), float(anchor["z"])),
            (float(track.anchor["x"]), float(track.anchor["y"]), float(track.anchor["z"])),
        )
        if track.age > 0:
            track.velocity = tuple((0.55 * delta[i_index]) + (0.45 * track.velocity[i_index]) for i_index in range(3))
        else:
            track.velocity = delta
        track.anchor = dict(anchor)
        track.missed_frames = 0
        track.age += 1

    def _prune_tracks(self) -> None:
        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed_frames > self.max_missed_frames
        ]
        for track_id in stale_ids:
            del self.tracks[track_id]
