from __future__ import annotations

from dataclasses import dataclass, field

from utils.math_utils import average_points, distance_3d, vector_add, vector_scale, vector_subtract


@dataclass(slots=True)
class TrackState:
    track_id: int
    anchor: dict
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    missed_frames: int = 0
    age: int = 0
    yolo_track_id: int | None = None
    profile_slot_id: int | None = None
    appearance_regions: dict = field(default_factory=dict)
    seen_since_frame: int = 0
    last_seen_frame: int = 0
    seen_since_timestamp_ms: int = 0
    last_seen_timestamp_ms: int = 0


class PersonTracker:
    def __init__(self, max_match_distance: float = 0.36, max_missed_frames: int = 18, identity_profiles: list | None = None):
        self.max_match_distance = float(max_match_distance)
        self.max_missed_frames = int(max_missed_frames)
        self.identity_profiles = {int(profile.slot_id): profile for profile in (identity_profiles or [])}
        self.next_track_id = max(self.identity_profiles.keys(), default=0) + 1
        self.tracks: dict[int, TrackState] = {}

    def update(self, persons: list[dict], frame_index: int = 0, timestamp_ms: int = 0) -> list[dict]:
        remaining_persons = list(persons)
        track_matches: list[tuple[TrackState, dict]] = []
        claimed_profile_slots = {
            track.profile_slot_id
            for track in self.tracks.values()
            if track.profile_slot_id is not None and track.missed_frames <= self.max_missed_frames
        }

        for track in sorted(self.tracks.values(), key=lambda item: item.track_id):
            predicted_anchor = self._predict_anchor(track)
            best_index = None
            best_score = None

            for person_index, person in enumerate(remaining_persons):
                score = self._match_score(track, person, predicted_anchor)
                if score is None:
                    continue
                if best_score is None or score < best_score:
                    best_score = score
                    best_index = person_index

            if best_index is not None:
                person = remaining_persons.pop(best_index)
                self._update_track(track, person, frame_index, timestamp_ms, claimed_profile_slots)
                track_matches.append((track, person))
            else:
                track.missed_frames += 1

        for person in remaining_persons:
            track = self._create_track(person, frame_index, timestamp_ms, claimed_profile_slots)
            track_matches.append((track, person))
            if track.profile_slot_id is not None:
                claimed_profile_slots.add(track.profile_slot_id)

        self._prune_tracks()

        updated_people = []
        for track, person in sorted(track_matches, key=lambda item: item[0].track_id):
            updated_person = dict(person)
            updated_person["id"] = track.track_id
            updated_person["identity"] = self._build_identity_payload(track, person)
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

    def _create_track(self, person: dict, frame_index: int, timestamp_ms: int, claimed_profile_slots: set[int | None]) -> TrackState:
        anchor = person.get("_anchor") or {"x": 0.0, "y": 0.0, "z": 0.0}
        profile_slot_id = self._choose_profile_slot(person, claimed_profile_slots)
        track_id = profile_slot_id if profile_slot_id is not None else self.next_track_id
        if profile_slot_id is None:
            self.next_track_id += 1

        track = TrackState(
            track_id=track_id,
            anchor=dict(anchor),
            age=1,
            yolo_track_id=person.get("_yolo_track_id"),
            profile_slot_id=profile_slot_id,
            appearance_regions=self._appearance_regions(person),
            seen_since_frame=int(frame_index),
            last_seen_frame=int(frame_index),
            seen_since_timestamp_ms=int(timestamp_ms),
            last_seen_timestamp_ms=int(timestamp_ms),
        )
        self.tracks[track.track_id] = track
        return track

    def _update_track(self, track: TrackState, person: dict, frame_index: int, timestamp_ms: int, claimed_profile_slots: set[int | None]) -> None:
        anchor = person.get("_anchor") or track.anchor
        delta = vector_subtract(
            (float(anchor["x"]), float(anchor["y"]), float(anchor["z"])),
            (float(track.anchor["x"]), float(track.anchor["y"]), float(track.anchor["z"])),
        )
        if track.age > 0:
            track.velocity = tuple((0.55 * delta[index]) + (0.45 * track.velocity[index]) for index in range(3))
        else:
            track.velocity = delta

        track.anchor = dict(anchor)
        track.missed_frames = 0
        track.age += 1
        track.last_seen_frame = int(frame_index)
        track.last_seen_timestamp_ms = int(timestamp_ms)
        if person.get("_yolo_track_id") is not None:
            track.yolo_track_id = int(person["_yolo_track_id"])
        if track.profile_slot_id is None:
            track.profile_slot_id = self._choose_profile_slot(person, claimed_profile_slots)
            if track.profile_slot_id is not None:
                claimed_profile_slots.add(track.profile_slot_id)
        track.appearance_regions = self._merge_regions(track.appearance_regions, self._appearance_regions(person))

    def _match_score(self, track: TrackState, person: dict, predicted_anchor: dict) -> float | None:
        anchor = person.get("_anchor")
        if anchor is None:
            return None

        distance = distance_3d(anchor, predicted_anchor)
        same_yolo_track = track.yolo_track_id is not None and person.get("_yolo_track_id") == track.yolo_track_id
        profile_score = self._profile_score(person, track.profile_slot_id)
        strong_identity_link = same_yolo_track or profile_score >= 0.28
        if distance > self.max_match_distance and not strong_identity_link:
            return None

        score = distance / max(self.max_match_distance, 1e-6)
        score += 0.55 * self._appearance_distance(track.appearance_regions, self._appearance_regions(person))

        if same_yolo_track:
            score -= 0.8
        elif track.yolo_track_id is not None and person.get("_yolo_track_id") is not None:
            score += 0.35

        if track.profile_slot_id is not None:
            score -= min(profile_score, 1.0) * 0.9
            if profile_score < 0.08 and distance > (self.max_match_distance * 0.5):
                score += 0.25

        return score

    def _appearance_regions(self, person: dict) -> dict:
        return ((person.get("_appearance") or {}).get("regions") or {})

    def _appearance_distance(self, track_regions: dict, person_regions: dict) -> float:
        if not track_regions or not person_regions:
            return 0.0

        total = 0.0
        weight_total = 0.0
        for region_name in set(track_regions).intersection(person_regions):
            track_payload = track_regions.get(region_name) or {}
            person_payload = person_regions.get(region_name) or {}
            weight = max(float(track_payload.get("score") or 0.0), float(person_payload.get("score") or 0.0), 0.15)
            mismatch = 0.0 if track_payload.get("color") == person_payload.get("color") else 1.0
            total += mismatch * weight
            weight_total += weight

        if weight_total <= 0.0:
            return 0.0
        return total / weight_total

    def _merge_regions(self, track_regions: dict, person_regions: dict) -> dict:
        merged = dict(track_regions or {})
        for region_name, payload in (person_regions or {}).items():
            current = merged.get(region_name) or {}
            current_score = float(current.get("score") or 0.0)
            new_score = float(payload.get("score") or 0.0)
            if new_score >= current_score:
                merged[region_name] = {
                    "color": payload.get("color"),
                    "score": round(new_score, 4),
                }
        return merged

    def _choose_profile_slot(self, person: dict, claimed_profile_slots: set[int | None]) -> int | None:
        profile_scores = ((person.get("_appearance") or {}).get("profile_scores") or {})
        best_slot = None
        best_score = 0.0
        for slot_id, score in profile_scores.items():
            numeric_slot = int(slot_id)
            if numeric_slot in claimed_profile_slots:
                continue
            if float(score) > best_score:
                best_slot = numeric_slot
                best_score = float(score)
        return best_slot if best_score >= 0.08 else None

    def _profile_score(self, person: dict, profile_slot_id: int | None) -> float:
        if profile_slot_id is None:
            return 0.0
        profile_scores = ((person.get("_appearance") or {}).get("profile_scores") or {})
        return float(profile_scores.get(profile_slot_id) or 0.0)

    def _build_identity_payload(self, track: TrackState, person: dict) -> dict:
        profile = self.identity_profiles.get(track.profile_slot_id)
        appearance = person.get("_appearance") or {}
        regions = appearance.get("regions") or {}
        top_region = regions.get("top") or {}
        torso_region = regions.get("torso") or {}
        return {
            "label": profile.label if profile is not None else f"Person {track.track_id}",
            "profile_slot": track.profile_slot_id,
            "profile_color": getattr(profile, "color_name", None),
            "profile_region": getattr(profile, "region", None),
            "profile_score": round(self._profile_score(person, track.profile_slot_id), 4) if track.profile_slot_id is not None else None,
            "top_color": top_region.get("color"),
            "torso_color": torso_region.get("color"),
            "yolo_track_id": track.yolo_track_id,
            "seen_since_frame": track.seen_since_frame,
            "last_seen_frame": track.last_seen_frame,
            "seen_since_timestamp_ms": track.seen_since_timestamp_ms,
            "last_seen_timestamp_ms": track.last_seen_timestamp_ms,
        }

    def _prune_tracks(self) -> None:
        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed_frames > self.max_missed_frames
        ]
        for track_id in stale_ids:
            del self.tracks[track_id]
