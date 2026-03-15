import json

BLAZEPOSE_LANDMARKS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]


def build_packet(people: list, frame_index: int) -> bytes:
    """
    people:      list of person landmark lists (each 33 world landmarks)
    frame_index: current frame number
    Returns:     UTF-8 encoded JSON bytes ready for UDP
    """
    persons = []

    for person_landmarks in people:
        joints = {}
        for i, name in enumerate(BLAZEPOSE_LANDMARKS):
            lm = person_landmarks[i]
            joints[name] = {
                "x":          round(lm.x, 6),
                "y":          round(lm.y, 6),
                "z":          round(lm.z, 6),
                "visibility": round(lm.visibility, 4),
            }
        persons.append(joints)

    packet = {
        "frame":   frame_index,
        "count":   len(persons),
        "persons": persons,
    }

    return json.dumps(packet).encode("utf-8")