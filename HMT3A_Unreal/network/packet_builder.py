import json

# Landmarks 11-32 from the original Blazepose set
BODY_LANDMARKS = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# Standard 21 Hand Landmarks
HAND_LANDMARKS = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
]

def build_packet(people: list, frame_index: int) -> bytes:
    persons = []

    for person_landmarks in people:
        joints = {}
        num_landmarks = len(person_landmarks)

        # 1. Map Body Joints (Indices 0-21 in the filtered list)
        for i in range(min(num_landmarks, 22)):
            name = BODY_LANDMARKS[i]
            lm = person_landmarks[i]
            joints[name] = {
                "x": round(lm.x, 6),
                "y": round(lm.y, 6),
                "z": round(lm.z, 6),
                "visibility": getattr(lm, 'visibility', 1.0) # Hands don't always have visibility
            }

        # 2. Map Hand Joints (Indices 22+ if they exist)
        if num_landmarks > 22:
            hand_data = person_landmarks[22:]
            # We label them as hand_0_wrist, hand_1_wrist, etc.
            for j, lm in enumerate(hand_data):
                hand_idx = j // 21
                joint_idx = j % 21
                name = f"hand_{hand_idx}_{HAND_LANDMARKS[joint_idx]}"
                joints[name] = {
                    "x": round(lm.x, 6),
                    "y": round(lm.y, 6),
                    "z": round(lm.z, 6)
                }

        persons.append(joints)

    packet = {
        "frame": frame_index,
        "count": len(persons),
        "persons": persons,
    }

    return json.dumps(packet).encode("utf-8")