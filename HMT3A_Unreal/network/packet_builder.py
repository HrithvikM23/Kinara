import json
import numpy as np

BODY_25_JOINTS = [
    "Nose", "Neck",
    "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist",
    "MidHip",
    "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar",
    "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel"
]


def build_packet(keypoints: np.ndarray, frame_index: int) -> bytes:
    """
    keypoints:   shape (25, 3) — x, y, confidence per joint
    frame_index: current frame number
    Returns:     UTF-8 encoded JSON bytes ready for UDP
    """
    joints = {}
    for i, name in enumerate(BODY_25_JOINTS):
        joints[name] = {
            "x":          float(keypoints[i][0]),
            "y":          float(keypoints[i][1]),
            "confidence": float(keypoints[i][2]),
        }

    packet = {
        "frame":  frame_index,
        "joints": joints,
    }

    return json.dumps(packet).encode("utf-8")