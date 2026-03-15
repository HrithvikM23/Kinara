from config import OPENPOSE_BIN_PATH, OPENPOSE_MODELS_PATH, POSE_MODEL, LOGGING_LEVEL

import pyopenpose as op


def compute_net_resolution(frame, base=16, target_short_side=368):
    """
    Computes a net_resolution string that:
    - Preserves the video's aspect ratio
    - Keeps the short side close to target_short_side
    - Rounds both dimensions to the nearest multiple of base (required by OpenPose)
    """
    h, w = frame.shape[:2]

    if h < w:
        scale = target_short_side / h
    else:
        scale = target_short_side / w

    new_w = round((w * scale) / base) * base
    new_h = round((h * scale) / base) * base

    return f"{new_w}x{new_h}"


class PoseDetector:

    def __init__(self, first_frame):
        resolution = compute_net_resolution(first_frame)
        print(f"Net resolution set to: {resolution}")

        params = {
            "model_folder":   OPENPOSE_MODELS_PATH,
            "model_pose":     POSE_MODEL,
            "net_resolution": resolution,
            "logging_level":  LOGGING_LEVEL,
        }

        self.wrapper = op.WrapperPython()
        self.wrapper.configure(params)
        self.wrapper.start()
        print("PoseDetector ready.")

    def detect(self, frame):
        """
        Input:  BGR numpy frame
        Output: (keypoints, rendered_frame)
                keypoints → numpy array shape (25, 3) [x, y, confidence]
                            or None if no person detected
        """
        datum = op.Datum()
        datum.cvInputData = frame
        self.wrapper.emplaceAndPop(op.VectorDatum([datum]))

        kp = datum.poseKeypoints

        if kp is None or kp.shape[0] == 0:
            return None, datum.cvOutputData

        return kp[0], datum.cvOutputData