import cv2
import numpy as np
import onnxruntime as ort

BODY_MODEL_PATH = r"D:\IDT\Kinara\models\movenet.onnx"
HAND_MODEL_PATH = r"D:\IDT\Kinara\models\hand_pose.onnx"
VIDEO_PATH = r"location of video"
OUTPUT_PATH = r"location of output folder"

BODY_INPUT_NAME = "input"
HAND_INPUT_NAME = "images"

BODY_INPUT_SIZE = 192
HAND_INPUT_SIZE = 640

BODY_CONF_THRESHOLD = 0.3
HAND_DET_THRESHOLD = 0.15
HAND_KP_THRESHOLD = 0.20

BODY_EDGES = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6), (5, 11),
    (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

BODY_KEYPOINTS = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

body_session = ort.InferenceSession(
    BODY_MODEL_PATH,
    providers=["CUDAExecutionProvider"]
)

hand_session = ort.InferenceSession(
    HAND_MODEL_PATH,
    providers=["CUDAExecutionProvider"]
)

def run_hand_on_crop(frame_bgr, x1, y1, x2, y2):
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(crop_rgb, (HAND_INPUT_SIZE, HAND_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    hand_input = resized.astype(np.float32) / 255.0
    hand_input = np.transpose(hand_input, (2, 0, 1))
    hand_input = np.expand_dims(hand_input, axis=0)

    hand_output = hand_session.run(None, {HAND_INPUT_NAME: hand_input})
    detections = np.asarray(hand_output[0], dtype=np.float32)[0]

    best = detections[np.argmax(detections[:, 4])]
    if float(best[4]) <= HAND_DET_THRESHOLD:
        return None

    crop_w = x2 - x1
    crop_h = y2 - y1

    points = []
    for i in range(21):
        base = 6 + i * 3
        x = float(best[base])
        y = float(best[base + 1])
        conf = float(best[base + 2])

        px = x1 + int((x / HAND_INPUT_SIZE) * crop_w)
        py = y1 + int((y / HAND_INPUT_SIZE) * crop_h)
        points.append((px, py, conf))

    return points

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video file: {VIDEO_PATH}")

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

fourcc = cv2.VideoWriter.fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"Could not open video writer: {OUTPUT_PATH}")

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    body_resized = cv2.resize(rgb, (BODY_INPUT_SIZE, BODY_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    body_input = np.expand_dims(body_resized, axis=0).astype(np.int32)

    body_output = body_session.run(None, {BODY_INPUT_NAME: body_input})
    body_keypoints = np.asarray(body_output[0], dtype=np.float32)[0, 0]

    body_points = []
    for y, x, conf in body_keypoints:
        px = int(x * frame_w)
        py = int(y * frame_h)
        body_points.append((px, py, float(conf)))

    for start_idx, end_idx in BODY_EDGES:
        x1, y1, c1 = body_points[start_idx]
        x2, y2, c2 = body_points[end_idx]
        if c1 > BODY_CONF_THRESHOLD and c2 > BODY_CONF_THRESHOLD:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    for idx, (px, py, conf) in enumerate(body_points):
        if idx in BODY_KEYPOINTS and conf > BODY_CONF_THRESHOLD:
            cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

    wrist_indices = [9, 10]
    elbow_indices = {9: 7, 10: 8}

    for wrist_idx in wrist_indices:
        wx, wy, wc = body_points[wrist_idx]
        ex, ey, ec = body_points[elbow_indices[wrist_idx]]

        if wc <= BODY_CONF_THRESHOLD or ec <= BODY_CONF_THRESHOLD:
            continue

        forearm_len = int(np.hypot(wx - ex, wy - ey))
        box_size = max(160, forearm_len * 2)

        x1 = max(0, wx - box_size // 2)
        y1 = max(0, wy - box_size // 2)
        x2 = min(frame_w, wx + box_size // 2)
        y2 = min(frame_h, wy + box_size // 2)

        hand_points = run_hand_on_crop(frame, x1, y1, x2, y2)
        if hand_points is None:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 255), 1)

        for start_idx, end_idx in HAND_EDGES:
            x1p, y1p, c1 = hand_points[start_idx]
            x2p, y2p, c2 = hand_points[end_idx]
            if c1 > HAND_KP_THRESHOLD and c2 > HAND_KP_THRESHOLD:
                cv2.line(frame, (x1p, y1p), (x2p, y2p), (0, 255, 255), 2)

        for px, py, conf in hand_points:
            if conf > HAND_KP_THRESHOLD:
                cv2.circle(frame, (px, py), 3, (0, 165, 255), -1)

    writer.write(frame)
    cv2.imshow("Pose + Hand Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"Saved: {OUTPUT_PATH}")
