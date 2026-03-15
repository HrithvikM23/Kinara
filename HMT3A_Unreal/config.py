import sys

# ── OpenPose paths ────────────────────────────────────────────────
OPENPOSE_PYTHON_PATH = r"D:\IDT\openpose\build\python\openpose\Release"
OPENPOSE_BIN_PATH    = r"D:\IDT\openpose\build\bin"
OPENPOSE_MODELS_PATH = r"D:\IDT\openpose\models"

sys.path.append(OPENPOSE_PYTHON_PATH)
sys.path.append(OPENPOSE_BIN_PATH)

# ── UDP settings ──────────────────────────────────────────────────
UDP_IP   = "127.0.0.1"
UDP_PORT = 7000

# ── Pose settings ─────────────────────────────────────────────────
POSE_MODEL    = "BODY_25"
LOGGING_LEVEL = 3