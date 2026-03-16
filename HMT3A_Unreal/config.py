# ── UDP settings ──────────────────────────────────────────────────
UDP_IP   = "127.0.0.1"
UDP_PORT = 7000

# ── Pose settings ─────────────────────────────────────────────────

MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE  = 0.8
MIN_POSE_PRESENCE_CONFIDENCE = 0.7

# ── Hand settings ───────────────────────────────────────────────

MIN_HAND_DETECTION_CONFIDENCE = 0.7  # Lower this to catch "difficult" hands
MIN_HAND_PRESENCE_CONFIDENCE = 0.8   # Lower this to keep the landmarks visible
MIN_HAND_TRACKING_CONFIDENCE = 0.7   # Keep this stable

# ── Debug settings ───────────────────────────────────────────────

MAX_PERSONS = 1      #default value, can be overridden in main.py