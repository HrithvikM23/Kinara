from __future__ import annotations

import math


def build_hand_box(
    wrist_point: tuple[int, int, float],
    elbow_point: tuple[int, int, float],
    frame_width: int,
    frame_height: int,
    min_box_size: int,
    scale: float,
) -> tuple[int, int, int, int]:
    wx, wy, _ = wrist_point
    ex, ey, _ = elbow_point

    forearm_len = int(math.hypot(wx - ex, wy - ey))
    box_size = max(min_box_size, int(forearm_len * scale))

    x1 = max(0, wx - box_size // 2)
    y1 = max(0, wy - box_size // 2)
    x2 = min(frame_width, wx + box_size // 2)
    y2 = min(frame_height, wy + box_size // 2)
    return x1, y1, x2, y2
