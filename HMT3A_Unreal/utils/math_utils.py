from __future__ import annotations

import math


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def average(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def distance_2d(point_a, point_b) -> float:
    dx = float(point_a["x"]) - float(point_b["x"])
    dy = float(point_a["y"]) - float(point_b["y"])
    return math.hypot(dx, dy)


def vector_between(start_point, end_point):
    if start_point is None or end_point is None:
        return None
    return (
        float(end_point["x"]) - float(start_point["x"]),
        float(end_point["y"]) - float(start_point["y"]),
        float(end_point["z"]) - float(start_point["z"]),
    )


def vector_length(vector) -> float:
    if vector is None:
        return 0.0
    return math.sqrt(sum(component * component for component in vector))


def normalize(vector):
    if vector is None:
        return None
    length = vector_length(vector)
    if length == 0:
        return (0.0, 0.0, 0.0)
    return tuple(component / length for component in vector)


def dot(vector_a, vector_b) -> float:
    return sum(component_a * component_b for component_a, component_b in zip(vector_a, vector_b))


def angle_between_points(point_a, point_b, point_c):
    if point_a is None or point_b is None or point_c is None:
        return None

    vector_ba = vector_between(point_b, point_a)
    vector_bc = vector_between(point_b, point_c)

    length_ba = vector_length(vector_ba)
    length_bc = vector_length(vector_bc)
    if length_ba == 0 or length_bc == 0:
        return None

    cosine = dot(vector_ba, vector_bc) / (length_ba * length_bc)
    cosine = clamp(cosine, -1.0, 1.0)
    return math.degrees(math.acos(cosine))
