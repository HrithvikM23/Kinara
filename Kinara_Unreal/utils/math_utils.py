from __future__ import annotations

import math


EPSILON = 1e-8


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def average(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def round_vector(vector, digits: int = 6):
    if vector is None:
        return None
    return [round(float(component), digits) for component in vector]


def as_vector(point) -> tuple[float, float, float]:
    return (float(point["x"]), float(point["y"]), float(point["z"]))


def vector_to_point(vector, template=None, digits: int = 6) -> dict:
    payload = {
        "x": round(float(vector[0]), digits),
        "y": round(float(vector[1]), digits),
        "z": round(float(vector[2]), digits),
    }
    if template is not None:
        for key, value in template.items():
            if key not in payload:
                payload[key] = value
    return payload


def average_points(points, weights=None):
    points = list(points)
    if not points:
        return None

    if weights is None:
        weights = [1.0] * len(points)
    else:
        weights = [float(weight) for weight in weights]

    total_weight = sum(max(weight, 0.0) for weight in weights)
    if total_weight <= EPSILON:
        total_weight = float(len(points))
        weights = [1.0] * len(points)

    x_value = sum(float(point["x"]) * weight for point, weight in zip(points, weights)) / total_weight
    y_value = sum(float(point["y"]) * weight for point, weight in zip(points, weights)) / total_weight
    z_value = sum(float(point["z"]) * weight for point, weight in zip(points, weights)) / total_weight

    template = next((point for point in points if point is not None), None)
    return vector_to_point((x_value, y_value, z_value), template=template)


def add_points(point_a, point_b, digits: int = 6):
    return vector_to_point(vector_add(as_vector(point_a), as_vector(point_b)), digits=digits)


def subtract_points(point_a, point_b, digits: int = 6):
    return vector_to_point(vector_subtract(as_vector(point_a), as_vector(point_b)), digits=digits)


def midpoint(point_a, point_b):
    if point_a is None or point_b is None:
        return None
    return average_points([point_a, point_b])


def distance_2d(point_a, point_b) -> float:
    dx = float(point_a["x"]) - float(point_b["x"])
    dy = float(point_a["y"]) - float(point_b["y"])
    return math.hypot(dx, dy)


def distance_3d(point_a, point_b) -> float:
    return vector_length(vector_between(point_a, point_b))


def vector_between(start_point, end_point):
    if start_point is None or end_point is None:
        return None
    return (
        float(end_point["x"]) - float(start_point["x"]),
        float(end_point["y"]) - float(start_point["y"]),
        float(end_point["z"]) - float(start_point["z"]),
    )


def vector_add(vector_a, vector_b):
    return tuple(float(component_a) + float(component_b) for component_a, component_b in zip(vector_a, vector_b))


def vector_subtract(vector_a, vector_b):
    return tuple(float(component_a) - float(component_b) for component_a, component_b in zip(vector_a, vector_b))


def vector_scale(vector, scalar: float):
    return tuple(float(component) * float(scalar) for component in vector)


def vector_length(vector) -> float:
    if vector is None:
        return 0.0
    return math.sqrt(sum(component * component for component in vector))


def normalize(vector):
    if vector is None:
        return None
    length = vector_length(vector)
    if length <= EPSILON:
        return (0.0, 0.0, 0.0)
    return tuple(component / length for component in vector)


def dot(vector_a, vector_b) -> float:
    return sum(component_a * component_b for component_a, component_b in zip(vector_a, vector_b))


def cross(vector_a, vector_b):
    ax, ay, az = vector_a
    bx, by, bz = vector_b
    return (
        (ay * bz) - (az * by),
        (az * bx) - (ax * bz),
        (ax * by) - (ay * bx),
    )


def project_vector(vector, onto_vector):
    onto_norm = normalize(onto_vector)
    if onto_norm is None or vector_length(onto_norm) <= EPSILON:
        return (0.0, 0.0, 0.0)
    scale = dot(vector, onto_norm)
    return vector_scale(onto_norm, scale)


def angle_between_points(point_a, point_b, point_c):
    if point_a is None or point_b is None or point_c is None:
        return None

    vector_ba = vector_between(point_b, point_a)
    vector_bc = vector_between(point_b, point_c)

    length_ba = vector_length(vector_ba)
    length_bc = vector_length(vector_bc)
    if length_ba <= EPSILON or length_bc <= EPSILON:
        return None

    cosine = dot(vector_ba, vector_bc) / (length_ba * length_bc)
    cosine = clamp(cosine, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def quaternion_identity() -> tuple[float, float, float, float]:
    return (1.0, 0.0, 0.0, 0.0)


def quaternion_normalize(quaternion):
    length = math.sqrt(sum(component * component for component in quaternion))
    if length <= EPSILON:
        return quaternion_identity()
    return tuple(component / length for component in quaternion)


def quaternion_conjugate(quaternion):
    w_value, x_value, y_value, z_value = quaternion
    return (w_value, -x_value, -y_value, -z_value)


def quaternion_inverse(quaternion):
    conjugate = quaternion_conjugate(quaternion)
    magnitude = sum(component * component for component in quaternion)
    if magnitude <= EPSILON:
        return quaternion_identity()
    return tuple(component / magnitude for component in conjugate)


def quaternion_multiply(quaternion_a, quaternion_b):
    aw, ax, ay, az = quaternion_a
    bw, bx, by, bz = quaternion_b
    return quaternion_normalize(
        (
            (aw * bw) - (ax * bx) - (ay * by) - (az * bz),
            (aw * bx) + (ax * bw) + (ay * bz) - (az * by),
            (aw * by) - (ax * bz) + (ay * bw) + (az * bx),
            (aw * bz) + (ax * by) - (ay * bx) + (az * bw),
        )
    )


def quaternion_from_axis_angle(axis, angle_radians: float):
    axis = normalize(axis)
    if axis is None or vector_length(axis) <= EPSILON:
        return quaternion_identity()
    half_angle = angle_radians / 2.0
    sin_half = math.sin(half_angle)
    return quaternion_normalize(
        (
            math.cos(half_angle),
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
        )
    )


def quaternion_between_vectors(vector_a, vector_b):
    unit_a = normalize(vector_a)
    unit_b = normalize(vector_b)
    if unit_a is None or unit_b is None:
        return quaternion_identity()

    cosine = clamp(dot(unit_a, unit_b), -1.0, 1.0)
    if cosine >= (1.0 - EPSILON):
        return quaternion_identity()

    if cosine <= (-1.0 + EPSILON):
        axis = cross((1.0, 0.0, 0.0), unit_a)
        if vector_length(axis) <= EPSILON:
            axis = cross((0.0, 1.0, 0.0), unit_a)
        return quaternion_from_axis_angle(axis, math.pi)

    axis = cross(unit_a, unit_b)
    quaternion = (1.0 + cosine, axis[0], axis[1], axis[2])
    return quaternion_normalize(quaternion)


def quaternion_from_matrix(matrix_rows):
    m00, m01, m02 = matrix_rows[0]
    m10, m11, m12 = matrix_rows[1]
    m20, m21, m22 = matrix_rows[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            0.25 * scale,
            (m21 - m12) / scale,
            (m02 - m20) / scale,
            (m10 - m01) / scale,
        )
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        quaternion = (
            (m21 - m12) / scale,
            0.25 * scale,
            (m01 + m10) / scale,
            (m02 + m20) / scale,
        )
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        quaternion = (
            (m02 - m20) / scale,
            (m01 + m10) / scale,
            0.25 * scale,
            (m12 + m21) / scale,
        )
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        quaternion = (
            (m10 - m01) / scale,
            (m02 + m20) / scale,
            (m12 + m21) / scale,
            0.25 * scale,
        )

    return quaternion_normalize(quaternion)


def quaternion_from_forward_up(forward_vector, up_hint=None):
    forward = normalize(forward_vector)
    if forward is None or vector_length(forward) <= EPSILON:
        return quaternion_identity()

    if up_hint is None or vector_length(up_hint) <= EPSILON:
        up_hint = (0.0, 1.0, 0.0)
        if abs(dot(forward, up_hint)) >= 0.95:
            up_hint = (0.0, 0.0, 1.0)

    right = cross(up_hint, forward)
    if vector_length(right) <= EPSILON:
        fallback = (1.0, 0.0, 0.0) if abs(forward[0]) < 0.9 else (0.0, 0.0, 1.0)
        right = cross(fallback, forward)

    right = normalize(right)
    if right is None or vector_length(right) <= EPSILON:
        return quaternion_identity()

    up = normalize(cross(forward, right))
    if up is None or vector_length(up) <= EPSILON:
        return quaternion_identity()

    matrix_rows = (
        (right[0], forward[0], up[0]),
        (right[1], forward[1], up[1]),
        (right[2], forward[2], up[2]),
    )
    return quaternion_from_matrix(matrix_rows)


def rotate_vector(vector, quaternion):
    quaternion_vector = (0.0, vector[0], vector[1], vector[2])
    rotated = quaternion_multiply(quaternion_multiply(quaternion, quaternion_vector), quaternion_inverse(quaternion))
    return (rotated[1], rotated[2], rotated[3])


def quaternion_to_euler_degrees(quaternion):
    w_value, x_value, y_value, z_value = quaternion_normalize(quaternion)

    sinr_cosp = 2.0 * ((w_value * x_value) + (y_value * z_value))
    cosr_cosp = 1.0 - (2.0 * ((x_value * x_value) + (y_value * y_value)))
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * ((w_value * y_value) - (z_value * x_value))
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * ((w_value * z_value) + (x_value * y_value))
    cosy_cosp = 1.0 - (2.0 * ((y_value * y_value) + (z_value * z_value)))
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return [round(math.degrees(angle), 6) for angle in (roll, pitch, yaw)]


def rotation_matrix_from_euler_degrees(rotation_deg):
    x_deg, y_deg, z_deg = rotation_deg
    x_rad = math.radians(float(x_deg))
    y_rad = math.radians(float(y_deg))
    z_rad = math.radians(float(z_deg))

    cx, sx = math.cos(x_rad), math.sin(x_rad)
    cy, sy = math.cos(y_rad), math.sin(y_rad)
    cz, sz = math.cos(z_rad), math.sin(z_rad)

    matrix_x = (
        (1.0, 0.0, 0.0),
        (0.0, cx, -sx),
        (0.0, sx, cx),
    )
    matrix_y = (
        (cy, 0.0, sy),
        (0.0, 1.0, 0.0),
        (-sy, 0.0, cy),
    )
    matrix_z = (
        (cz, -sz, 0.0),
        (sz, cz, 0.0),
        (0.0, 0.0, 1.0),
    )

    return matrix_multiply(matrix_z, matrix_multiply(matrix_y, matrix_x))


def matrix_multiply(matrix_a, matrix_b):
    rows = []
    for row_index in range(3):
        row = []
        for column_index in range(3):
            row.append(
                sum(matrix_a[row_index][k_index] * matrix_b[k_index][column_index] for k_index in range(3))
            )
        rows.append(tuple(row))
    return tuple(rows)


def apply_matrix(matrix_rows, vector):
    return (
        (matrix_rows[0][0] * vector[0]) + (matrix_rows[0][1] * vector[1]) + (matrix_rows[0][2] * vector[2]),
        (matrix_rows[1][0] * vector[0]) + (matrix_rows[1][1] * vector[1]) + (matrix_rows[1][2] * vector[2]),
        (matrix_rows[2][0] * vector[0]) + (matrix_rows[2][1] * vector[1]) + (matrix_rows[2][2] * vector[2]),
    )


def transform_point(point, rotation_deg=(0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0), scale: float = 1.0):
    base_vector = vector_scale(as_vector(point), scale)
    rotation_matrix = rotation_matrix_from_euler_degrees(rotation_deg)
    rotated = apply_matrix(rotation_matrix, base_vector)
    translated = vector_add(rotated, translation)
    return vector_to_point(translated, template=point)
