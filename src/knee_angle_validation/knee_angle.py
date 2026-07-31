"""Geometry for MediaPipe and manually annotated knee points."""

from collections.abc import Sequence
from typing import Any

import numpy as np


def _point_array(point: Any) -> np.ndarray | None:
    """Convert a point-like value to a finite two-element array."""
    if point is None:
        return None
    if hasattr(point, "x") and hasattr(point, "y"):
        values = (point.x, point.y)
    elif isinstance(point, Sequence) and not isinstance(point, (str, bytes)):
        values = point
    else:
        return None
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size < 2 or not np.all(np.isfinite(array[:2])):
        return None
    return array[:2]


def calculate_knee_angle(hip: Any, knee: Any, ankle: Any) -> float:
    """Return the 2D hip-knee-ankle angle in degrees, or NaN for invalid data."""
    hip_point = _point_array(hip)
    knee_point = _point_array(knee)
    ankle_point = _point_array(ankle)
    if hip_point is None or knee_point is None or ankle_point is None:
        return float("nan")

    thigh = hip_point - knee_point
    shin = ankle_point - knee_point
    thigh_norm = float(np.linalg.norm(thigh))
    shin_norm = float(np.linalg.norm(shin))
    if thigh_norm == 0.0 or shin_norm == 0.0:
        return float("nan")

    cosine = float(np.dot(thigh, shin) / (thigh_norm * shin_norm))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
