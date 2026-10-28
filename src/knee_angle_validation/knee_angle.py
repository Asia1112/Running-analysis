"""Geometry for MediaPipe and manually annotated knee points."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def _point_array(point: Any) -> np.ndarray | None:
    """Convert a point-like value to a finite two-element array."""
    if point is None:
        return None
    if hasattr(point, "x") and hasattr(point, "y"):
        values = (point.x, point.y)
    elif isinstance(point, (Sequence, np.ndarray)) and not isinstance(point, (str, bytes)):
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
    """Return the internal 2D angle (extension=180°), using equal-scale axes.

    Pass image pixels, not x/width and y/height. Invalid or collapsed
    segments return NaN. This unsigned angle does not measure hyperextension.
    """
    hip_point = _point_array(hip)
    knee_point = _point_array(knee)
    ankle_point = _point_array(ankle)
    if hip_point is None or knee_point is None or ankle_point is None:
        return float("nan")

    thigh = hip_point - knee_point
    shin = ankle_point - knee_point
    thigh_norm = float(np.linalg.norm(thigh))
    shin_norm = float(np.linalg.norm(shin))
    if thigh_norm <= 1e-6 or shin_norm <= 1e-6:
        return float("nan")

    cosine = float(np.dot(thigh, shin) / (thigh_norm * shin_norm))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def add_pixel_angle_columns(data: pd.DataFrame, reference: bool = False) -> pd.DataFrame:
    """Recalculate both raw angles from saved pixels, preserving the input.

    Legacy angle aliases remain internal angles. Existing filtered values
    are removed because they may have been computed in distorted geometry.
    No files are written and no MediaPipe inference is needed.
    """
    suffix = "ref" if reference else "px"
    columns = [f"{point}_{axis}_{suffix}" for point in ("hip", "knee", "ankle") for axis in ("x", "y")]
    missing = set(columns).difference(data.columns)
    if missing:
        raise ValueError(f"Brak współrzędnych pikselowych: {', '.join(sorted(missing))}")
    result = data.copy()
    internal = [calculate_knee_angle(values[:2], values[2:4], values[4:6])
                for values in result[columns].itertuples(index=False, name=None)]
    result["knee_internal_angle_deg"] = pd.Series(internal, index=result.index, dtype=float)
    result["knee_flexion_deg"] = 180.0 - result["knee_internal_angle_deg"]
    result["knee_angle_reference" if reference else "knee_angle_raw"] = result["knee_internal_angle_deg"]
    result["angle_geometry"] = "pixel_2d_v1"
    if not reference:
        result = result.drop(columns=["knee_angle_smoothed", "knee_flexion_smoothed_deg"], errors="ignore")
    return result
