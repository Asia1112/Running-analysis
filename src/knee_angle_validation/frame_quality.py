"""Frame-level visibility, temporal stability, and reliability assessment."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, ValidationConfig
from .models import FrameQuality


def classify_visibility(value: float, config: ValidationConfig = DEFAULT_CONFIG) -> str:
    """Assign a visibility value to a configured analysis bin."""
    if not np.isfinite(value):
        return "missing"
    low, medium, high = config.visibility_bins
    if value < low:
        return "<0.60"
    if value < medium:
        return "0.60-0.79"
    if value < high:
        return "0.80-0.89"
    return ">=0.90"


def angle_changes(values: Iterable[float]) -> np.ndarray:
    """Return absolute frame-to-frame angle differences."""
    array = np.asarray(list(values), dtype=float)
    return np.abs(np.diff(array, prepend=np.nan))


def detect_angle_jumps(values: Iterable[float], threshold_deg: float) -> np.ndarray:
    """Return a boolean mask for changes exceeding a chosen threshold."""
    return angle_changes(values) > threshold_deg


def point_acceleration(x: Iterable[float], y: Iterable[float]) -> np.ndarray:
    """Return the magnitude of the second positional difference."""
    points = np.column_stack((np.asarray(list(x), dtype=float), np.asarray(list(y), dtype=float)))
    acceleration = np.full(len(points), np.nan)
    if len(points) >= 3:
        acceleration[2:] = np.linalg.norm(np.diff(points, n=2, axis=0), axis=1)
    return acceleration


def assess_frame_quality(
    complete: bool,
    min_visibility: float,
    angle_jump_deg: float,
    smoothing_residual_deg: float,
    point_stability_error: float,
    config: ValidationConfig = DEFAULT_CONFIG,
) -> FrameQuality:
    """Classify one frame using configurable, auditable measurement-quality rules."""
    reasons: list[str] = []
    severity = 0
    if not complete:
        reasons.append("missing required point")
        severity = 2
    if not np.isfinite(min_visibility) or min_visibility < config.unreliable_visibility:
        reasons.append(f"min_visibility < {config.unreliable_visibility:.2f}")
        severity = 2
    elif min_visibility < config.review_visibility:
        reasons.append(f"min_visibility < {config.review_visibility:.2f}")
        severity = max(severity, 1)
    if np.isfinite(angle_jump_deg) and angle_jump_deg > config.jump_thresholds_deg[-1]:
        reasons.append(f"angle jump > {config.jump_thresholds_deg[-1]:.0f} deg")
        severity = 2
    elif np.isfinite(angle_jump_deg) and angle_jump_deg > config.jump_thresholds_deg[0]:
        reasons.append(f"angle jump > {config.jump_thresholds_deg[0]:.0f} deg")
        severity = max(severity, 1)
    if np.isfinite(smoothing_residual_deg) and smoothing_residual_deg > config.residual_unreliable_deg:
        reasons.append("large deviation from local trend")
        severity = 2
    elif np.isfinite(smoothing_residual_deg) and smoothing_residual_deg > config.residual_review_deg:
        reasons.append("deviation from local trend")
        severity = max(severity, 1)
    if np.isfinite(point_stability_error) and point_stability_error > config.point_acceleration_unreliable:
        reasons.append("large landmark acceleration")
        severity = 2
    elif np.isfinite(point_stability_error) and point_stability_error > config.point_acceleration_review:
        reasons.append("landmark acceleration requires review")
        severity = max(severity, 1)
    return FrameQuality(("reliable", "needs_review", "unreliable")[severity], reasons)


def add_quality_columns(data: pd.DataFrame, smoothed_column: str = "knee_angle_smoothed") -> pd.DataFrame:
    """Add temporal features, quality status, and reasons to extracted frames."""
    result = data.copy()
    visibility = result[["hip_visibility", "knee_visibility", "ankle_visibility"]]
    result["min_visibility"] = visibility.min(axis=1, skipna=False)
    result["mean_visibility"] = visibility.mean(axis=1, skipna=False)
    result["angle_jump_deg"] = angle_changes(result["knee_angle_raw"])
    result["smoothing_residual_deg"] = (result["knee_angle_raw"] - result[smoothed_column]).abs()
    accelerations = []
    for point in ("hip", "knee", "ankle"):
        accelerations.append(point_acceleration(result[f"{point}_x_norm"], result[f"{point}_y_norm"]))
    result["point_stability_error"] = np.nanmax(np.column_stack(accelerations), axis=1)
    qualities = [
        assess_frame_quality(
            bool(row.pose_found) and np.isfinite(row.knee_angle_raw), row.min_visibility,
            row.angle_jump_deg, row.smoothing_residual_deg, row.point_stability_error,
        )
        for row in result.itertuples()
    ]
    result["frame_quality_status"] = [item.status for item in qualities]
    result["suspicious_reason"] = ["; ".join(item.reasons) for item in qualities]
    return result
