"""Join MediaPipe measurements with reference annotations and compute errors."""

import numpy as np
import pandas as pd

from .knee_angle import add_pixel_angle_columns


JOIN_KEYS = ["video_id", "person_id", "frame_number", "side"]


def _point_distance(data: pd.DataFrame, point: str) -> pd.Series:
    """Calculate MediaPipe-reference pixel distance for one anatomical point."""
    return np.hypot(
        data[f"{point}_x_px"] - data[f"{point}_x_ref"],
        data[f"{point}_y_px"] - data[f"{point}_y_ref"],
    )


def merge_with_reference(extracted: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Inner-join extracted frames and manual references, then calculate errors."""
    missing = [key for key in JOIN_KEYS if key not in extracted or key not in reference]
    if missing:
        raise ValueError(f"Missing join keys: {', '.join(missing)}")
    # Always derive both methods from pixels, including older CSV inputs.
    extracted = add_pixel_angle_columns(extracted)
    reference = add_pixel_angle_columns(reference, reference=True)
    merged = extracted.merge(reference, on=JOIN_KEYS, how="inner", validate="many_to_many",
                             suffixes=("_mediapipe", "_reference"))
    merged["angle_mediapipe"] = merged["knee_flexion_deg_mediapipe"]
    merged["angle_reference"] = merged["knee_flexion_deg_reference"]
    merged["angle_definition"] = "knee_flexion_2d_deg"
    merged["error_signed"] = merged["angle_mediapipe"] - merged["angle_reference"]
    merged["error_internal_signed"] = (
        merged["knee_internal_angle_deg_mediapipe"] - merged["knee_internal_angle_deg_reference"]
    )
    merged["error_absolute"] = merged["error_signed"].abs()
    merged["squared_error"] = merged["error_signed"].pow(2)
    for point in ("hip", "knee", "ankle"):
        merged[f"{point}_distance_px"] = _point_distance(merged, point)
    thigh_length = np.hypot(
        merged["hip_x_ref"] - merged["knee_x_ref"],
        merged["hip_y_ref"] - merged["knee_y_ref"],
    ).replace(0, np.nan)
    for point in ("hip", "knee", "ankle"):
        merged[f"{point}_distance_thigh_norm"] = merged[f"{point}_distance_px"] / thigh_length
    return merged
