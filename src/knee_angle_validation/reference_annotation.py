"""Reference-frame selection and manual annotation data helpers."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .knee_angle import calculate_knee_angle


REFERENCE_COLUMNS = [
    "video_id", "person_id", "frame_number", "side", "annotator_id",
    "annotation_session", "hip_x_ref", "hip_y_ref", "knee_x_ref", "knee_y_ref",
    "ankle_x_ref", "ankle_y_ref", "knee_angle_reference", "notes",
    "knee_internal_angle_deg", "knee_flexion_deg", "angle_geometry",
]


def select_reference_frames(
    data: pd.DataFrame,
    count: int = 30,
    *,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> pd.DataFrame:
    """Choose nearest frames to equal-time-bin midpoints, without model scores.

    Use the supplied time_s values (including irregular timestamps). At a tie
    choose the earlier frame. When targets map to the same frame, retain only
    its closest target (earlier target at a tie); do not substitute other frames.
    Thus a short/sparse recording may yield fewer than count unique frames.
    """
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("Liczba znaczników musi być dodatnią liczbą całkowitą.")
    if data.empty or not {"frame_number", "time_s"}.issubset(data.columns):
        raise ValueError("Wymagane są klatki z kolumnami frame_number i time_s.")
    for key in ("video_id", "person_id", "side"):
        if key in data and (data[key].isna().any() or data[key].nunique() != 1):
            raise ValueError(f"Wybieraj klatki osobno dla jednego {key}.")
    candidates = data.copy()
    for key in ("frame_number", "time_s"):
        values = pd.to_numeric(candidates[key], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Kolumna {key} musi zawierać skończone, nieujemne wartości.")
        if key == "frame_number" and (values != np.floor(values)).any():
            raise ValueError("Numery klatek muszą być całkowite.")
        candidates[key] = values.astype(int) if key == "frame_number" else values
    if candidates["frame_number"].duplicated().any():
        raise ValueError("Numery klatek nie mogą się powtarzać.")
    candidates = candidates.sort_values("frame_number").reset_index(drop=True)
    times = candidates["time_s"].to_numpy()
    if (np.diff(times) <= 0).any():
        raise ValueError("Czasy klatek muszą rosnąć wraz z numerami klatek.")
    start = float(times[0] if start_time_s is None else start_time_s)
    end = float(times[-1] if end_time_s is None else end_time_s)
    if not np.isfinite([start, end]).all() or not times[0] <= start <= end <= times[-1]:
        raise ValueError("Zakres czasu musi mieścić się w nagraniu; początek nie może być późniejszy niż koniec.")
    candidates = candidates.loc[candidates["time_s"].between(start, end)].reset_index(drop=True)
    if candidates.empty:
        raise ValueError("W wybranym zakresie czasu nie ma klatek.")
    times = candidates["time_s"].to_numpy()
    targets = start + (np.arange(count) + 0.5) * ((end - start) / count)
    right = np.clip(np.searchsorted(times, targets), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    left_distance, right_distance = np.abs(times[left] - targets), np.abs(times[right] - targets)
    positions = np.where((left_distance <= right_distance) |
                         np.isclose(left_distance, right_distance, atol=1e-12, rtol=0), left, right)
    selected = candidates.iloc[positions].copy()
    selected["selection_index"] = np.arange(1, count + 1)
    selected["selection_target_time_s"] = targets
    selected["selection_time_error_s"] = selected["time_s"].to_numpy() - targets
    selected["_distance"] = np.abs(selected["selection_time_error_s"])
    selected = selected.sort_values("_distance", kind="stable").drop_duplicates("frame_number")
    selected = selected.drop(columns="_distance").sort_values("frame_number").reset_index(drop=True)
    selected["selection_method"] = "time_bin_midpoints_v1"
    selected["selection_requested_count"] = count
    selected["selection_start_s"] = start
    selected["selection_end_s"] = end
    return selected


def build_reference_annotation(
    frame: pd.Series,
    points: Sequence[tuple[float, float]],
    annotator_id: str,
    annotation_session: str,
    notes: str = "",
) -> dict[str, object]:
    """Build one reference record from ordered hip, knee, and ankle clicks."""
    if len(points) != 3:
        raise ValueError("Exactly three points are required: hip, knee, ankle")
    hip, knee, ankle = points
    internal = calculate_knee_angle(hip, knee, ankle)
    return {
        "video_id": frame["video_id"], "person_id": frame["person_id"],
        "frame_number": int(frame["frame_number"]), "side": frame["side"],
        "annotator_id": annotator_id, "annotation_session": annotation_session,
        "hip_x_ref": hip[0], "hip_y_ref": hip[1],
        "knee_x_ref": knee[0], "knee_y_ref": knee[1],
        "ankle_x_ref": ankle[0], "ankle_y_ref": ankle[1],
        "knee_angle_reference": internal, "notes": notes,
        "knee_internal_angle_deg": internal, "knee_flexion_deg": 180.0 - internal,
        "angle_geometry": "pixel_2d_v1",
    }
