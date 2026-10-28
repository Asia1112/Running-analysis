"""Join MediaPipe measurements with reference annotations and compute errors."""

import numpy as np
import pandas as pd

from .knee_angle import add_pixel_angle_columns
from .annotation_storage import select_reference_session
from .provenance import PROVENANCE_KEYS, RANGE_KEYS, TIMING_KEYS, recording_metadata
from .experiment_metadata import EXPERIMENT_KEYS, experiment_values


JOIN_KEYS = ["video_id", "person_id", "frame_number", "side"]


def _point_distance(data: pd.DataFrame, point: str) -> pd.Series:
    """Calculate MediaPipe-reference pixel distance for one anatomical point."""
    return np.hypot(
        data[f"{point}_x_px"] - data[f"{point}_x_ref"],
        data[f"{point}_y_px"] - data[f"{point}_y_ref"],
    )


def _unique_measurements(data: pd.DataFrame, label: str) -> pd.DataFrame:
    """Validate and normalize frame identities without discarding any records."""
    if not data.columns.is_unique:
        raise ValueError(f"{label}: powielone nazwy kolumn.")
    missing = [key for key in JOIN_KEYS if key not in data]
    if missing:
        raise ValueError(f"{label}: brak kluczy: {', '.join(missing)}")
    normalized = data.copy()
    for key in ("video_id", "person_id", "side"):
        if normalized[key].isna().any():
            raise ValueError(f"{label}: puste pole {key}.")
        normalized[key] = normalized[key].astype(str).str.strip()
        if normalized[key].eq("").any():
            raise ValueError(f"{label}: puste pole {key}.")
    frames = pd.to_numeric(normalized["frame_number"], errors="coerce")
    if (not np.isfinite(frames).all() or (frames < 0).any()
            or (frames % 1 != 0).any() or (frames >= 2**63).any()):
        raise ValueError(f"{label}: numery klatek muszą być nieujemnymi liczbami całkowitymi.")
    normalized["frame_number"] = frames.astype("int64")
    if not normalized["side"].isin(["left", "right"]).all():
        raise ValueError(f"{label}: strona musi mieć wartość left albo right.")
    duplicates = normalized.duplicated(JOIN_KEYS, keep=False)
    if duplicates.any():
        examples = normalized.loc[duplicates, JOIN_KEYS].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"{label}: powielone klucze pomiarów ({int(duplicates.sum())} wierszy). "
                         f"Przykłady: {examples}. Nie usunięto ani nie uśredniono rekordów.")
    return normalized


def merge_with_reference(extracted: pd.DataFrame, reference: pd.DataFrame, *,
                         annotator_id: str | None = None,
                         session: str | None = None) -> pd.DataFrame:
    """Compare unique frames against exactly one annotator/session, one-to-one.

    Multiple reference sessions require explicit selection. A reference containing
    just one pair is unambiguous. Duplicate measurements are always errors.
    """
    if not reference.columns.is_unique:
        raise ValueError("Referencja: powielone nazwy kolumn.")
    reference = select_reference_session(reference, annotator_id, session)
    extracted = _unique_measurements(extracted, "MediaPipe")
    reference = _unique_measurements(reference, "Referencja")
    if set(PROVENANCE_KEYS).intersection(extracted.columns) or set(PROVENANCE_KEYS).intersection(reference.columns):
        if recording_metadata(extracted) != recording_metadata(reference):
            raise ValueError("MediaPipe i referencja pochodzą z innego filmu, analizy lub geometrii obrazu.")
        reference = reference.drop(columns=list(PROVENANCE_KEYS))
        reference = reference.drop(columns=list(RANGE_KEYS), errors="ignore")
        reference = reference.drop(columns=list(TIMING_KEYS), errors="ignore")
    for key in ("time_s", "source_pts_s", "time_pts_s"):
        if key in reference and key in extracted:
            paired = extracted[JOIN_KEYS + [key]].merge(reference[JOIN_KEYS + [key]], on=JOIN_KEYS, validate="one_to_one")
            if not np.isclose(pd.to_numeric(paired[f"{key}_x"], errors="coerce"),
                              pd.to_numeric(paired[f"{key}_y"], errors="coerce"), atol=1e-9, rtol=0).all():
                raise ValueError(f"MediaPipe i referencja mają inny czas klatki: {key}.")
            reference = reference.drop(columns=key)
    if set(EXPERIMENT_KEYS).intersection(extracted.columns) or set(EXPERIMENT_KEYS).intersection(reference.columns):
        if experiment_values(extracted) != experiment_values(reference):
            raise ValueError("MediaPipe i referencja mają inne metadane eksperymentu.")
        reference = reference.drop(columns=list(EXPERIMENT_KEYS))
    # Always derive both methods from pixels, including older CSV inputs.
    extracted = add_pixel_angle_columns(extracted)
    reference = add_pixel_angle_columns(reference, reference=True)
    merged = extracted.merge(reference, on=JOIN_KEYS, how="inner", validate="one_to_one",
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
