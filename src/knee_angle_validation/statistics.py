"""Descriptive validation statistics and visibility associations."""

import numpy as np
import pandas as pd


def error_statistics(data: pd.DataFrame) -> dict[str, float | int]:
    """Calculate core angle-error metrics for finite paired measurements."""
    numeric = pd.to_numeric(data["error_signed"], errors="coerce")
    error = numeric.loc[np.isfinite(numeric)]
    absolute = error.abs()
    if error.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan}
    return {
        "n": int(len(error)), "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error.pow(2)))), "bias": float(error.mean()),
        "median_absolute_error": float(absolute.median()), "error_std": float(error.std(ddof=1)),
        "error_min": float(error.min()), "error_max": float(error.max()),
        "within_3_deg_pct": float((absolute <= 3).mean() * 100),
        "within_5_deg_pct": float((absolute <= 5).mean() * 100),
        "within_10_deg_pct": float((absolute <= 10).mean() * 100),
    }


def grouped_statistics(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Calculate error statistics for each requested grouping column."""
    rows: list[dict[str, object]] = [{"group_type": "overall", "group_value": "all", **error_statistics(data)}]
    for column in group_columns:
        if column not in data:
            continue
        for value, group in data.groupby(column, dropna=False):
            rows.append({"group_type": column, "group_value": value, **error_statistics(group)})
    return pd.DataFrame(rows)


def reference_quality_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Count reference rows in the stated trial scope, including unmatched rows."""
    if "reference_scope" in data:
        data = data.loc[data.reference_scope & data.has_reference_row]
    labels = data.get("frame_quality", pd.Series("UNASSESSED", index=data.index))
    errors = pd.to_numeric(data["error_signed"], errors="coerce")
    return pd.DataFrame([
        {"frame_quality": label, "n_reference": int(labels.eq(label).sum()),
         "reference_pct": float(labels.eq(label).mean() * 100) if len(data) else np.nan,
         "n_valid_pairs": int((labels.eq(label) & np.isfinite(errors)).sum())}
        for label in ("CLEAR", "UNCERTAIN", "UNMARKABLE", "UNASSESSED")])


def grouped_reference_statistics(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Protocol: CLEAR primary; CLEAR+UNCERTAIN sensitivity; legacy separately."""
    if "reference_scope" in data:
        data = data.loc[data.reference_scope]
    labels = data.get("frame_quality", pd.Series("UNASSESSED", index=data.index))
    results = []
    subsets = [("primary_clear", ["CLEAR"]), ("sensitivity_clear_uncertain", ["CLEAR", "UNCERTAIN"])]
    if labels.eq("UNASSESSED").any():
        subsets.append(("legacy_unassessed", ["UNASSESSED"]))
    for name, allowed in subsets:
        selected = data.loc[labels.isin(allowed)]
        summary = grouped_statistics(selected, group_columns)
        summary["analysis_subset"] = name
        summary["n_reference_subset"] = len(selected)
        results.append(summary)
    return pd.concat(results, ignore_index=True)


def availability_statistics(data: pd.DataFrame) -> dict:
    """Separate full-video detection coverage from reference-trial availability."""
    scope = data.reference_scope
    mp, ref = data.mediapipe_available, data.reference_available
    planned = bool(data.selection_provided.all()) if len(data) else False
    n_source, n_scope = int(data.in_extraction.sum()), int(scope.sum())
    return {
        "scope_definition": "selected_frames" if planned else "annotation_rows_or_mixed",
        "n_frame_keys": len(data), "n_source_frames": n_source,
        "n_mediapipe_available": int(mp.sum()), "n_mediapipe_missing": int((data.in_extraction & ~mp).sum()),
        "mediapipe_available_pct": float(mp.sum() / n_source * 100) if n_source else np.nan,
        "n_selected_reference": int(data.selected_for_reference.sum()) if planned else np.nan,
        "n_reference_scope": n_scope,
        "n_reference_annotations": int((scope & data.has_reference_row).sum()),
        "n_not_annotated": int((scope & ~data.has_reference_row).sum()) if planned else np.nan,
        "n_reference_available": int((scope & ref).sum()),
        "n_reference_unavailable": int((scope & ~ref).sum()),
        "n_mediapipe_available_in_scope": int((scope & mp).sum()),
        "n_mediapipe_unavailable_in_scope": int((scope & ~mp).sum()),
        "n_paired_available": int((scope & mp & ref).sum()),
        "paired_available_pct": float((scope & mp & ref).sum() / n_scope * 100) if n_scope else np.nan,
        "n_only_mediapipe": int((scope & mp & ~ref).sum()),
        "n_only_reference": int((scope & ~mp & ref).sum()),
        "n_both_unavailable": int((scope & ~mp & ~ref).sum()),
        "n_primary_pairs": int((scope & mp & ref & data.frame_quality.eq("CLEAR")).sum()),
        "n_uncertain_reference": int((scope & data.frame_quality.eq("UNCERTAIN")).sum()),
        "n_unmarkable_reference": int((scope & data.frame_quality.eq("UNMARKABLE")).sum()),
        "n_unassessed_reference": int((scope & data.frame_quality.eq("UNASSESSED")).sum()),
        "n_reference_without_extraction": int((data.has_reference_row & ~data.in_extraction).sum()),
        "n_selected_without_extraction": int((data.selected_for_reference & ~data.in_extraction).sum()) if planned else np.nan,
        "n_reference_outside_selection": int((data.has_reference_row & ~data.selected_for_reference).sum()) if planned else np.nan,
        "n_recordings": len(data[[key for key in ("analysis_id", "video_id", "person_id", "side") if key in data]].drop_duplicates()),
        "n_persons": data.person_id.nunique(),
    }


def grouped_availability(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = [{"group_type": "overall", "group_value": "all", **availability_statistics(data)}]
    for column in group_columns:
        if column in data:
            for value, group in data.groupby(column, dropna=False):
                rows.append({"group_type": column, "group_value": value, **availability_statistics(group)})
    return pd.DataFrame(rows)


def recording_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """One descriptive result per recording and reference-quality subset."""
    keys = [key for key in ("analysis_id", "video_id", "person_id", "side", "speed_kmh") if key in data]
    results = []
    for identity, group in data.groupby(keys, dropna=False):
        summary = grouped_reference_statistics(group, [])
        for key, value in zip(keys, identity if isinstance(identity, tuple) else (identity,)):
            summary[key] = value
        results.append(summary)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def visibility_correlations(data: pd.DataFrame, minimum_n: int = 3) -> pd.DataFrame:
    """Calculate Pearson and Spearman correlations with absolute angle error."""
    rows = []
    for column in ("hip_visibility", "knee_visibility", "ankle_visibility", "min_visibility", "mean_visibility"):
        paired = data[[column, "error_absolute"]].dropna()
        if len(paired) < minimum_n or paired[column].nunique() < 2:
            rows.append({"feature": column, "n": len(paired), "pearson_r": np.nan, "spearman_r": np.nan})
            continue
        rows.append({
            "feature": column, "n": len(paired),
            "pearson_r": float(paired[column].corr(paired["error_absolute"], method="pearson")),
            "spearman_r": float(
                paired[column].rank().corr(paired["error_absolute"].rank(), method="pearson")
            ),
        })
    return pd.DataFrame(rows)
