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
    """Count every matched reference row, including unavailable angles."""
    labels = data.get("frame_quality", pd.Series("UNASSESSED", index=data.index))
    errors = pd.to_numeric(data["error_signed"], errors="coerce")
    return pd.DataFrame([
        {"frame_quality": label, "n_reference": int(labels.eq(label).sum()),
         "reference_pct": float(labels.eq(label).mean() * 100) if len(data) else np.nan,
         "n_valid_pairs": int((labels.eq(label) & np.isfinite(errors)).sum())}
        for label in ("CLEAR", "UNCERTAIN", "UNMARKABLE", "UNASSESSED")])


def grouped_reference_statistics(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Protocol: CLEAR primary; CLEAR+UNCERTAIN sensitivity; legacy separately."""
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
