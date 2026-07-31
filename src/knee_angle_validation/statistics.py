"""Descriptive validation statistics and visibility associations."""

import numpy as np
import pandas as pd


def error_statistics(data: pd.DataFrame) -> dict[str, float | int]:
    """Calculate core angle-error metrics for finite paired measurements."""
    error = pd.to_numeric(data["error_signed"], errors="coerce").dropna()
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
