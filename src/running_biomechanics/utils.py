import numpy as np


def to_float(value):
    if value is None:
        return np.nan

    value = str(value).strip()
    if value == "" or value.lower() in {"none", "nan"}:
        return np.nan

    return float(value)


def moving_avg_nan(values, k=9):
    values = np.asarray(values, dtype=float)
    result = np.full_like(values, np.nan, dtype=float)
    half = k // 2

    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        window = values[start:end]
        window = window[~np.isnan(window)]

        if len(window) > 0:
            result[i] = window.mean()

    return result
