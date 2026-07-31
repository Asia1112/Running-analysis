"""NaN-aware smoothing methods used for comparison experiments."""

import numpy as np


def _validate_window(window: int) -> None:
    """Validate a positive smoothing window."""
    if window < 1:
        raise ValueError("window must be positive")


def _interpolate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate internal NaNs while retaining the original missing mask."""
    result = np.asarray(values, dtype=float).copy()
    missing = ~np.isfinite(result)
    valid = np.flatnonzero(~missing)
    if not valid.size:
        return result, missing
    result[missing] = np.interp(np.flatnonzero(missing), valid, result[valid])
    return result, missing


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Return a centered moving average without introducing temporal delay."""
    _validate_window(window)
    array = np.asarray(values, dtype=float)
    kernel = np.ones(window, dtype=float)
    valid = np.isfinite(array).astype(float)
    sums = np.convolve(np.nan_to_num(array), kernel, mode="same")
    counts = np.convolve(valid, kernel, mode="same")
    return np.divide(sums, counts, out=np.full_like(array, np.nan), where=counts > 0)


def median_filter(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Return a centered NaN-aware median filter."""
    _validate_window(window)
    array = np.asarray(values, dtype=float)
    radius = window // 2
    return np.array([
        np.nanmedian(array[max(0, i - radius):min(len(array), i + radius + 1)])
        for i in range(len(array))
    ])


def savitzky_golay(values: np.ndarray, window: int = 7, order: int = 2) -> np.ndarray:
    """Return a Savitzky-Golay curve, preserving positions missing in the input."""
    array = np.asarray(values, dtype=float)
    if len(array) < 3 or np.isfinite(array).sum() < 3:
        return array.copy()
    effective = min(window, len(array) if len(array) % 2 else len(array) - 1)
    effective = max(effective, order + 1 + (order + 1) % 2)
    if effective > len(array):
        return array.copy()
    interpolated, missing = _interpolate(array)
    radius = effective // 2
    result = np.empty_like(interpolated)
    for index in range(len(interpolated)):
        start = max(0, index - radius)
        stop = min(len(interpolated), index + radius + 1)
        x = np.arange(start, stop, dtype=float) - index
        degree = min(order, len(x) - 1)
        coefficients = np.polyfit(x, interpolated[start:stop], degree)
        result[index] = np.polyval(coefficients, 0.0)
    result[missing] = np.nan
    return result


def smooth_series(values: np.ndarray, method: str, window: int = 5) -> np.ndarray:
    """Apply one of the supported, explicitly named smoothing methods."""
    methods = {
        "moving_average": lambda: moving_average(values, window),
        "median": lambda: median_filter(values, window),
        "savgol": lambda: savitzky_golay(values, window if window % 2 else window + 1),
    }
    if method not in methods:
        raise ValueError(f"Unsupported smoothing method: {method}")
    return methods[method]()
