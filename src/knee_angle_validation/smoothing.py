"""Optional frame-window filters that never interpolate or cross missing data."""

import numpy as np


def _validate_window(window: int) -> None:
    """Validate a positive smoothing window."""
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")


def _array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional signal")
    return array


def _segments(array: np.ndarray):
    """Yield finite runs, with NaN and infinity both acting as boundaries."""
    edges = np.flatnonzero(np.diff(np.r_[False, np.isfinite(array), False]))
    for start, stop in zip(edges[::2], edges[1::2]):
        yield int(start), int(stop)


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered mean within each finite run; shrink windows at run boundaries."""
    _validate_window(window)
    array = _array(values)
    result = np.full_like(array, np.nan)
    radius = window // 2
    for start, stop in _segments(array):
        for index in range(start, stop):
            result[index] = np.mean(array[max(start, index - radius):min(stop, index + radius + 1)])
    return result


def median_filter(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered median within each finite run, preserving every missing position."""
    _validate_window(window)
    array = _array(values)
    radius = window // 2
    result = np.full_like(array, np.nan)
    for start, stop in _segments(array):
        for index in range(start, stop):
            result[index] = np.median(array[max(start, index - radius):min(stop, index + radius + 1)])
    return result


def savitzky_golay(values: np.ndarray, window: int = 7, order: int = 2) -> np.ndarray:
    """Local polynomial fits inside finite runs, without filling or bridging gaps.

    Runs shorter than order+1 samples are copied. At boundaries the window and
    degree are reduced to the available samples, rather than extrapolating.
    """
    _validate_window(window)
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or not 0 <= order < window:
        raise ValueError("order must be a nonnegative integer smaller than window")
    array = _array(values)
    result = np.full_like(array, np.nan)
    radius = window // 2
    for start, stop in _segments(array):
        if stop - start < order + 1:
            result[start:stop] = array[start:stop]
            continue
        for index in range(start, stop):
            left, right = max(start, index - radius), min(stop, index + radius + 1)
            x = np.arange(left, right, dtype=float) - index
            coefficients = np.polyfit(x, array[left:right], min(order, len(x) - 1))
            result[index] = np.polyval(coefficients, 0.0)
    return result


def smooth_series(values: np.ndarray, method: str, window: int = 5) -> np.ndarray:
    """Apply one of the supported, explicitly named smoothing methods."""
    methods = {
        "moving_average": lambda: moving_average(values, window),
        "median": lambda: median_filter(values, window),
        "savgol": lambda: savitzky_golay(values, window),
    }
    if method not in methods:
        raise ValueError(f"Unsupported smoothing method: {method}")
    return methods[method]()
