import argparse
import csv

import numpy as np

from .config import ANALYSIS_VISIBILITY_THRESHOLD, CORE_LANDMARKS, DEFAULT_CSV_PATH
from .utils import moving_avg_nan, to_float


WINDOW_S = 6.0
SMOOTH_K = 5
MARGIN_S = 0.5
MIN_SEP_HARD = 0.35
CAD_MIN = 120
CAD_MAX = 220
MAX_LR_DIFF = 25
PROMINENCE_TH = 3


def read_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_xy(rows, name, visibility_threshold):
    x = np.array([to_float(r[f"{name}_x"]) for r in rows], dtype=float)
    y = np.array([to_float(r[f"{name}_y"]) for r in rows], dtype=float)
    vis = np.array([to_float(r[f"{name}_vis"]) for r in rows], dtype=float)

    x[vis < visibility_threshold] = np.nan
    y[vis < visibility_threshold] = np.nan

    return x, y, vis


def coverage_mask(xy_by_name, i0, i1):
    good = np.ones(i1 - i0, dtype=bool)

    for x, y, _ in xy_by_name.values():
        good &= ~np.isnan(x[i0:i1]) & ~np.isnan(y[i0:i1])

    return good


def pick_best_window(t, xy_by_name, win_s=WINDOW_S):
    duration = t[-1] - t[0]

    if duration <= win_s + 2 * MARGIN_S:
        i0 = np.searchsorted(t, t[0] + MARGIN_S, side="left")
        i1 = np.searchsorted(t, t[-1] - MARGIN_S, side="right")

        if i1 - i0 < 10:
            i0 = 0
            i1 = len(t)

        good = coverage_mask(xy_by_name, i0, i1)
        return i0, i1, float(np.mean(good))

    best = None
    best_score = -1
    start_min = t[0] + MARGIN_S
    start_max = t[-1] - MARGIN_S - win_s

    for i0, t0 in enumerate(t):
        if t0 < start_min:
            continue

        if t0 > start_max:
            break

        i1 = np.searchsorted(t, t0 + win_s, side="right")
        if i1 - i0 < 10:
            continue

        good = coverage_mask(xy_by_name, i0, i1)
        score = float(np.mean(good))

        if score > best_score:
            best_score = score
            best = (i0, i1, score)

    return best


def find_local_minima_strong(values):
    idx = []

    for i in range(2, len(values) - 2):
        if np.any(np.isnan(values[i - 2:i + 3])):
            continue

        is_min = values[i] < values[i - 1] and values[i] < values[i + 1]
        left_peak = max(values[i - 2], values[i - 1])
        right_peak = max(values[i + 1], values[i + 2])
        prominence = min(left_peak, right_peak) - values[i]

        if is_min and prominence >= PROMINENCE_TH:
            idx.append(i)

    return np.array(idx, dtype=int)


def median_period(idx, t):
    if len(idx) < 2:
        return np.nan

    periods = np.diff(t[idx])
    periods = periods[(periods > 0.25) & (periods < 1.2)]

    if len(periods) == 0:
        return np.nan

    return np.median(periods)


def filter_minima_by_sep(idx, t, min_sep_s):
    keep = []
    last_t = -1e9

    for k in idx:
        if t[k] - last_t >= min_sep_s:
            keep.append(k)
            last_t = t[k]

    return np.array(keep, dtype=int)


def cadence(idx, t):
    if len(idx) < 2:
        return np.nan

    periods = np.diff(t[idx])
    periods = periods[(periods >= MIN_SEP_HARD) & (periods <= 1.2)]

    if len(periods) == 0:
        return np.nan

    return (1 / np.mean(periods)) * 60 * 2


def analyze_csv(csv_path=DEFAULT_CSV_PATH, visibility_threshold=ANALYSIS_VISIBILITY_THRESHOLD):
    rows = read_rows(csv_path)
    if len(rows) < 10:
        raise RuntimeError("Too little data in CSV.")

    t = np.array([to_float(r["t_ms"]) / 1000.0 for r in rows], dtype=float)
    xy_by_name = {name: get_xy(rows, name, visibility_threshold) for name in CORE_LANDMARKS}

    left_knee = np.array([to_float(r.get("left_knee_deg", np.nan)) for r in rows], dtype=float)
    right_knee = np.array([to_float(r.get("right_knee_deg", np.nan)) for r in rows], dtype=float)
    left_knee_s = moving_avg_nan(left_knee, SMOOTH_K)
    right_knee_s = moving_avg_nan(right_knee, SMOOTH_K)

    window = pick_best_window(t, xy_by_name, WINDOW_S)
    if window is None:
        raise RuntimeError("Could not find a useful analysis window.")

    i0, i1, coverage = window
    t_w = t[i0:i1]
    left_w = left_knee_s[i0:i1]
    right_w = right_knee_s[i0:i1]

    left_min0 = find_local_minima_strong(left_w)
    right_min0 = find_local_minima_strong(right_w)
    p_left = median_period(left_min0, t_w)
    p_right = median_period(right_min0, t_w)
    period = np.nanmedian([p_left, p_right])

    if np.isnan(period):
        raise RuntimeError("Could not estimate step period. Not enough stable minima.")

    min_sep = max(MIN_SEP_HARD, 0.6 * period)
    left_min = filter_minima_by_sep(left_min0, t_w, min_sep)
    right_min = filter_minima_by_sep(right_min0, t_w, min_sep)

    cadence_left = cadence(left_min, t_w)
    cadence_right = cadence(right_min, t_w)
    cadence_all = np.nanmean([cadence_left, cadence_right])
    events = len(left_min) + len(right_min)

    if np.isnan(cadence_left) or np.isnan(cadence_right) or np.isnan(cadence_all):
        diff_lr = np.nan
    else:
        diff_lr = abs(cadence_left - cadence_right)

    if np.isnan(cadence_all):
        confidence = "low"
    elif cadence_all < CAD_MIN or cadence_all > CAD_MAX:
        confidence = "low"
    elif not np.isnan(diff_lr) and diff_lr > MAX_LR_DIFF:
        confidence = "medium"
    elif coverage > 0.75 and events >= 8:
        confidence = "high"
    elif coverage > 0.55 and events >= 5:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "window_start_s": float(t_w[0]),
        "window_end_s": float(t_w[-1]),
        "coverage": float(coverage),
        "cadence": float(cadence_all),
        "cadence_left": float(cadence_left),
        "cadence_right": float(cadence_right),
        "events": int(events),
        "lr_cadence_difference": float(diff_lr),
        "confidence": confidence,
    }


def print_analysis(result):
    print(
        "Selected window: "
        f"{result['window_start_s']:.2f}s - {result['window_end_s']:.2f}s, "
        f"coverage={result['coverage'] * 100:.1f}%"
    )
    print(
        "Cadence ~ "
        f"{result['cadence']:.1f} spm "
        f"(L={result['cadence_left']:.1f}, R={result['cadence_right']:.1f})"
    )
    print("Events:", result["events"])
    print("L/R cadence difference:", result["lr_cadence_difference"])
    print("Confidence:", result["confidence"])


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze running landmarks CSV.")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Input landmarks CSV path.")
    parser.add_argument(
        "--min-vis",
        type=float,
        default=ANALYSIS_VISIBILITY_THRESHOLD,
        help="Minimum landmark visibility used for analysis.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = analyze_csv(args.csv, args.min_vis)
    print_analysis(result)

