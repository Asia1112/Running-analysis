import argparse
import csv
import json

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


def safe_nanmean(values):
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return np.nan

    return float(np.nanmean(values))


def safe_nanmin(values):
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return np.nan

    return float(np.nanmin(values))


def safe_nanmax(values):
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return np.nan

    return float(np.nanmax(values))


def safe_nanstd(values):
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return np.nan

    return float(np.nanstd(values))


def nanmean_pair(a, b):
    result = np.full_like(a, np.nan, dtype=float)
    a_ok = ~np.isnan(a)
    b_ok = ~np.isnan(b)
    both_ok = a_ok & b_ok

    result[both_ok] = (a[both_ok] + b[both_ok]) / 2
    result[a_ok & ~b_ok] = a[a_ok & ~b_ok]
    result[~a_ok & b_ok] = b[~a_ok & b_ok]

    return result


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


def segment_angle_deg(x1, y1, x2, y2):
    """Angle of a segment relative to the horizontal image axis."""
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def trunk_lean_deg(xy_by_name):
    l_sh_x, l_sh_y, _ = xy_by_name["L_SHOULDER"]
    r_sh_x, r_sh_y, _ = xy_by_name["R_SHOULDER"]
    l_hip_x, l_hip_y, _ = xy_by_name["L_HIP"]
    r_hip_x, r_hip_y, _ = xy_by_name["R_HIP"]

    shoulder_x = nanmean_pair(l_sh_x, r_sh_x)
    shoulder_y = nanmean_pair(l_sh_y, r_sh_y)
    hip_x = nanmean_pair(l_hip_x, r_hip_x)
    hip_y = nanmean_pair(l_hip_y, r_hip_y)

    # Image y grows downward. A side-view trunk close to vertical is near 0 deg here.
    return np.degrees(np.arctan2(shoulder_x - hip_x, hip_y - shoulder_y))


def hip_center_y(xy_by_name):
    _, l_hip_y, _ = xy_by_name["L_HIP"]
    _, r_hip_y, _ = xy_by_name["R_HIP"]
    return nanmean_pair(l_hip_y, r_hip_y)


def estimate_stride_symmetry(cadence_left, cadence_right, left_knee_rom, right_knee_rom):
    cadence_diff = abs(cadence_left - cadence_right)
    knee_rom_diff = abs(left_knee_rom - right_knee_rom)

    score = 100.0
    if not np.isnan(cadence_diff):
        score -= min(35.0, cadence_diff * 1.5)
    if not np.isnan(knee_rom_diff):
        score -= min(35.0, knee_rom_diff * 1.2)

    return float(max(0.0, min(100.0, score)))


def classify_metric_flags(result):
    flags = []

    cadence = result["cadence"]
    if not np.isnan(cadence):
        if cadence < 160:
            flags.append("Cadence is relatively low for running; possible overstriding risk.")
        elif cadence > 200:
            flags.append("Cadence is very high; verify whether gait events were detected correctly.")

    if result["lr_cadence_difference"] > 15:
        flags.append("Left-right cadence difference is elevated; inspect gait-event detection and symmetry.")

    if result["knee_rom_difference_deg"] > 12:
        flags.append("Left-right knee range of motion differs noticeably.")

    if abs(result["trunk_lean_mean_deg"]) > 15:
        flags.append("Average trunk lean is high in the selected side-view window.")

    if result["vertical_oscillation_proxy"] > 0.035:
        flags.append("Hip vertical oscillation proxy is high; this can indicate excessive bouncing.")

    if result["coverage"] < 0.75 or result["core_visible_pct"] < 75:
        flags.append("Pose quality is limited; report should be treated as lower confidence.")

    return flags


def report_status(result):
    """App-facing decision about whether the automatic report is trustworthy."""
    if (
        result["coverage"] < 0.70
        or result["core_visible_pct"] < 75
        or result["events"] < 5
        or result["cadence"] < 140
        or result["cadence"] > 220
    ):
        return "low_confidence"

    if (
        result["cadence"] < 160
        or result["cadence"] > 200
        or result["coverage"] < 0.80
        or result["core_visible_pct"] < 85
        or result["lr_cadence_difference"] > 15
        or result["knee_rom_difference_deg"] > 12
        or abs(result["trunk_lean_mean_deg"]) > 15
    ):
        return "needs_review"

    return "usable"


def report_recommendation(status):
    if status == "usable":
        return "Automatic assessment can be shown to the user."
    if status == "needs_review":
        return "Show the report, but mark detected irregularities as requiring verification."

    return "Do not make a strong technique decision from this video; ask for a better recording."


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
    trunk_w = trunk_lean_deg(xy_by_name)[i0:i1]
    hip_y_w = hip_center_y(xy_by_name)[i0:i1]

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
    left_knee_min = safe_nanmin(left_w)
    left_knee_max = safe_nanmax(left_w)
    right_knee_min = safe_nanmin(right_w)
    right_knee_max = safe_nanmax(right_w)
    left_knee_rom = left_knee_max - left_knee_min
    right_knee_rom = right_knee_max - right_knee_min

    core_visible_values = np.array([to_float(r.get("core_visible", np.nan)) for r in rows], dtype=float)
    core_quality_values = np.array([to_float(r.get("core_quality", np.nan)) for r in rows], dtype=float)
    if np.all(np.isnan(core_visible_values)):
        core_visible_pct = coverage * 100
    else:
        core_visible_pct = safe_nanmean(core_visible_values[i0:i1]) * 100

    vertical_oscillation_proxy = safe_nanstd(hip_y_w)
    symmetry_score = estimate_stride_symmetry(
        cadence_left,
        cadence_right,
        left_knee_rom,
        right_knee_rom,
    )

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

    result = {
        "window_start_s": float(t_w[0]),
        "window_end_s": float(t_w[-1]),
        "coverage": float(coverage),
        "core_visible_pct": float(core_visible_pct),
        "core_quality_mean": safe_nanmean(core_quality_values[i0:i1]),
        "cadence": float(cadence_all),
        "cadence_left": float(cadence_left),
        "cadence_right": float(cadence_right),
        "events": int(events),
        "lr_cadence_difference": float(diff_lr),
        "left_knee_min_deg": float(left_knee_min),
        "left_knee_max_deg": float(left_knee_max),
        "left_knee_rom_deg": float(left_knee_rom),
        "right_knee_min_deg": float(right_knee_min),
        "right_knee_max_deg": float(right_knee_max),
        "right_knee_rom_deg": float(right_knee_rom),
        "knee_rom_difference_deg": float(abs(left_knee_rom - right_knee_rom)),
        "trunk_lean_mean_deg": safe_nanmean(trunk_w),
        "trunk_lean_std_deg": safe_nanstd(trunk_w),
        "vertical_oscillation_proxy": float(vertical_oscillation_proxy),
        "symmetry_score": float(symmetry_score),
        "confidence": confidence,
    }
    result["flags"] = classify_metric_flags(result)
    result["report_status"] = report_status(result)
    result["recommendation"] = report_recommendation(result["report_status"])

    return result


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
    print(
        "Knee ROM: "
        f"L={result['left_knee_rom_deg']:.1f} deg, "
        f"R={result['right_knee_rom_deg']:.1f} deg, "
        f"diff={result['knee_rom_difference_deg']:.1f} deg"
    )
    print(
        "Trunk lean: "
        f"mean={result['trunk_lean_mean_deg']:.1f} deg, "
        f"std={result['trunk_lean_std_deg']:.1f} deg"
    )
    print(f"Vertical oscillation proxy: {result['vertical_oscillation_proxy']:.4f}")
    print(f"Symmetry score: {result['symmetry_score']:.1f}/100")
    print(f"Core visible: {result['core_visible_pct']:.1f}%")
    print("Confidence:", result["confidence"])
    print("Report status:", result["report_status"])
    print("Recommendation:", result["recommendation"])

    if result["flags"]:
        print("Flags:")
        for flag in result["flags"]:
            print(f"- {flag}")
    else:
        print("Flags: none")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze running landmarks CSV.")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Input landmarks CSV path.")
    parser.add_argument(
        "--min-vis",
        type=float,
        default=ANALYSIS_VISIBILITY_THRESHOLD,
        help="Minimum landmark visibility used for analysis.",
    )
    parser.add_argument("--json", help="Optional path for a JSON report.")
    return parser.parse_args()


def main():
    args = parse_args()
    result = analyze_csv(args.csv, args.min_vis)
    print_analysis(result)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved JSON report: {args.json}")
