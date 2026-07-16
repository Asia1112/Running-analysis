import numpy as np

from .analysis import (
    MIN_SEP_HARD,
    PROMINENCE_TH,
    SMOOTH_K,
    cadence_status,
    filter_minima_by_sep,
    find_local_minima_strong,
    get_xy,
    median_period,
    nanmean_pair,
    optional_round,
    read_rows,
    rom_value,
    safe_nanmean,
    safe_nanmax,
    safe_nanmin,
    safe_nanstd,
    trunk_lean_deg,
)
from .config import ANALYSIS_VISIBILITY_THRESHOLD, LANDMARKS_TO_SAVE, ROM_VISIBILITY_THRESHOLD
from .utils import moving_avg_nan, to_float


ANGLE_STABLE_DELTA_DEG = 10.0
MIN_METRIC_COVERAGE = 0.60


RELIABILITY_LABELS = {
    "good": "dobra",
    "review": "wymaga weryfikacji",
    "unreliable": "niewiarygodna",
}


def _reliability(level, reason):
    return {
        "level": level,
        "label": RELIABILITY_LABELS[level],
        "reason": reason,
    }


def _build_reliability(result, validation, details):
    cadence_value = result.get("cadence", np.nan)
    events = result.get("events", 0)
    coverage = result.get("coverage", 0)
    core_visible = result.get("core_visible_pct", 0)
    cadence_level = cadence_status(result)
    if cadence_level == "low_confidence":
        if np.isfinite(cadence_value) and (cadence_value < 140 or cadence_value > 220):
            reason = (
                f"Kadencja {cadence_value:.1f} kroków/min jest poza zakresem "
                "140–220; wykrywanie zdarzeń chodu jest prawdopodobnie niestabilne."
            )
        else:
            reason = (
                "Za mało stabilnych zdarzeń lub zbyt słabe pokrycie punktów "
                "do wiarygodnego wyznaczenia rytmu."
            )
        cadence = _reliability("unreliable", reason)
    elif cadence_level == "needs_review":
        reasons = []
        if cadence_value < 160 or cadence_value > 200:
            reasons.append("kadencja jest poza typowym zakresem 160–200")
        if result.get("lr_cadence_difference", 0) > 15:
            reasons.append("różnica L/P przekracza 15 kroków/min")
        if events < 8 or coverage < 0.80 or core_visible < 85:
            reasons.append("liczba zdarzeń lub jakość punktów ogranicza pewność")
        cadence = _reliability(
            "review",
            "; ".join(reasons).capitalize() + ".",
        )
    else:
        cadence = _reliability(
            "good",
            "Kadencja, różnica L/P, liczba zdarzeń i pokrycie punktów spełniają progi wiarygodności.",
        )

    if core_visible < 70:
        knee = _reliability("unreliable", "Widoczność głównych punktów kończyn dolnych jest zbyt niska.")
    elif validation.get("tracking_error"):
        knee = _reliability("review", "ROM wyraźnie zmienia się po wygładzeniu danych.")
    else:
        knee = _reliability("good", "ROM jest stabilny po wygładzeniu, a punkty są dostatecznie widoczne.")

    hip_valid = min(
        details["hips"]["left"].get("valid_pct") or 0,
        details["hips"]["right"].get("valid_pct") or 0,
    )
    if not details["hips"]["available"]:
        hip = _reliability("unreliable", "Za mało poprawnych klatek dla kąta biodra L/P.")
    elif hip_valid >= 85:
        hip = _reliability("good", "Kąty bioder są dostępne w większości klatek.")
    else:
        hip = _reliability("review", "Część klatek bioder została odrzucona przez próg visibility.")

    trunk_valid = details["trunk"].get("valid_pct") or 0
    if trunk_valid < 60:
        trunk = _reliability("unreliable", "Barki lub biodra są zbyt często niewidoczne.")
    elif trunk_valid < 85:
        trunk = _reliability("review", "Pochylenie obliczono z ograniczonej liczby poprawnych klatek.")
    else:
        trunk = _reliability("good", "Tułów jest widoczny w większości analizowanych klatek.")

    technique = details["technique"]
    if technique["overstriding_available"]:
        overstriding = _reliability(
            "review",
            "To wskaźnik pomocniczy z obrazu 2D; kontakt stopy nie jest mierzony platformą sił.",
        )
    else:
        overstriding = _reliability(
            "unreliable",
            "Nie udało się pewnie wyznaczyć kierunku ruchu lub kontaktów stopy.",
        )

    overall_level = {
        "usable": "good",
        "needs_review": "review",
        "low_confidence": "unreliable",
    }.get(result.get("report_status"), "review")
    overall = _reliability(
        overall_level,
        "Ocena łączy pokrycie punktów, liczbę zdarzeń i stabilność sygnałów kątowych.",
    )
    return {
        "cadence": cadence,
        "knee_rom": knee,
        "hip_rom": hip,
        "trunk": trunk,
        "overstriding": overstriding,
        "overall": overall,
    }


def _columns_available(rows, landmark_names):
    if not rows:
        return False
    keys = rows[0].keys()
    return all(
        f"{name}_{suffix}" in keys
        for name in landmark_names
        for suffix in ("x", "y", "vis")
    )


def _joint_angle(a, b, c):
    ax, ay, _ = a
    bx, by, _ = b
    cx, cy, _ = c
    bax = ax - bx
    bay = ay - by
    bcx = cx - bx
    bcy = cy - by
    dot = bax * bcx + bay * bcy
    norm = np.hypot(bax, bay) * np.hypot(bcx, bcy)
    cosine = np.divide(dot, norm, out=np.full_like(dot, np.nan), where=norm > 1e-9)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _angle_series(rows, names, visibility_threshold):
    if not _columns_available(rows, names):
        return np.full(len(rows), np.nan, dtype=float)
    points = [get_xy(rows, name, visibility_threshold) for name in names]
    return _joint_angle(*points)


def _valid_pct(values):
    if len(values) == 0:
        return 0.0
    return float(np.mean(np.isfinite(values)) * 100)


def _stability(values, threshold=ANGLE_STABLE_DELTA_DEG):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    delta = np.abs(np.diff(values))
    valid = np.isfinite(delta)
    if not np.any(valid):
        return None
    return optional_round(np.mean(delta[valid] <= threshold) * 100, 1)


def _side_summary(raw, smoothing_window=SMOOTH_K):
    smoothed = moving_avg_nan(raw, smoothing_window)
    return {
        "raw_rom_deg": optional_round(rom_value(raw), 2),
        "smoothed_rom_deg": optional_round(rom_value(smoothed), 2),
        "valid_pct": optional_round(_valid_pct(raw), 1),
        "stability_pct": _stability(raw),
    }, smoothed


def _chart_series(t, left, right, left_s, right_s):
    points = []
    for side, label, raw, smoothed in (
        ("left", "Lewa", left, left_s),
        ("right", "Prawa", right, right_s),
    ):
        for i in range(len(t)):
            points.append({
                "time_s": optional_round(t[i], 3),
                "side": side,
                "side_label": label,
                "angle_raw_deg": optional_round(raw[i], 3),
                "angle_smoothed_deg": optional_round(smoothed[i], 3),
            })
    return points


def _interval_cv(indices, t):
    if len(indices) < 3:
        return None
    periods = np.diff(t[indices])
    periods = periods[(periods >= MIN_SEP_HARD) & (periods <= 1.2)]
    if len(periods) < 2 or np.mean(periods) <= 0:
        return None
    return optional_round(np.std(periods) / np.mean(periods) * 100, 1)


def _cadence_regularity(t, left_knee, right_knee):
    indices = []
    cvs = []
    for values in (left_knee, right_knee):
        minima0 = find_local_minima_strong(moving_avg_nan(values, SMOOTH_K))
        period = median_period(minima0, t)
        if np.isfinite(period):
            minima = filter_minima_by_sep(minima0, t, max(MIN_SEP_HARD, 0.6 * period))
        else:
            minima = minima0
        indices.append(minima)
        cv = _interval_cv(minima, t)
        if cv is not None:
            cvs.append(cv)
    overall = optional_round(np.mean(cvs), 1) if cvs else None
    if overall is None:
        label = "niewystarczające dane"
    elif overall <= 5:
        label = "wysoka"
    elif overall <= 10:
        label = "umiarkowana"
    else:
        label = "niska"
    return {
        "left_stride_cv_pct": _interval_cv(indices[0], t),
        "right_stride_cv_pct": _interval_cv(indices[1], t),
        "regularity_cv_pct": overall,
        "regularity_label": label,
    }


def _visibility_quality(rows, visibility_threshold=ROM_VISIBILITY_THRESHOLD):
    labels = {
        "L_SHOULDER": "Bark L", "R_SHOULDER": "Bark P",
        "L_ELBOW": "Łokieć L", "R_ELBOW": "Łokieć P",
        "L_WRIST": "Nadgarstek L", "R_WRIST": "Nadgarstek P",
        "L_HIP": "Biodro L", "R_HIP": "Biodro P",
        "L_KNEE": "Kolano L", "R_KNEE": "Kolano P",
        "L_ANKLE": "Kostka L", "R_ANKLE": "Kostka P",
        "L_FOOT": "Stopa L", "R_FOOT": "Stopa P",
    }
    points = []
    for name in LANDMARKS_TO_SAVE:
        column = f"{name}_vis"
        if column not in rows[0]:
            continue
        values = np.array([to_float(row.get(column)) for row in rows], dtype=float)
        points.append({
            "punkt": labels.get(name, name),
            "mean_visibility": optional_round(safe_nanmean(values), 3),
            "visible_pct": optional_round(np.mean(values >= visibility_threshold) * 100, 1),
        })
    return points


def _local_maxima(values, t, min_sep=0.35):
    candidates = []
    for i in range(1, len(values) - 1):
        if np.isfinite(values[i]) and values[i] >= values[i - 1] and values[i] > values[i + 1]:
            candidates.append(i)
    keep = []
    for idx in candidates:
        if not keep or t[idx] - t[keep[-1]] >= min_sep:
            keep.append(idx)
        elif values[idx] > values[keep[-1]]:
            keep[-1] = idx
    return np.asarray(keep, dtype=int)


def _technique(rows, t, xy, hip_center_x, hip_center_y):
    contacts = {}
    shin_values = {}
    overstride_values = []
    overstride_by_side = {}

    valid_hip = np.isfinite(hip_center_x)
    direction = None
    displacement = None
    if np.sum(valid_hip) >= 10:
        x = hip_center_x[valid_hip]
        displacement = float(np.nanpercentile(x, 90) - np.nanpercentile(x, 10))
        slope = np.polyfit(t[valid_hip], x, 1)[0]
        if displacement >= 0.08 and abs(slope) >= 0.005:
            direction = 1.0 if slope > 0 else -1.0

    l_sh_x, l_sh_y, _ = xy["L_SHOULDER"]
    r_sh_x, r_sh_y, _ = xy["R_SHOULDER"]
    shoulder_y = nanmean_pair(l_sh_y, r_sh_y)
    torso_length = np.hypot(nanmean_pair(l_sh_x, r_sh_x) - hip_center_x, shoulder_y - hip_center_y)
    torso_scale = np.nanmedian(torso_length[np.isfinite(torso_length)]) if np.any(np.isfinite(torso_length)) else np.nan
    hip_s = moving_avg_nan(hip_center_y, SMOOTH_K)
    valid_hip_y = np.isfinite(hip_s)
    if np.sum(valid_hip_y) >= 10:
        time_step = np.nanmedian(np.diff(t))
        baseline_window = max(5, int(round(0.6 / time_step))) if time_step > 0 else 15
        if baseline_window % 2 == 0:
            baseline_window += 1
        hip_residual = hip_s - moving_avg_nan(hip_s, baseline_window)
        hip_range = float(
            np.nanpercentile(hip_residual, 95) - np.nanpercentile(hip_residual, 5)
        )
    else:
        hip_range = np.nan
    oscillation_pct = hip_range / torso_scale * 100 if np.isfinite(torso_scale) and torso_scale > 0 else np.nan

    for side in ("L", "R"):
        ankle_x, ankle_y, _ = xy[f"{side}_ANKLE"]
        knee_x, knee_y, _ = xy[f"{side}_KNEE"]
        foot_x, foot_y, _ = xy[f"{side}_FOOT"]
        event_source = moving_avg_nan(foot_y, 3)
        event_idx = _local_maxima(event_source, t)
        contacts[side] = event_idx
        shin = np.degrees(np.arctan2(ankle_x - knee_x, ankle_y - knee_y))
        shin_values[side] = shin[event_idx]

        if direction is not None:
            leg_length = np.hypot(knee_x - ankle_x, knee_y - ankle_y)
            offset = direction * (foot_x - hip_center_x)
            normalized = np.divide(
                offset,
                leg_length,
                out=np.full_like(offset, np.nan),
                where=leg_length > 1e-9,
            )
            valid_offsets = normalized[event_idx][np.isfinite(normalized[event_idx])]
            overstride_values.extend(valid_offsets)
            overstride_by_side[side] = (
                float(np.nanmedian(valid_offsets)) if len(valid_offsets) else np.nan
            )

    contact_count = int(len(contacts["L"]) + len(contacts["R"]))
    shin_left = safe_nanmean(shin_values["L"])
    shin_right = safe_nanmean(shin_values["R"])
    contact_available = contact_count >= 4 and (
        np.isfinite(shin_left) or np.isfinite(shin_right)
    )
    overstride_median = float(np.nanmedian(overstride_values)) if overstride_values else np.nan
    overstride_available = contact_available and direction is not None and np.isfinite(overstride_median)
    return {
        "contact_available": bool(contact_available),
        "contact_count": contact_count,
        "direction_available": direction is not None,
        "shin_angle_left_deg": optional_round(shin_left, 1) if contact_available else None,
        "shin_angle_right_deg": optional_round(shin_right, 1) if contact_available else None,
        "overstriding_available": bool(overstride_available),
        "overstriding_offset_leg_ratio": optional_round(overstride_median, 3) if overstride_available else None,
        "overstriding_left_ratio": optional_round(overstride_by_side.get("L"), 3) if overstride_available else None,
        "overstriding_right_ratio": optional_round(overstride_by_side.get("R"), 3) if overstride_available else None,
        "possible_overstriding": bool(overstride_available and overstride_median > 0.15),
        "vertical_hip_oscillation_pct_torso": optional_round(oscillation_pct, 1),
        "note": (
            "Kontakt stopy i overstriding są estymacjami z obrazu 2D, a nie pomiarem sił reakcji podłoża."
        ),
    }


def analyze_extended_metrics(csv_path, result, validation, visibility_threshold=ANALYSIS_VISIBILITY_THRESHOLD):
    """Build quality-gated 2D metrics for the detailed Streamlit dashboard."""
    rows = read_rows(csv_path)
    t = np.array([to_float(row.get("t_ms")) / 1000.0 for row in rows], dtype=float)
    frame = np.array([int(to_float(row.get("frame_idx", i))) for i, row in enumerate(rows)])
    xy = {
        name: get_xy(rows, name, visibility_threshold)
        for name in LANDMARKS_TO_SAVE
        if _columns_available(rows, [name])
    }

    left_knee = np.array([to_float(row.get("left_knee_deg")) for row in rows])
    right_knee = np.array([to_float(row.get("right_knee_deg")) for row in rows])
    cadence_details = _cadence_regularity(t, left_knee, right_knee)

    left_hip = _angle_series(rows, ("L_SHOULDER", "L_HIP", "L_KNEE"), visibility_threshold)
    right_hip = _angle_series(rows, ("R_SHOULDER", "R_HIP", "R_KNEE"), visibility_threshold)
    left_hip_summary, left_hip_s = _side_summary(left_hip)
    right_hip_summary, right_hip_s = _side_summary(right_hip)
    hip_available = min(_valid_pct(left_hip), _valid_pct(right_hip)) >= MIN_METRIC_COVERAGE * 100

    arms_have_columns = _columns_available(
        rows, ("L_SHOULDER", "R_SHOULDER", "L_ELBOW", "R_ELBOW", "L_WRIST", "R_WRIST")
    )
    left_elbow = _angle_series(rows, ("L_SHOULDER", "L_ELBOW", "L_WRIST"), visibility_threshold)
    right_elbow = _angle_series(rows, ("R_SHOULDER", "R_ELBOW", "R_WRIST"), visibility_threshold)
    left_elbow_summary, left_elbow_s = _side_summary(left_elbow)
    right_elbow_summary, right_elbow_s = _side_summary(right_elbow)
    left_swing = _angle_series(rows, ("L_ELBOW", "L_SHOULDER", "L_HIP"), visibility_threshold)
    right_swing = _angle_series(rows, ("R_ELBOW", "R_SHOULDER", "R_HIP"), visibility_threshold)
    left_swing_rom = rom_value(moving_avg_nan(left_swing, SMOOTH_K))
    right_swing_rom = rom_value(moving_avg_nan(right_swing, SMOOTH_K))
    arm_available = arms_have_columns and min(_valid_pct(left_elbow), _valid_pct(right_elbow)) >= MIN_METRIC_COVERAGE * 100

    required_technique = ("L_SHOULDER", "R_SHOULDER", "L_HIP", "R_HIP", "L_KNEE", "R_KNEE", "L_ANKLE", "R_ANKLE", "L_FOOT", "R_FOOT")
    if all(name in xy for name in required_technique):
        hip_x = nanmean_pair(xy["L_HIP"][0], xy["R_HIP"][0])
        hip_y = nanmean_pair(xy["L_HIP"][1], xy["R_HIP"][1])
        technique = _technique(rows, t, xy, hip_x, hip_y)
        trunk = trunk_lean_deg(xy)
    else:
        technique = {
            "contact_available": False, "contact_count": 0, "direction_available": False,
            "shin_angle_left_deg": None, "shin_angle_right_deg": None,
            "overstriding_available": False, "overstriding_offset_leg_ratio": None,
            "overstriding_left_ratio": None, "overstriding_right_ratio": None,
            "possible_overstriding": False, "vertical_hip_oscillation_pct_torso": None,
            "note": "Brak wystarczających punktów do oceny techniki.",
        }
        trunk = np.full(len(rows), np.nan)

    knee_stabilities = [_stability(left_knee), _stability(right_knee)]
    knee_stabilities = [value for value in knee_stabilities if value is not None]
    unique_suspicious = len(validation.get("suspicious_frame_numbers", []))
    hip_rom_diff = (
        abs(left_hip_summary["smoothed_rom_deg"] - right_hip_summary["smoothed_rom_deg"])
        if left_hip_summary["smoothed_rom_deg"] is not None and right_hip_summary["smoothed_rom_deg"] is not None
        else None
    )
    arm_rom_diff = (
        abs(left_swing_rom - right_swing_rom)
        if np.isfinite(left_swing_rom) and np.isfinite(right_swing_rom) else np.nan
    )

    landmark_quality = _visibility_quality(rows)
    visible_values = [
        item["visible_pct"] for item in landmark_quality if item["visible_pct"] is not None
    ]
    worst_landmarks = sorted(
        landmark_quality,
        key=lambda item: item["visible_pct"] if item["visible_pct"] is not None else -1,
    )[:3]
    suspicious_ratio = unique_suspicious / max(1, len(rows)) * 100
    mean_visible_pct = float(np.mean(visible_values)) if visible_values else 0.0
    if result.get("core_visible_pct", 0) >= 85 and mean_visible_pct >= 85 and suspicious_ratio <= 5:
        quality_level = "good"
    elif result.get("core_visible_pct", 0) >= 70 and mean_visible_pct >= 70:
        quality_level = "review"
    else:
        quality_level = "unreliable"

    hip_difference = optional_round(hip_rom_diff, 2)
    if hip_difference is None:
        hip_symmetry = "brak wiarygodnej oceny"
    elif hip_difference <= 8:
        hip_symmetry = "dobra"
    elif hip_difference <= 15:
        hip_symmetry = "wymaga weryfikacji"
    else:
        hip_symmetry = "wyraźna różnica - wymaga weryfikacji"

    details = {
        "core_visible_pct": optional_round(result.get("core_visible_pct"), 1),
        "frame_count": len(rows),
        "duration_s": optional_round(t[-1] - t[0], 2) if len(t) else None,
        "quality": {
            "landmarks": landmark_quality,
            "worst_landmarks": worst_landmarks,
            "mean_visible_pct": optional_round(mean_visible_pct, 1),
            "suspicious_frame_count": unique_suspicious,
            "suspicious_frame_pct": optional_round(suspicious_ratio, 1),
            "angle_stability_pct": optional_round(np.mean(knee_stabilities), 1) if knee_stabilities else None,
            "angle_stability_left_pct": _stability(left_knee),
            "angle_stability_right_pct": _stability(right_knee),
            "level": quality_level,
            "label": RELIABILITY_LABELS[quality_level],
        },
        "cadence": cadence_details,
        "hips": {
            "available": bool(hip_available),
            "left": left_hip_summary,
            "right": right_hip_summary,
            "rom_difference_deg": hip_difference,
            "symmetry_label": hip_symmetry,
            "series": _chart_series(t, left_hip, right_hip, left_hip_s, right_hip_s),
        },
        "trunk": {
            "mean_deg": optional_round(safe_nanmean(trunk), 2),
            "min_deg": optional_round(safe_nanmin(trunk), 2),
            "max_deg": optional_round(safe_nanmax(trunk), 2),
            "std_deg": optional_round(safe_nanstd(trunk), 2),
            "stability_pct": _stability(trunk, threshold=3.0),
            "valid_pct": optional_round(_valid_pct(trunk), 1),
        },
        "technique": technique,
        "arms": {
            "available": bool(arm_available),
            "left_elbow": left_elbow_summary,
            "right_elbow": right_elbow_summary,
            "left_arm_rom_deg": optional_round(left_swing_rom, 2),
            "right_arm_rom_deg": optional_round(right_swing_rom, 2),
            "rom_difference_deg": optional_round(arm_rom_diff, 2),
            "symmetry_score": optional_round(max(0.0, 100.0 - arm_rom_diff * 2), 1) if np.isfinite(arm_rom_diff) else None,
            "series": _chart_series(t, left_elbow, right_elbow, left_elbow_s, right_elbow_s),
            "note": "ROM ramion oznacza zakres kąta ramię–bark–biodro w płaszczyźnie obrazu.",
        },
        "frames": frame.tolist(),
    }
    details["reliability"] = _build_reliability(result, validation, details)
    return details
