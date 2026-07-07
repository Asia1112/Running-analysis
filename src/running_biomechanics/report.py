import argparse
import json
from datetime import datetime

from .analysis import analyze_csv
from .config import DEFAULT_CSV_PATH


STATUS_LABELS = {
    "usable": "Usable automatic assessment",
    "needs_review": "Assessment requires verification",
    "low_confidence": "Low-confidence recording",
}


def status_summary(status):
    if status == "usable":
        return (
            "The recording quality and biomechanical signals are sufficient for an "
            "automatic preliminary running-technique assessment."
        )
    if status == "needs_review":
        return (
            "The system detected usable pose data, but one or more metrics should be "
            "verified before making a strong conclusion."
        )

    return (
        "The recording or detected motion pattern is not reliable enough for a strong "
        "automatic technique assessment."
    )


def quality_label(result):
    if result["coverage"] >= 0.85 and result["core_visible_pct"] >= 85:
        return "good"
    if result["coverage"] >= 0.70 and result["core_visible_pct"] >= 75:
        return "moderate"
    return "limited"


def cadence_observation(result):
    cadence = result["cadence"]
    if cadence < 160:
        return "Cadence is relatively low for running, which may indicate longer steps or possible overstriding."
    if cadence > 200:
        return "Cadence is very high; the gait-event detection should be verified."
    return "Cadence is within a typical recreational running range."


def symmetry_observation(result):
    if result["lr_cadence_difference"] > 15 or result["knee_rom_difference_deg"] > 12:
        return (
            "Left-right symmetry requires attention because cadence or knee range of "
            "motion differs between sides."
        )
    return "Left-right rhythm and knee range of motion are relatively balanced."


def trunk_observation(result):
    trunk = result["trunk_lean_mean_deg"]
    if abs(trunk) > 15:
        return "Average trunk lean is elevated in the selected analysis window."
    return "Average trunk lean is not elevated in the selected analysis window."


def vertical_motion_observation(result):
    if result["vertical_oscillation_proxy"] > 0.035:
        return "Hip vertical oscillation proxy is elevated, which may indicate excessive up-down movement."
    return "Hip vertical oscillation proxy is within the expected range for this video-based estimate."


def build_report(result, source_name=None):
    observations = [
        cadence_observation(result),
        symmetry_observation(result),
        trunk_observation(result),
        vertical_motion_observation(result),
    ]

    if result["flags"]:
        observations.extend(result["flags"])

    return {
        "title": "Running Technique Report",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": source_name,
        "status": result["report_status"],
        "status_label": STATUS_LABELS.get(result["report_status"], result["report_status"]),
        "summary": status_summary(result["report_status"]),
        "recommendation": result["recommendation"],
        "quality": {
            "label": quality_label(result),
            "coverage_pct": round(result["coverage"] * 100, 1),
            "core_visible_pct": round(result["core_visible_pct"], 1),
            "confidence": result["confidence"],
        },
        "metrics": {
            "cadence_spm": round(result["cadence"], 1),
            "cadence_left_spm": round(result["cadence_left"], 1),
            "cadence_right_spm": round(result["cadence_right"], 1),
            "lr_cadence_difference_spm": round(result["lr_cadence_difference"], 1),
            "left_knee_rom_deg": round(result["left_knee_rom_deg"], 1),
            "right_knee_rom_deg": round(result["right_knee_rom_deg"], 1),
            "knee_rom_difference_deg": round(result["knee_rom_difference_deg"], 1),
            "trunk_lean_mean_deg": round(result["trunk_lean_mean_deg"], 1),
            "trunk_lean_std_deg": round(result["trunk_lean_std_deg"], 1),
            "vertical_oscillation_proxy": round(result["vertical_oscillation_proxy"], 4),
            "symmetry_score": round(result["symmetry_score"], 1),
        },
        "analysis_window": {
            "start_s": round(result["window_start_s"], 2),
            "end_s": round(result["window_end_s"], 2),
            "events": result["events"],
        },
        "observations": observations,
        "limitations": [
            "Metrics are estimated from 2D side-view video and should be treated as proxy values.",
            "Results depend on full-body visibility, stable camera position, and correct runner selection.",
            "Detected irregularities should be verified with better recordings or expert review.",
        ],
    }


def report_to_markdown(report):
    metrics = report["metrics"]
    quality = report["quality"]
    window = report["analysis_window"]

    lines = [
        f"# {report['title']}",
        "",
        f"Status: {report['status_label']}",
        f"Quality: {quality['label']} ({quality['coverage_pct']}% coverage, {quality['core_visible_pct']}% core visibility)",
        f"Analysis window: {window['start_s']}s - {window['end_s']}s, events={window['events']}",
        "",
        "## Summary",
        "",
        report["summary"],
        "",
        "## Key Metrics",
        "",
        f"- Cadence: {metrics['cadence_spm']} spm",
        f"- Left/right cadence: {metrics['cadence_left_spm']} / {metrics['cadence_right_spm']} spm",
        f"- Knee ROM left/right: {metrics['left_knee_rom_deg']} / {metrics['right_knee_rom_deg']} deg",
        f"- Knee ROM difference: {metrics['knee_rom_difference_deg']} deg",
        f"- Trunk lean: {metrics['trunk_lean_mean_deg']} deg",
        f"- Vertical oscillation proxy: {metrics['vertical_oscillation_proxy']}",
        f"- Symmetry score: {metrics['symmetry_score']}/100",
        "",
        "## Observations",
        "",
    ]

    lines.extend(f"- {observation}" for observation in report["observations"])
    lines.extend([
        "",
        "## Recommendation",
        "",
        report["recommendation"],
        "",
        "## Limitations",
        "",
    ])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])

    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a report from landmarks CSV.")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Input landmarks CSV path.")
    parser.add_argument("--source-name", help="Optional source video name for the report.")
    parser.add_argument("--json", help="Optional output JSON report path.")
    parser.add_argument("--markdown", help="Optional output Markdown report path.")
    return parser.parse_args()


def main():
    args = parse_args()
    result = analyze_csv(args.csv)
    report = build_report(result, source_name=args.source_name)
    markdown = report_to_markdown(report)
    print(markdown)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved JSON report: {args.json}")

    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"Saved Markdown report: {args.markdown}")
