import argparse
import json
from datetime import datetime

from .analysis import analyze_csv
from .config import DEFAULT_CSV_PATH


STATUS_LABELS = {
    "usable": "Wiarygodna ocena automatyczna",
    "needs_review": "Ocena wymaga weryfikacji",
    "low_confidence": "Nagranie o niskiej wiarygodności",
}


def status_summary(status):
    if status == "usable":
        return (
            "Jakość nagrania i sygnały biomechaniczne są wystarczające do "
            "automatycznej, wstępnej oceny techniki biegu."
        )
    if status == "needs_review":
        return (
            "System wykrył użyteczne dane pozy, ale co najmniej jeden wskaźnik "
            "należy zweryfikować przed sformułowaniem jednoznacznych wniosków."
        )

    return (
        "Nagranie lub wykryty wzorzec ruchu nie są wystarczająco wiarygodne, "
        "aby przeprowadzić jednoznaczną automatyczną ocenę techniki."
    )


def quality_label(result):
    if result["coverage"] >= 0.85 and result["core_visible_pct"] >= 85:
        return "dobra"
    if result["coverage"] >= 0.70 and result["core_visible_pct"] >= 75:
        return "umiarkowana"
    return "ograniczona"


def cadence_observation(result):
    cadence = result["cadence"]
    if cadence < 160:
        return "Kadencja jest stosunkowo niska, co może wskazywać na dłuższe kroki lub nadmierne wydłużanie kroku."
    if cadence > 200:
        return "Kadencja jest bardzo wysoka; należy zweryfikować wykrywanie zdarzeń chodu."
    return "Kadencja mieści się w typowym zakresie dla biegu rekreacyjnego."


def symmetry_observation(result):
    if result["lr_cadence_difference"] > 15 or result["knee_rom_difference_deg"] > 12:
        return (
            "Symetria lewej i prawej strony wymaga uwagi, ponieważ kadencja lub "
            "zakres ruchu kolan różnią się między stronami."
        )
    return "Rytm lewej i prawej strony oraz zakres ruchu kolan są względnie zrównoważone."


def trunk_observation(result):
    trunk = result["trunk_lean_mean_deg"]
    if abs(trunk) > 15:
        return "Średnie pochylenie tułowia jest podwyższone w wybranym oknie analizy."
    return "Średnie pochylenie tułowia nie jest podwyższone w wybranym oknie analizy."


def vertical_motion_observation(result):
    if result["vertical_oscillation_proxy"] > 0.035:
        return "Szacowana oscylacja pionowa bioder jest podwyższona, co może oznaczać nadmierny ruch góra–dół."
    return "Szacowana oscylacja pionowa bioder mieści się w oczekiwanym zakresie dla analizy wideo."


def report_recommendation(status):
    if status == "usable":
        return "Automatyczna ocena może zostać przedstawiona użytkownikowi."
    if status == "needs_review":
        return "Pokaż raport, ale oznacz wykryte nieprawidłowości jako wymagające weryfikacji."
    return "Nie wyciągaj jednoznacznych wniosków z tego nagrania; poproś o lepsze nagranie."


def build_report(result, source_name=None):
    observations = [
        cadence_observation(result),
        symmetry_observation(result),
        trunk_observation(result),
        vertical_motion_observation(result),
    ]

    return {
        "title": "Raport techniki biegu",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": source_name,
        "status": result["report_status"],
        "status_label": STATUS_LABELS.get(result["report_status"], result["report_status"]),
        "summary": status_summary(result["report_status"]),
        "recommendation": report_recommendation(result["report_status"]),
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
            "Wskaźniki są szacowane na podstawie dwuwymiarowego nagrania z boku i należy je traktować jako wartości przybliżone.",
            "Wyniki zależą od widoczności całego ciała i stabilnego ustawienia kamery.",
            "Wykryte nieprawidłowości należy zweryfikować na lepszym nagraniu lub skonsultować ze specjalistą.",
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
        f"Jakość: {quality['label']} (pokrycie {quality['coverage_pct']}%, widoczność głównych punktów {quality['core_visible_pct']}%)",
        f"Okno analizy: {window['start_s']} s – {window['end_s']} s, zdarzenia: {window['events']}",
        "",
        "## Podsumowanie",
        "",
        report["summary"],
        "",
        "## Najważniejsze wskaźniki",
        "",
        f"- Kadencja: {metrics['cadence_spm']} kroków/min",
        f"- Kadencja lewa/prawa: {metrics['cadence_left_spm']} / {metrics['cadence_right_spm']} kroków/min",
        f"- Zakres ruchu kolan lewego/prawego: {metrics['left_knee_rom_deg']}° / {metrics['right_knee_rom_deg']}°",
        f"- Różnica zakresu ruchu kolan: {metrics['knee_rom_difference_deg']}°",
        f"- Pochylenie tułowia: {metrics['trunk_lean_mean_deg']}°",
        f"- Szacowana oscylacja pionowa: {metrics['vertical_oscillation_proxy']}",
        f"- Wynik symetrii: {metrics['symmetry_score']}/100",
        "",
        "## Obserwacje",
        "",
    ]

    lines.extend(f"- {observation}" for observation in report["observations"])

    validation = report.get("rom_validation")
    if validation:
        lines.extend([
            "",
            "## Walidacja ROM kolan",
            "",
        ])
        for side_key in ("left", "right"):
            side = validation["sides"][side_key]
            raw_min = side["raw_extrema"]["min"]
            raw_max = side["raw_extrema"]["max"]
            lines.extend([
                f"### {side['label']} kolano",
                "",
                f"- ROM surowy: {side['raw_rom_deg']}°",
                f"- ROM po wygładzeniu: {side['smoothed_rom_deg']}°",
                f"- Spadek ROM: {side['rom_drop_deg']}° ({side['rom_drop_pct']}%)",
                f"- Skoki kąta > {validation['jump_threshold_deg']:g}°: {side['jump_count']}",
                f"- Klatka minimum: {raw_min['frame'] if raw_min else 'brak'}",
                f"- Klatka maksimum: {raw_max['frame'] if raw_max else 'brak'}",
                f"- Prawdopodobny błąd śledzenia: {'tak' if side['tracking_error'] else 'nie'}",
                "",
            ])

        frame_numbers = validation["suspicious_frame_numbers"]
        frames_text = ", ".join(str(frame) for frame in frame_numbers) if frame_numbers else "brak"
        lines.extend([
            f"Podejrzane klatki: {frames_text}",
            "",
            f"Wyeksportowane obrazy diagnostyczne: {validation.get('debug_frames_dir', 'brak')}",
        ])

    extended = report.get("extended_metrics")
    if extended:
        cadence = extended["cadence"]
        hips = extended["hips"]
        trunk = extended["trunk"]
        technique = extended["technique"]
        arms = extended["arms"]
        lines.extend([
            "",
            "## Dodatkowe wskaźniki 2D",
            "",
            f"- Regularność kroków: {cadence['regularity_label']} "
            f"(CV: {cadence['regularity_cv_pct'] if cadence['regularity_cv_pct'] is not None else 'brak danych'}%)",
            f"- ROM biodra L/P: {hips['left']['smoothed_rom_deg']}° / {hips['right']['smoothed_rom_deg']}°",
            f"- Różnica ROM bioder: {hips['rom_difference_deg']}°",
            f"- Zmienność pochylenia tułowia: {trunk['std_deg']}°",
            f"- Stabilność sylwetki: {trunk['stability_pct']}%",
            f"- Oscylacja pionowa bioder: {technique['vertical_hip_oscillation_pct_torso']}% długości tułowia",
            f"- Overstriding (proxy 2D): "
            f"{'możliwy' if technique['possible_overstriding'] else 'nie wykryto' if technique['overstriding_available'] else 'brak wiarygodnej oceny'}",
            f"- ROM ramion L/P: {arms['left_arm_rom_deg']}° / {arms['right_arm_rom_deg']}°",
            f"- Symetria pracy rąk: {arms['symmetry_score'] if arms['symmetry_score'] is not None else 'brak danych'}/100",
            "",
            "Wskaźniki kontaktu stopy i overstriding są estymacjami z obrazu 2D, nie pomiarem sił reakcji podłoża.",
        ])

    lines.extend([
        "",
        "## Zalecenie",
        "",
        report["recommendation"],
        "",
        "## Ograniczenia",
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
