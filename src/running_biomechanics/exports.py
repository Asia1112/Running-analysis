import csv
import html
import io
import zipfile
from pathlib import Path


def rows_to_csv_bytes(rows, fieldnames=None):
    """Serialize dictionaries to UTF-8 CSV bytes suitable for Streamlit downloads."""
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def build_angles_csv(report):
    validation = report["rom_validation"]
    hips = report["extended_metrics"]["hips"]
    hip_lookup = {
        (point["time_s"], point["side"]): point
        for point in hips.get("series", [])
    }
    rows = []
    for knee in validation.get("series", []):
        hip = hip_lookup.get((knee["time_s"], knee["side"]), {})
        rows.append({
            "frame": knee["frame"],
            "time_s": knee["time_s"],
            "side": knee["side_label"],
            "knee_angle_raw_deg": knee["angle_raw_deg"],
            "knee_angle_smoothed_deg": knee["angle_smoothed_deg"],
            "knee_angle_change_deg": knee["angle_change_deg"],
            "hip_angle_raw_deg": hip.get("angle_raw_deg"),
            "hip_angle_smoothed_deg": hip.get("angle_smoothed_deg"),
        })
    return rows_to_csv_bytes(rows)


def suspicious_rows(report):
    rows = []
    for item in report["rom_validation"].get("suspicious_frames", []):
        rows.append({
            "frame": item.get("frame"),
            "time_s": item.get("time_s"),
            "side": item.get("side_label"),
            "problem": item.get("problem"),
            "problem_visibility": item.get("problem_visibility"),
            "hip_visibility": item.get("hip_visibility"),
            "knee_visibility": item.get("knee_visibility"),
            "ankle_visibility": item.get("ankle_visibility"),
            "angle_deg": item.get("angle_raw_deg"),
            "angle_change_deg": item.get("angle_change_deg"),
            "reason": "; ".join(item.get("reasons", [])),
        })
    return rows


def build_suspicious_csv(report):
    return rows_to_csv_bytes(suspicious_rows(report))


def build_debug_frames_zip(debug_frames_dir):
    if not debug_frames_dir:
        return None
    directory = Path(debug_frames_dir)
    if not directory.exists():
        return None
    images = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        return None
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for image in images:
            archive.write(image, arcname=image.name)
    return output.getvalue()


def _register_pdf_fonts():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("ReportRegular", str(regular)))
            pdfmetrics.registerFont(TTFont("ReportBold", str(bold)))
            return "ReportRegular", "ReportBold"
    return "Helvetica", "Helvetica-Bold"


def build_pdf_report(report):
    """Create a compact Polish engineering report as PDF bytes."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            KeepTogether,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError("Eksport PDF wymaga biblioteki reportlab.") from exc

    regular_font, bold_font = _register_pdf_fonts()
    output = io.BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Raport techniki biegu",
        author="Running Biomechanics Analysis",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=19,
        leading=23,
        textColor=colors.HexColor("#132238"),
        alignment=TA_CENTER,
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="ReportHeading",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#E85D04"),
        spaceBefore=9,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="ReportBody",
        parent=styles["BodyText"],
        fontName=regular_font,
        fontSize=8.7,
        leading=12,
        textColor=colors.HexColor("#263238"),
    ))
    styles.add(ParagraphStyle(
        name="ReportSmall",
        parent=styles["BodyText"],
        fontName=regular_font,
        fontSize=7.2,
        leading=9,
        textColor=colors.HexColor("#455A64"),
    ))
    styles.add(ParagraphStyle(
        name="ReportTableHeader",
        parent=styles["BodyText"],
        fontName=bold_font,
        fontSize=7.2,
        leading=9,
        textColor=colors.white,
    ))

    def p(value, style="ReportBody"):
        return Paragraph(html.escape(str(value)), styles[style])

    def table(data, widths=None, header=True):
        if header and data:
            data[0] = [
                Paragraph(
                    html.escape(cell.getPlainText() if hasattr(cell, "getPlainText") else str(cell)),
                    styles["ReportTableHeader"],
                )
                for cell in data[0]
            ]
        item = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
        commands = [
            ("FONTNAME", (0, 0), (-1, -1), regular_font),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CFD8DC")),
            ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#F5F7F9")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        if header:
            commands.extend([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#132238")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), bold_font),
            ])
        item.setStyle(TableStyle(commands))
        return item

    metrics = report["metrics"]
    quality = report["quality"]
    extended = report["extended_metrics"]
    validation = report["rom_validation"]
    reliability = extended["reliability"]
    story = [
        p("Raport techniki biegu", "ReportTitle"),
        p(
            f"Źródło: {report.get('source') or 'brak'} | Status: {report['status_label']} | "
            f"Utworzono: {report['created_at']}",
            "ReportSmall",
        ),
        Spacer(1, 6),
        p("Podsumowanie", "ReportHeading"),
        p(report["summary"]),
        Spacer(1, 6),
        table([
            [p("Wskaźnik", "ReportSmall"), p("Wynik", "ReportSmall")],
            [p("Klatki", "ReportSmall"), p(extended["frame_count"], "ReportSmall")],
            [p("Okno analizy", "ReportSmall"), p(f"{report['analysis_window']['start_s']} - {report['analysis_window']['end_s']} s", "ReportSmall")],
            [p("Widoczność głównych punktów", "ReportSmall"), p(f"{quality['core_visible_pct']}%", "ReportSmall")],
            [p("Kadencja", "ReportSmall"), p(f"{metrics['cadence_spm']} kroków/min", "ReportSmall")],
            [p("ROM kolan L/P", "ReportSmall"), p(f"{metrics['left_knee_rom_deg']}° / {metrics['right_knee_rom_deg']}°", "ReportSmall")],
            [p("ROM bioder L/P", "ReportSmall"), p(f"{extended['hips']['left']['smoothed_rom_deg']}° / {extended['hips']['right']['smoothed_rom_deg']}°", "ReportSmall")],
            [p("Pochylenie tułowia", "ReportSmall"), p(f"{extended['trunk']['mean_deg']}° ± {extended['trunk']['std_deg']}°", "ReportSmall")],
            [p("Oscylacja pionowa bioder", "ReportSmall"), p(f"{extended['technique']['vertical_hip_oscillation_pct_torso']}% długości tułowia", "ReportSmall")],
        ], widths=[62 * mm, 100 * mm]),
        p("Wiarygodność wyników", "ReportHeading"),
        table([
            [p("Metryka", "ReportSmall"), p("Ocena", "ReportSmall"), p("Uzasadnienie", "ReportSmall")],
            *[
                [p(label, "ReportSmall"), p(reliability[key]["label"], "ReportSmall"), p(reliability[key]["reason"], "ReportSmall")]
                for key, label in (
                    ("cadence", "Kadencja"),
                    ("knee_rom", "ROM kolan"),
                    ("hip_rom", "ROM bioder"),
                    ("trunk", "Pochylenie tułowia"),
                    ("overstriding", "Overstriding proxy"),
                    ("overall", "Cały raport"),
                )
            ],
        ], widths=[36 * mm, 38 * mm, 88 * mm]),
        p("Walidacja ROM kolan", "ReportHeading"),
        table([
            [p("Strona", "ReportSmall"), p("ROM surowy", "ReportSmall"), p("ROM wygładzony", "ReportSmall"), p("Spadek", "ReportSmall"), p("Skoki >30°", "ReportSmall")],
            *[
                [
                    p(validation["sides"][side]["label"], "ReportSmall"),
                    p(f"{validation['sides'][side]['raw_rom_deg']}°", "ReportSmall"),
                    p(f"{validation['sides'][side]['smoothed_rom_deg']}°", "ReportSmall"),
                    p(f"{validation['sides'][side]['rom_drop_deg']}°", "ReportSmall"),
                    p(validation["sides"][side]["jump_count"], "ReportSmall"),
                ]
                for side in ("left", "right")
            ],
        ], widths=[30 * mm, 32 * mm, 38 * mm, 30 * mm, 32 * mm]),
    ]

    suspicious = suspicious_rows(report)
    story.extend([p("Podejrzane klatki", "ReportHeading")])
    if suspicious:
        story.append(table([
            [p("Klatka", "ReportSmall"), p("Strona", "ReportSmall"), p("Problem", "ReportSmall"), p("Kąt", "ReportSmall"), p("Powód", "ReportSmall")],
            *[
                [
                    p(row["frame"], "ReportSmall"),
                    p(row["side"], "ReportSmall"),
                    p(row["problem"], "ReportSmall"),
                    p(row["angle_deg"], "ReportSmall"),
                    p(row["reason"], "ReportSmall"),
                ]
                for row in suspicious[:30]
            ],
        ], widths=[20 * mm, 25 * mm, 30 * mm, 24 * mm, 63 * mm]))
        if len(suspicious) > 30:
            story.append(p(f"Tabela pokazuje 30 z {len(suspicious)} wpisów. Pełna lista jest dostępna w eksporcie CSV.", "ReportSmall"))
    else:
        story.append(p("Nie wykryto podejrzanych klatek."))

    story.extend([
        KeepTogether([
            p("Ograniczenia interpretacji", "ReportHeading"),
            p(
                "Wyniki są wskaźnikami technicznymi i estymacjami z obrazu 2D. "
                "Nie stanowią diagnozy medycznej. Kontakt stopy, overstriding i kąty stawów "
                "wymagają weryfikacji na nagraniu, szczególnie przy słabej visibility punktów.",
            ),
        ])
    ])

    def draw_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(regular_font, 7)
        canvas.setFillColor(colors.HexColor("#607D8B"))
        canvas.drawString(16 * mm, 9 * mm, "Raport orientacyjny - analiza 2D MediaPipe")
        canvas.drawRightString(194 * mm, 9 * mm, f"Strona {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return output.getvalue()
