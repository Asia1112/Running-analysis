from pathlib import Path

import altair as alt
import streamlit as st

from src.running_biomechanics.analysis import analyze_csv, validate_knee_rom
from src.running_biomechanics.pose_extraction import process_video
from src.running_biomechanics.report import build_report, report_to_markdown


APP_RUNS_DIR = Path("app_runs")
APP_RUNS_DIR.mkdir(exist_ok=True)


STATUS_META = {
    "usable": ("Dobrze", "#65d44d"),
    "needs_review": ("Weryfikacja", "#d5b82f"),
    "low_confidence": ("Niska pewność", "#e45835"),
}


CROP_LABELS = {
    "none": "Bez kadrowania",
    "left": "Lewa połowa",
    "right": "Prawa połowa",
    "center": "Środek",
}


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --bg: #0d0f10;
            --panel: #141719;
            --panel-soft: #1b1f22;
            --text: #f3f3ef;
            --muted: #888b8c;
            --accent: #ff7a2f;
            --green: #65d44d;
            --yellow: #d5b82f;
            --red: #e45835;
        }

        .stApp {
            background:
                radial-gradient(circle at 20% 0%, rgba(255, 122, 47, 0.08), transparent 28rem),
                var(--bg);
            color: var(--text);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: #101214;
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.6rem;
        }

        h1, h2, h3, p, label, span, div {
            letter-spacing: 0;
        }

        h1 {
            font-size: 2.3rem;
            font-weight: 760;
            margin-bottom: 0.2rem;
        }

        h2 {
            font-size: 1.1rem;
            color: var(--accent);
            border-bottom: 1px solid rgba(255, 122, 47, 0.65);
            padding-bottom: 0.45rem;
            margin-top: 0.2rem;
            margin-bottom: 1.1rem;
        }

        .hero-subtitle {
            color: var(--muted);
            margin-bottom: 1.4rem;
        }

        .status-strip {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            background: linear-gradient(135deg, #181b1e, #101214);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }

        .status-title {
            font-size: 1.15rem;
            font-weight: 720;
        }

        .status-copy {
            color: var(--muted);
            margin-top: 0.2rem;
            line-height: 1.45;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 4.8rem;
            padding: 0.35rem 0.72rem;
            border-radius: 999px;
            color: #111;
            font-size: 0.86rem;
            font-weight: 800;
        }

        .metric-card {
            position: relative;
            min-height: 13.4rem;
            background: linear-gradient(180deg, #151819, #090b0c);
            border: 1px solid rgba(255, 255, 255, 0.055);
            border-radius: 8px;
            padding: 1.05rem;
            box-shadow: 0 14px 30px rgba(0, 0, 0, 0.28);
            overflow: hidden;
        }

        .metric-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
            margin-bottom: 1.6rem;
        }

        .gauge-wrap {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .gauge {
            width: 3.8rem;
            height: 3.8rem;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background: conic-gradient(var(--metric-color) calc(var(--score) * 1%), rgba(255, 255, 255, 0.08) 0);
        }

        .gauge-inner {
            width: 2.75rem;
            height: 2.75rem;
            display: grid;
            place-items: center;
            background: #171a1c;
            border-radius: 50%;
            color: var(--metric-color);
            font-size: 1.1rem;
            font-weight: 800;
        }

        .score {
            font-size: 1.35rem;
            font-weight: 760;
        }

        .metric-label {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.35;
        }

        .metric-value {
            color: var(--text);
            font-size: 1.38rem;
            line-height: 1.15;
            font-weight: 760;
            margin-top: 0.35rem;
        }

        .metric-foot {
            position: absolute;
            left: 0;
            right: 0;
            bottom: 0;
            padding: 0.58rem 1rem;
            background: #1b1f22;
            color: #b6b9b8;
            font-size: 0.82rem;
        }

        .observation-card {
            background: #141719;
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.65rem;
            color: #d6d8d5;
        }

        .stButton > button {
            background: var(--accent);
            color: #111;
            border: 0;
            border-radius: 8px;
            font-weight: 800;
            min-height: 2.75rem;
        }

        .stDownloadButton > button {
            border-radius: 8px;
            border-color: rgba(255, 122, 47, 0.65);
            color: var(--text);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_color(score):
    if score >= 75:
        return "#65d44d"
    if score >= 50:
        return "#d5b82f"
    return "#e45835"


def card(title, value, score, badge, footer):
    color = metric_color(score)
    return f"""
    <div class="metric-card" style="--metric-color: {color}; --score: {score};">
        <div class="metric-top">
            <div class="gauge-wrap">
                <div class="gauge"><div class="gauge-inner">{int(round(score))}</div></div>
                <div class="score">{int(round(score))}%</div>
            </div>
            <div class="pill" style="background: {color};">{badge}</div>
        </div>
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-foot">{footer}</div>
    </div>
    """


def bounded_score(value, ideal, tolerance):
    distance = abs(value - ideal)
    return max(0, min(100, 100 - (distance / tolerance) * 100))


def build_cards(report):
    metrics = report["metrics"]
    quality = report["quality"]

    cadence_score = bounded_score(metrics["cadence_spm"], 175, 45)
    symmetry_score = metrics["symmetry_score"]
    trunk_score = max(0, min(100, 100 - abs(metrics["trunk_lean_mean_deg"]) * 4))
    quality_score = min(100, (quality["coverage_pct"] + quality["core_visible_pct"]) / 2)

    return [
        card(
            "Kadencja",
            f"{metrics['cadence_spm']} kroków/min",
            cadence_score,
            "Dobrze" if cadence_score >= 75 else "Uwaga" if cadence_score >= 50 else "Słabo",
            "Szacowane tempo kroków",
        ),
        card(
            "Symetria lewa–prawa",
            f"{metrics['symmetry_score']}/100",
            symmetry_score,
            "Dobrze" if symmetry_score >= 75 else "Uwaga" if symmetry_score >= 50 else "Słabo",
            "Równowaga kadencji i zakresu ruchu kolan",
        ),
        card(
            "Pozycja tułowia",
            f"{metrics['trunk_lean_mean_deg']}°",
            trunk_score,
            "Dobrze" if trunk_score >= 75 else "Uwaga" if trunk_score >= 50 else "Słabo",
            "Średnie pochylenie w widoku z boku",
        ),
        card(
            "Jakość nagrania",
            quality["label"].capitalize(),
            quality_score,
            "Dobrze" if quality_score >= 85 else "Uwaga" if quality_score >= 70 else "Słabo",
            "Pokrycie i widoczność pozy",
        ),
    ]


def save_uploaded_video(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    path = APP_RUNS_DIR / f"uploaded{suffix}"
    path.write_bytes(uploaded_file.getbuffer())
    return path


def run_analysis(video_path, crop_mode):
    csv_path = APP_RUNS_DIR / "run_landmarks.csv"
    annotated_video_path = APP_RUNS_DIR / "annotated_run.mp4"
    processing_stats = process_video(
        source=str(video_path),
        csv_path=str(csv_path),
        crop_mode=crop_mode,
        resize_height=720,
        preview=False,
        debug_frames_dir="debug_frames",
        annotated_video_path=str(annotated_video_path),
    )
    result = analyze_csv(str(csv_path))
    report = build_report(result, source_name=video_path.name)
    validation = validate_knee_rom(str(csv_path))
    validation["debug_frames_dir"] = processing_stats["debug_frames_dir"]
    validation["debug_frames_count"] = processing_stats["debug_frames_count"]
    if validation["tracking_error"]:
        if report["status"] == "usable":
            report["status"] = "needs_review"
        report["status_label"] = "Prawdopodobny błąd śledzenia"
        report["summary"] += (
            " Walidacja ROM wykazała duży spadek zakresu ruchu po wygładzeniu danych."
        )
        report["recommendation"] = (
            "Zweryfikuj oznaczone klatki przed interpretacją zakresu ruchu kolan."
        )
    report["rom_validation"] = validation
    report["annotated_video_path"] = processing_stats["annotated_video_path"]
    return report


def extrema_caption(extrema):
    minimum = extrema.get("min")
    maximum = extrema.get("max")
    if minimum is None or maximum is None:
        return "Brak wystarczających danych o ekstremach."
    return (
        f"Minimum: klatka {minimum['frame']} ({minimum['angle_deg']}°) · "
        f"maksimum: klatka {maximum['frame']} ({maximum['angle_deg']}°)"
    )


def render_rom_validation(validation):
    st.markdown("### Walidacja ROM kolan")

    if validation["tracking_error"]:
        st.error(
            "ROM wyraźnie zmniejszył się po wygładzeniu. "
            "Wynik oznaczono jako prawdopodobny błąd śledzenia."
        )
    else:
        st.success("Wygładzenie nie spowodowało podejrzanie dużego spadku ROM.")

    chart_data = validation["series"]
    color = alt.Color(
        "side_label:N",
        title="Kolano",
        scale=alt.Scale(domain=["Lewe", "Prawe"], range=["#ff7a2f", "#65d44d"]),
    )
    base = alt.Chart(alt.Data(values=chart_data)).encode(
        x=alt.X("time_s:Q", title="Czas [s]"),
        color=color,
    )
    raw_lines = base.mark_line(strokeWidth=1.5, opacity=0.55).encode(
        y=alt.Y("angle_raw_deg:Q", title="Kąt kolana [°]"),
        tooltip=[
            alt.Tooltip("frame:Q", title="Klatka", format=".0f"),
            alt.Tooltip("time_s:Q", title="Czas [s]", format=".3f"),
            alt.Tooltip("side_label:N", title="Strona"),
            alt.Tooltip("angle_raw_deg:Q", title="Kąt surowy [°]", format=".1f"),
        ],
    )
    smoothed_lines = base.mark_line(strokeWidth=2.5, strokeDash=[7, 4]).encode(
        y=alt.Y("angle_smoothed_deg:Q", title="Kąt kolana [°]"),
    )
    jump_data = [point for point in chart_data if point["is_jump"]]
    jump_points = (
        alt.Chart(alt.Data(values=jump_data))
        .mark_point(filled=True, size=100, color="#e45835", stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("time_s:Q", title="Czas [s]"),
            y=alt.Y("angle_raw_deg:Q", title="Kąt kolana [°]"),
            tooltip=[
                alt.Tooltip("frame:Q", title="Klatka", format=".0f"),
                alt.Tooltip("side_label:N", title="Strona"),
                alt.Tooltip("angle_raw_deg:Q", title="Kąt [°]", format=".1f"),
                alt.Tooltip("angle_change_deg:Q", title="Zmiana [°]", format=".1f"),
                alt.Tooltip("knee_visibility:Q", title="Visibility kolana", format=".2f"),
                alt.Tooltip("ankle_visibility:Q", title="Visibility kostki", format=".2f"),
            ],
        )
    )
    st.altair_chart((raw_lines + smoothed_lines + jump_points).interactive())
    st.caption(
        "Linia ciągła: dane surowe · linia przerywana: dane wygładzone · "
        "czerwone punkty: zmiana kąta większa niż 30°."
    )

    side_columns = st.columns(2)
    for column, side_key in zip(side_columns, ("left", "right")):
        side = validation["sides"][side_key]
        with column:
            with st.container(border=True):
                st.markdown(f"**{side['label']} kolano**")
                metrics = st.columns(3)
                metrics[0].metric("ROM surowy", f"{side['raw_rom_deg']}°")
                metrics[1].metric("ROM wygładzony", f"{side['smoothed_rom_deg']}°")
                metrics[2].metric("Spadek ROM", f"{side['rom_drop_deg']}°")
                st.caption(extrema_caption(side["raw_extrema"]))
                st.caption(
                    f"Po wygładzeniu — {extrema_caption(side['smoothed_extrema']).lower()}"
                )
                st.caption(
                    f"Skoki >30°: {side['jump_count']} · "
                    f"niska visibility: {side['low_visibility_count']}"
                )
                if side["tracking_error"]:
                    st.warning("Prawdopodobny błąd śledzenia dla tej strony.")

    suspicious = validation["suspicious_frames"]
    st.markdown("#### Podejrzane klatki")
    if suspicious:
        table_rows = [
            {
                "Klatka": item["frame"],
                "Czas [s]": item["time_s"],
                "Strona": item["side_label"],
                "Kąt [°]": item["angle_raw_deg"],
                "Zmiana [°]": item["angle_change_deg"],
                "Visibility kolana": item["knee_visibility"],
                "Visibility kostki": item["ankle_visibility"],
                "Powód": "; ".join(item["reasons"]),
            }
            for item in suspicious
        ]
        st.dataframe(
            table_rows,
            column_config={
                "Klatka": st.column_config.NumberColumn(format="%d"),
                "Czas [s]": st.column_config.NumberColumn(format="%.3f"),
                "Kąt [°]": st.column_config.NumberColumn(format="%.1f"),
                "Zmiana [°]": st.column_config.NumberColumn(format="%.1f"),
                "Visibility kolana": st.column_config.NumberColumn(format="%.2f"),
                "Visibility kostki": st.column_config.NumberColumn(format="%.2f"),
            },
            hide_index=True,
        )
    else:
        st.success("Nie wykryto podejrzanych klatek.")

    st.caption(
        f"Wyeksportowano {validation['debug_frames_count']} oznaczonych klatek do: "
        f"{validation['debug_frames_dir']}"
    )


def reset_analysis_state():
    st.session_state.pop("report", None)
    st.session_state.pop("markdown_report", None)


def main():
    st.set_page_config(page_title="Analiza biegu", page_icon="RUN", layout="wide")
    inject_styles()

    st.markdown("<h2>Analiza</h2>", unsafe_allow_html=True)
    st.markdown("# Analiza biegu")
    st.markdown(
        '<div class="hero-subtitle">Prześlij nagranie biegu z boku i wygeneruj raport techniki.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Nagranie",
            type=["mp4", "mov", "avi", "mkv"],
            key="video_upload",
            on_change=reset_analysis_state,
        )
        crop_mode = st.selectbox(
            "Kadrowanie",
            ["none", "left", "right", "center"],
            index=0,
            format_func=CROP_LABELS.get,
        )
        analyze = st.button("Analizuj", width="stretch", disabled=uploaded_file is None)

    if uploaded_file is None:
        st.markdown(
            """
            <div class="status-strip">
                <div>
                    <div class="status-title">Oczekiwanie na nagranie</div>
                    <div class="status-copy">Wybierz nagranie biegu w panelu bocznym.</div>
                </div>
                <div class="pill" style="background: #d5b82f;">Gotowe</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    video_path = save_uploaded_video(uploaded_file)
    report = st.session_state.get("report")
    annotated_video_path = report.get("annotated_video_path") if report else None
    video_placeholder = st.empty()
    video_caption = st.empty()
    if annotated_video_path and Path(annotated_video_path).exists():
        video_placeholder.video(annotated_video_path)
        video_caption.caption("Nagranie z wykrytym szkieletem i kątami kolan.")
    else:
        video_placeholder.video(str(video_path))
        video_caption.caption("Po analizie nagranie zostanie zastąpione wersją ze szkieletem.")

    if analyze:
        with st.spinner("Analizowanie nagrania..."):
            report = run_analysis(video_path, crop_mode)
            st.session_state["report"] = report
            st.session_state["markdown_report"] = report_to_markdown(report)
            video_placeholder.video(report["annotated_video_path"])
            video_caption.caption("Nagranie z wykrytym szkieletem i kątami kolan.")

    report = st.session_state.get("report")
    if not report:
        return

    status_label, status_color = STATUS_META.get(report["status"], ("Weryfikacja", "#d5b82f"))
    st.markdown(
        f"""
        <div class="status-strip">
            <div>
                <div class="status-title">{report['status_label']}</div>
                <div class="status-copy">{report['summary']}</div>
            </div>
            <div class="pill" style="background: {status_color};">{status_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = build_cards(report)
    cols = st.columns(4)
    for col, html in zip(cols, cards):
        with col:
            st.markdown(html, unsafe_allow_html=True)

    render_rom_validation(report["rom_validation"])

    st.markdown("### Obserwacje")
    for observation in report["observations"]:
        st.markdown(f'<div class="observation-card">{observation}</div>', unsafe_allow_html=True)

    st.markdown("### Zalecenie")
    st.markdown(f'<div class="observation-card">{report["recommendation"]}</div>', unsafe_allow_html=True)

    markdown_report = st.session_state["markdown_report"]
    st.download_button(
        "Pobierz raport",
        data=markdown_report,
        file_name="raport_biegu.md",
        mime="text/markdown",
    )


if __name__ == "__main__":
    main()
