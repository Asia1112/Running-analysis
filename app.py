from pathlib import Path

import altair as alt
import streamlit as st

from src.running_biomechanics.analysis import analyze_csv, validate_knee_rom
from src.running_biomechanics.extended_analysis import analyze_extended_metrics
from src.running_biomechanics.exports import (
    build_angles_csv,
    build_debug_frames_zip,
    build_pdf_report,
    build_suspicious_csv,
    suspicious_rows,
)
from src.running_biomechanics.pose_extraction import process_video
from src.running_biomechanics.report import build_report, report_to_markdown


APP_RUNS_DIR = Path("app_runs")
APP_RUNS_DIR.mkdir(exist_ok=True)


STATUS_META = {
    "usable": ("Dobry", "#65d44d"),
    "needs_review": ("Wymaga weryfikacji", "#d5b82f"),
    "low_confidence": ("Niewiarygodny", "#e45835"),
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
            background: #f7f7f4;
            color: #171a1c !important;
            font-weight: 700;
        }

        .stDownloadButton > button p,
        .stDownloadButton > button span,
        .stDownloadButton > button [data-testid="stIconMaterial"] {
            color: #171a1c !important;
            fill: #171a1c !important;
        }

        .stDownloadButton > button:hover {
            background: #ff7a2f;
            border-color: #ff7a2f;
            color: #111 !important;
        }

        [data-testid="stMetric"] {
            background: #111416;
            border-color: rgba(255, 255, 255, 0.12) !important;
            min-height: 8.5rem;
        }

        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricLabel"] span {
            color: #c9cdd1 !important;
            opacity: 1 !important;
        }

        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] div,
        [data-testid="stMetricValue"] span {
            color: #ffffff !important;
            opacity: 1 !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            word-break: normal !important;
            overflow-wrap: anywhere !important;
        }

        [data-testid="stMetricValue"] > div {
            font-size: clamp(1.45rem, 2.1vw, 2rem) !important;
            line-height: 1.15 !important;
            max-width: 100% !important;
        }

        [data-testid="stMetricDelta"] div,
        [data-testid="stMetricDelta"] span {
            opacity: 1 !important;
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
    report["extended_metrics"] = analyze_extended_metrics(
        str(csv_path), result=result, validation=validation
    )
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
        report["extended_metrics"]["reliability"]["overall"] = {
            "level": "review",
            "label": "wymaga weryfikacji",
            "reason": "Walidacja wykazała podejrzany spadek ROM po wygładzeniu.",
        }
    report["rom_validation"] = validation
    report["annotated_video_path"] = processing_stats["annotated_video_path"]
    report["csv_path"] = str(csv_path)
    report["processing_stats"] = processing_stats
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


def render_rom_validation(validation, reliability=None):
    st.subheader("Kolana")

    if reliability:
        render_reliability(reliability, "Wiarygodność ROM kolan")

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

    left_side = validation["sides"]["left"]
    right_side = validation["sides"]["right"]
    raw_difference = abs(left_side["raw_rom_deg"] - right_side["raw_rom_deg"])
    smoothed_difference = abs(
        left_side["smoothed_rom_deg"] - right_side["smoothed_rom_deg"]
    )
    summary_columns = st.columns(3)
    summary_columns[0].metric("Różnica ROM L/P — surowa", f"{raw_difference:.1f}°")
    summary_columns[1].metric("Różnica ROM L/P — wygładzona", f"{smoothed_difference:.1f}°")
    summary_columns[2].metric(
        "Stabilność ROM po wygładzeniu",
        "Podejrzana" if validation["tracking_error"] else "Stabilna",
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

def display_value(value, suffix="", digits=1):
    if value is None:
        return "Brak danych"
    return f"{value:.{digits}f}{suffix}"


def render_reliability(reliability, title="Wiarygodność"):
    message = f"{title}: {reliability['label']}. {reliability['reason']}"
    if reliability["level"] == "good":
        st.success(message, icon=":material/check_circle:")
    elif reliability["level"] == "review":
        st.warning(message, icon=":material/warning:")
    else:
        st.error(message, icon=":material/error:")


def render_angle_chart(series, y_title):
    color = alt.Color(
        "side_label:N",
        title="Strona",
        scale=alt.Scale(domain=["Lewa", "Prawa"], range=["#ff7a2f", "#65d44d"]),
    )
    base = alt.Chart(alt.Data(values=series)).encode(
        x=alt.X("time_s:Q", title="Czas [s]"),
        color=color,
    )
    raw = base.mark_line(strokeWidth=1.3, opacity=0.45).encode(
        y=alt.Y("angle_raw_deg:Q", title=y_title),
        tooltip=[
            alt.Tooltip("time_s:Q", title="Czas [s]", format=".3f"),
            alt.Tooltip("side_label:N", title="Strona"),
            alt.Tooltip("angle_raw_deg:Q", title="Kąt surowy [°]", format=".1f"),
        ],
    )
    smooth = base.mark_line(strokeWidth=2.5).encode(
        y=alt.Y("angle_smoothed_deg:Q", title=y_title),
        tooltip=[
            alt.Tooltip("time_s:Q", title="Czas [s]", format=".3f"),
            alt.Tooltip("side_label:N", title="Strona"),
            alt.Tooltip("angle_smoothed_deg:Q", title="Kąt wygładzony [°]", format=".1f"),
        ],
    )
    st.altair_chart((raw + smooth).interactive())


def render_analysis_quality(details):
    st.subheader("Jakość detekcji")
    quality = details["quality"]
    columns = st.columns(4)
    columns[0].metric("Widoczność głównych punktów", f"{details['core_visible_pct']:.1f}%", border=True)
    columns[1].metric("Średnia jakość punktów", f"{quality['mean_visible_pct']:.1f}%", border=True)
    columns[2].metric("Podejrzane klatki", quality["suspicious_frame_count"], border=True)
    columns[3].metric(
        "Stabilność kątów kolan",
        display_value(quality["angle_stability_pct"], "%"),
        border=True,
    )
    if quality["level"] == "good":
        st.success(f"Ocena jakości detekcji: {quality['label']}.", icon=":material/check_circle:")
    elif quality["level"] == "review":
        st.warning(f"Ocena jakości detekcji: {quality['label']}.", icon=":material/warning:")
    else:
        st.error(f"Ocena jakości detekcji: {quality['label']}.", icon=":material/error:")
    st.caption(
        "Stabilność oznacza odsetek kolejnych klatek, w których zmiana kąta nie przekracza 10°."
    )
    st.markdown("**Najgorsze punkty:** " + ", ".join(
        f"{item['punkt']} ({item['visible_pct']}%)" for item in quality["worst_landmarks"]
    ))
    with st.expander("Tabela jakości wszystkich punktów", expanded=True):
        st.dataframe(
            quality["landmarks"],
            column_config={
                "punkt": "Punkt",
                "mean_visibility": st.column_config.NumberColumn("Średnia visibility", format="%.2f"),
                "visible_pct": st.column_config.ProgressColumn(
                    "Klatki z visibility > 0.6", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
            hide_index=True,
        )


def render_cadence_section(report, extended):
    st.subheader("Kadencja i rytm")
    metrics = report["metrics"]
    cadence = extended["cadence"]
    columns = st.columns(4)
    columns[0].metric("Kadencja ogólna [kroków/min]", f"{metrics['cadence_spm']:.1f}", border=True)
    columns[1].metric("Kadencja L/P [kroków/min]", f"{metrics['cadence_left_spm']:.1f} / {metrics['cadence_right_spm']:.1f}", border=True)
    columns[2].metric("Różnica L/P [kroków/min]", f"{metrics['lr_cadence_difference_spm']:.1f}", border=True)
    columns[3].metric("Wykryte zdarzenia", report["analysis_window"]["events"], border=True)
    st.metric(
        "Regularność kroków",
        "Brak danych" if cadence["regularity_label"] == "niewystarczające dane" else cadence["regularity_label"].capitalize(),
        display_value(cadence["regularity_cv_pct"], "% CV"),
        border=True,
    )
    st.caption(report["observations"][0])
    render_reliability(extended["reliability"]["cadence"], "Wiarygodność kadencji")
    st.caption(
        "Niższy współczynnik zmienności (CV) odstępów między krokami oznacza bardziej regularny rytm. "
        "Wynik wymaga co najmniej trzech wykrytych zdarzeń na stronę."
    )


def render_hip_section(hips, reliability=None):
    st.subheader("Biodra")
    if reliability:
        render_reliability(reliability, "Wiarygodność ROM bioder")
    if not hips["available"]:
        st.warning("Widoczność barków, bioder lub kolan jest zbyt niska dla wiarygodnego ROM bioder.")
        return
    render_angle_chart(hips["series"], "Kąt biodra [°]")
    columns = st.columns(3)
    columns[0].metric("ROM biodra L", display_value(hips["left"]["smoothed_rom_deg"], "°"))
    columns[1].metric("ROM biodra P", display_value(hips["right"]["smoothed_rom_deg"], "°"))
    columns[2].metric("Różnica L/P", display_value(hips["rom_difference_deg"], "°"))
    if hips["symmetry_label"] == "dobra":
        symmetry_value = "Dobra"
    elif "wyraźna różnica" in hips["symmetry_label"]:
        symmetry_value = "Wyraźna różnica"
    elif "weryfikacji" in hips["symmetry_label"]:
        symmetry_value = "Weryfikacja"
    else:
        symmetry_value = "Brak danych"
    st.metric("Ocena symetrii bioder", symmetry_value, border=True)
    st.caption(f"Pełna ocena symetrii: {hips['symmetry_label']}.")
    st.caption("Kąt biodra jest liczony w 2D z punktów bark–biodro–kolano; pokazany ROM używa danych wygładzonych.")


def render_trunk_section(trunk, reliability=None):
    st.subheader("Tułów")
    if reliability:
        render_reliability(reliability, "Wiarygodność pochylenia tułowia")
    columns = st.columns(4)
    columns[0].metric("Średnie pochylenie", display_value(trunk["mean_deg"], "°"))
    columns[1].metric("Zakres min/max", f"{display_value(trunk['min_deg'], '°')} / {display_value(trunk['max_deg'], '°')}")
    columns[2].metric("Zmienność pochylenia", display_value(trunk["std_deg"], "° SD"))
    columns[3].metric("Stabilność sylwetki", display_value(trunk["stability_pct"], "%"))
    if trunk["stability_pct"] is not None and trunk["stability_pct"] >= 90:
        st.success("Tułów wygląda stabilnie w większości kolejnych klatek.")
    else:
        st.warning("Zmienność tułowia może wskazywać na niestabilność albo błąd punktów i wymaga weryfikacji.")
    st.caption("Stabilność oznacza odsetek zmian między klatkami nieprzekraczających 3°.")


def render_technique_section(technique, reliability=None):
    st.subheader("Wskaźniki techniczne")
    if reliability:
        render_reliability(reliability, "Wiarygodność overstriding proxy")
    columns = st.columns(3)
    if technique["overstriding_available"]:
        columns[0].metric(
            "Możliwy overstriding",
            "Tak" if technique["possible_overstriding"] else "Nie wykryto",
            display_value(technique["overstriding_offset_leg_ratio"], "", 2),
        )
        columns[0].caption(
            f"Pozycja stopy L/P: {display_value(technique['overstriding_left_ratio'], '', 2)} / "
            f"{display_value(technique['overstriding_right_ratio'], '', 2)} długości podudzia"
        )
    else:
        columns[0].metric("Możliwy overstriding", "Brak danych")
    if technique["contact_available"]:
        columns[1].metric(
            "Kąt piszczeli przy kontakcie L/P",
            f"{display_value(technique['shin_angle_left_deg'], '°')} / "
            f"{display_value(technique['shin_angle_right_deg'], '°')}",
        )
    else:
        columns[1].metric("Kąt piszczeli przy kontakcie", "Brak danych")
    columns[2].metric(
        "Oscylacja bioder [% długości tułowia]",
        display_value(technique["vertical_hip_oscillation_pct_torso"], "%"),
    )
    st.info(technique["note"])
    if not technique["direction_available"]:
        st.caption("Overstriding ukryto, ponieważ z nagrania nie można pewnie ustalić kierunku przemieszczania się biegacza.")


def render_arms_section(arms):
    st.markdown("### 7. Praca ramion")
    if not arms["available"]:
        st.warning(
            "Brak wystarczająco widocznych łokci i nadgarstków. Nowe analizy zapisują te punkty; "
            "dla starszego pliku CSV trzeba ponownie przetworzyć film."
        )
        return
    render_angle_chart(arms["series"], "Kąt łokcia [°]")
    columns = st.columns(4)
    columns[0].metric("ROM ramienia L", display_value(arms["left_arm_rom_deg"], "°"))
    columns[1].metric("ROM ramienia P", display_value(arms["right_arm_rom_deg"], "°"))
    columns[2].metric("Różnica ROM", display_value(arms["rom_difference_deg"], "°"))
    columns[3].metric("Symetria rąk", display_value(arms["symmetry_score"], "/100"))
    st.caption(arms["note"])


def render_preview_tab(report):
    st.subheader("Podgląd analizy")
    annotated_path = report.get("annotated_video_path")
    if annotated_path and Path(annotated_path).exists():
        st.video(annotated_path)
        st.caption("Nagranie z punktami MediaPipe, szkieletem i kątami kolan.")
    else:
        st.warning("Nie znaleziono wygenerowanego filmu ze szkieletem.")

    extended = report["extended_metrics"]
    window = report["analysis_window"]
    status_label = {
        "usable": "Dobry",
        "needs_review": "Weryfikacja",
        "low_confidence": "Niewiarygodny",
    }.get(report["status"], "Weryfikacja")
    columns = st.columns(4)
    columns[0].metric("Przeanalizowane klatki", extended["frame_count"], border=True)
    columns[1].metric("Długość nagrania", display_value(extended["duration_s"], " s"), border=True)
    columns[2].metric("Zakres analizy", f"{window['start_s']}–{window['end_s']} s", border=True)
    columns[3].metric("Status raportu", status_label, border=True)
    render_reliability(extended["reliability"]["overall"], "Wiarygodność całego raportu")
    with st.expander("Obserwacje i zalecenie", expanded=True, icon=":material/description:"):
        for observation in report["observations"]:
            st.markdown(f"- {observation}")
        st.markdown(f"**Zalecenie:** {report['recommendation']}")
    st.caption(
        "Wyniki są orientacyjnymi wskaźnikami technicznymi z obrazu 2D i nie stanowią diagnozy medycznej."
    )


def render_suspicious_frames(report):
    st.subheader("Podejrzane klatki")
    rows = suspicious_rows(report)
    if not rows:
        st.success("Nie wykryto podejrzanych klatek.", icon=":material/check_circle:")
    else:
        st.warning(
            f"Znaleziono {len(report['rom_validation']['suspicious_frame_numbers'])} unikalnych podejrzanych klatek "
            f"({len(rows)} wpisów L/P).",
            icon=":material/warning:",
        )
        st.dataframe(
            rows,
            column_config={
                "frame": st.column_config.NumberColumn("Klatka", format="%d", pinned=True),
                "time_s": st.column_config.NumberColumn("Czas [s]", format="%.3f"),
                "side": "Strona",
                "problem": "Problematyczny punkt",
                "problem_visibility": st.column_config.NumberColumn("Visibility punktu", format="%.2f"),
                "hip_visibility": st.column_config.NumberColumn("Visibility biodra", format="%.2f"),
                "knee_visibility": st.column_config.NumberColumn("Visibility kolana", format="%.2f"),
                "ankle_visibility": st.column_config.NumberColumn("Visibility kostki", format="%.2f"),
                "angle_deg": st.column_config.NumberColumn("Kąt [°]", format="%.1f"),
                "angle_change_deg": st.column_config.NumberColumn("Zmiana [°]", format="%.1f"),
                "reason": "Powód oznaczenia",
            },
            hide_index=True,
        )

    debug_dir = report["rom_validation"].get("debug_frames_dir")
    images = []
    if debug_dir and Path(debug_dir).exists():
        images = sorted(
            path for path in Path(debug_dir).iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    st.caption(f"Zapisane obrazy debug: {len(images)} · folder: {debug_dir or 'brak'}")
    if images:
        with st.expander("Podgląd zapisanych klatek", icon=":material/image:"):
            for start in range(0, min(len(images), 12), 3):
                columns = st.columns(3)
                for column, image_path in zip(columns, images[start:start + 3]):
                    column.image(str(image_path), caption=image_path.stem)
            if len(images) > 12:
                st.caption("Pokazano pierwsze 12 obrazów. Pełny zestaw można pobrać jako ZIP.")


def render_export_tab(report):
    st.subheader("Eksport wyników")
    st.caption("Pliki są generowane dla aktualnie przeanalizowanego nagrania.")
    csv_path = Path(report["csv_path"])
    pdf_bytes = build_pdf_report(report)
    angles_csv = build_angles_csv(report)
    suspicious_csv = build_suspicious_csv(report)
    debug_zip = build_debug_frames_zip(report["rom_validation"].get("debug_frames_dir"))

    with st.container(horizontal=True):
        st.download_button(
            "Raport PDF",
            data=pdf_bytes,
            file_name="raport_biegu.pdf",
            mime="application/pdf",
            icon=":material/picture_as_pdf:",
        )
        st.download_button(
            "Punkty MediaPipe CSV",
            data=csv_path.read_bytes(),
            file_name="punkty_mediapipe.csv",
            mime="text/csv",
            icon=":material/table_view:",
        )
        st.download_button(
            "Kąty CSV",
            data=angles_csv,
            file_name="katy_biegu.csv",
            mime="text/csv",
            icon=":material/query_stats:",
        )
        st.download_button(
            "Podejrzane klatki CSV",
            data=suspicious_csv,
            file_name="podejrzane_klatki.csv",
            mime="text/csv",
            icon=":material/warning:",
        )
        if debug_zip:
            st.download_button(
                "Klatki debug ZIP",
                data=debug_zip,
                file_name="debug_frames.zip",
                mime="application/zip",
                icon=":material/folder_zip:",
            )
    st.download_button(
        "Raport tekstowy Markdown",
        data=report_to_markdown(report),
        file_name="raport_biegu.md",
        mime="text/markdown",
        icon=":material/description:",
    )


def render_report_tabs(report):
    extended = report["extended_metrics"]
    tabs = st.tabs([
        "Podgląd",
        "Jakość detekcji",
        "Kadencja i rytm",
        "Kolana",
        "Biodra",
        "Tułów",
        "Wskaźniki techniczne",
        "Podejrzane klatki",
        "Eksport",
    ], key="report_tabs", on_change="rerun")
    if tabs[0].open:
        with tabs[0]:
            render_preview_tab(report)
    if tabs[1].open:
        with tabs[1]:
            render_analysis_quality(extended)
    if tabs[2].open:
        with tabs[2]:
            render_cadence_section(report, extended)
    if tabs[3].open:
        with tabs[3]:
            render_rom_validation(report["rom_validation"], extended["reliability"]["knee_rom"])
    if tabs[4].open:
        with tabs[4]:
            render_hip_section(extended["hips"], extended["reliability"]["hip_rom"])
    if tabs[5].open:
        with tabs[5]:
            render_trunk_section(extended["trunk"], extended["reliability"]["trunk"])
    if tabs[6].open:
        with tabs[6]:
            render_technique_section(extended["technique"], extended["reliability"]["overstriding"])
            with st.expander("Dodatkowo: praca ramion", icon=":material/accessibility_new:"):
                render_arms_section(extended["arms"])
    if tabs[7].open:
        with tabs[7]:
            render_suspicious_frames(report)
    if tabs[8].open:
        with tabs[8]:
            render_export_tab(report)


def reset_analysis_state():
    st.session_state.pop("report", None)
    st.session_state.pop("markdown_report", None)
    st.session_state.pop("report_tabs", None)


def main():
    st.set_page_config(page_title="Analiza biegu", page_icon=":material/directions_run:", layout="wide")
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
    if not report:
        st.video(str(video_path))
        st.caption("Podgląd oryginalnego nagrania. Po analizie w zakładce Podgląd pojawi się film ze szkieletem.")

    if analyze:
        with st.spinner("Analizowanie nagrania..."):
            report = run_analysis(video_path, crop_mode)
            st.session_state["report"] = report
            st.session_state["markdown_report"] = report_to_markdown(report)

    report = st.session_state.get("report")
    if not report:
        return

    status_label, status_color = STATUS_META.get(report["status"], ("Wymaga weryfikacji", "#d5b82f"))
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
    st.caption("Skrót wyników. Szczegóły, wiarygodność i eksport znajdują się w zakładkach poniżej.")
    render_report_tabs(report)


if __name__ == "__main__":
    main()
