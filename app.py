from pathlib import Path

import streamlit as st

from src.running_biomechanics.analysis import analyze_csv
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
    process_video(
        source=str(video_path),
        csv_path=str(csv_path),
        crop_mode=crop_mode,
        resize_height=720,
        preview=False,
    )
    result = analyze_csv(str(csv_path))
    return build_report(result, source_name=video_path.name)


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
        uploaded_file = st.file_uploader("Nagranie", type=["mp4", "mov", "avi", "mkv"])
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
    st.video(str(video_path))

    if analyze:
        with st.spinner("Analizowanie nagrania..."):
            report = run_analysis(video_path, crop_mode)
            st.session_state["report"] = report
            st.session_state["markdown_report"] = report_to_markdown(report)

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
