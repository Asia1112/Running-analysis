"""Streamlit research workspace for side-view knee-angle validation."""

import base64
import hashlib
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.knee_angle_validation.annotation_component import annotate_image
from src.knee_angle_validation.annotation_storage import (
    annotation_exists, annotation_identity, annotation_widget_key,
    read_reference_annotations, save_reference_annotation,
)
from src.knee_angle_validation.frame_quality import add_quality_columns, classify_visibility, detect_angle_jumps
from src.knee_angle_validation.knee_angle import add_pixel_angle_columns
from src.knee_angle_validation.io_utils import read_csv, write_csv
from src.knee_angle_validation.pose_extraction import process_video, read_analysis_frame
from src.knee_angle_validation.overlay import draw_overlay, export_overlay_video, validate_overlay_data
from src.knee_angle_validation.reference_annotation import (
    build_reference_annotation,
    select_reference_frames,
)
from src.knee_angle_validation.smoothing import smooth_series
from src.knee_angle_validation.statistics import grouped_statistics, visibility_correlations
from src.knee_angle_validation.validation import merge_with_reference


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = DATA_DIR / "extracted"
REFERENCE_DIR = DATA_DIR / "reference"
RESULTS_DIR = DATA_DIR / "results"
for directory in (RAW_DIR, EXTRACTED_DIR, REFERENCE_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def _initialize_state() -> None:
    """Initialize all cross-view state in one place."""
    defaults = {
        "extracted": None, "selected_frames": None, "validation": None,
        "video_path": None, "crop_mode": "none", "side": "left",
        "annotation_drafts": {},
        "annotation_session": f"session_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _uploaded_path(uploaded_file: object) -> Path:
    """Persist one uploaded video under its original extension."""
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    path = RAW_DIR / f"uploaded{suffix.lower()}"
    path.write_bytes(uploaded_file.getbuffer())
    return path


def _image_data_url(frame: np.ndarray) -> str:
    """Encode an OpenCV frame for the inline annotation component."""
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("Cannot encode annotation frame")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def _download_csv(label: str, data: pd.DataFrame | None, filename: str) -> None:
    """Render a CSV download only when data are available."""
    if data is not None and not data.empty:
        st.download_button(
            label, data=data.to_csv(index=False).encode("utf-8-sig"), file_name=filename,
            mime="text/csv", icon=":material/download:",
        )


def _angle_chart(data: pd.DataFrame) -> None:
    """Render raw and smoothed knee-angle series with suspicious frames."""
    chart = data[["time_s", "knee_flexion_deg", "knee_flexion_smoothed_deg", "frame_quality_status"]].melt(
        id_vars=["time_s", "frame_quality_status"],
        value_vars=["knee_flexion_deg", "knee_flexion_smoothed_deg"],
        var_name="series", value_name="angle_deg",
    )
    lines = alt.Chart(chart).mark_line().encode(
        x=alt.X("time_s:Q", title="Czas [s]"), y=alt.Y("angle_deg:Q", title="Zgięcie kolana 2D [°]"),
        color=alt.Color("series:N", title="Sygnał"),
        strokeDash=alt.StrokeDash("series:N"),
        tooltip=[alt.Tooltip("time_s:Q", format=".3f"), alt.Tooltip("angle_deg:Q", format=".2f")],
    )
    suspicious = data[data["frame_quality_status"] != "reliable"]
    points = alt.Chart(suspicious).mark_point(color="#e45756", filled=True, size=55).encode(
        x="time_s:Q", y="knee_flexion_deg:Q",
        tooltip=["frame_number:Q", "frame_quality_status:N", "suspicious_reason:N"],
    )
    st.altair_chart((lines + points).interactive())


def _extraction_view() -> None:
    """Render video metadata, controlled side selection, and extraction action."""
    st.subheader("Nagranie i ekstrakcja punktów")
    uploaded = st.file_uploader("Nagranie boczne jednej osoby", type=["mp4", "mov", "avi", "mkv"])
    col1, col2, col3 = st.columns(3)
    video_id = col1.text_input("Video ID", value="video_001")
    person_id = col2.text_input("Person ID", value="person_001")
    side = col3.segmented_control("Noga bliższa kamerze", ["left", "right"], default="left")
    crop_mode = st.selectbox("Kadrowanie", ["none", "left", "right", "center"])
    if uploaded is not None:
        path = _uploaded_path(uploaded)
        st.session_state.video_path = str(path)
        st.video(str(path))
    if st.button("Uruchom ekstrakcję", icon=":material/play_arrow:", disabled=uploaded is None):
        with st.spinner("Przetwarzanie klatek..."):
            result = process_video(
                st.session_state.video_path, EXTRACTED_DIR / "extracted_landmarks.csv",
                video_id, person_id, side, crop_mode=crop_mode,
            )
        st.session_state.extracted = result.data
        st.session_state.crop_mode = crop_mode
        st.session_state.side = side
        st.success(f"Przetworzono {result.frames} klatek; pozycję wykryto w {result.pose_found_pct:.1f}% klatek.")
    uploaded_csv = st.file_uploader("Lub wczytaj extracted_landmarks.csv", type=["csv"], key="extracted_csv")
    if uploaded_csv is not None:
        signature = hashlib.sha256(uploaded_csv.getvalue()).hexdigest()
        if st.session_state.get("loaded_extracted_csv_signature") != signature:
            st.session_state.extracted = pd.read_csv(uploaded_csv)
            st.session_state.loaded_extracted_csv_signature = signature
            st.session_state.selected_frames = None
            st.session_state.validation = None
            st.success("Wczytano dane ekstrakcji.")
    else:
        st.session_state.pop("loaded_extracted_csv_signature", None)


def _angle_view(data: pd.DataFrame | None) -> None:
    """Render smoothing comparison and knee-angle timeline."""
    st.subheader("Kąt kolana")
    if data is None:
        st.info("Najpierw wykonaj lub wczytaj ekstrakcję.")
        return
    st.caption("Zgięcie 2D = 180° − kąt wewnętrzny. Wyprost: 0°. Obliczenia ze współrzędnych pikselowych.")
    method = st.segmented_control("Wygładzanie", ["median", "moving_average", "savgol"], default="median")
    window = st.number_input("Okno", min_value=3, max_value=21, value=5, step=2)
    data["knee_angle_smoothed"] = smooth_series(data["knee_angle_raw"].to_numpy(), method, int(window))
    data["knee_flexion_smoothed_deg"] = 180.0 - data["knee_angle_smoothed"]
    updated = add_quality_columns(data)
    data[updated.columns] = updated
    _angle_chart(data)
    comparison = []
    for candidate in ("median", "moving_average", "savgol"):
        values = 180.0 - smooth_series(data["knee_angle_raw"].to_numpy(), candidate, int(window))
        comparison.append({
            "method": candidate,
            "jumps_10": int(detect_angle_jumps(values, 10).sum()),
            "jumps_20": int(detect_angle_jumps(values, 20).sum()),
            "jumps_30": int(detect_angle_jumps(values, 30).sum()),
            "minimum_deg": float(np.nanmin(values)), "maximum_deg": float(np.nanmax(values)),
        })
    st.dataframe(pd.DataFrame(comparison), hide_index=True)


def _quality_view(data: pd.DataFrame | None) -> None:
    """Render landmark visibility and suspicious-frame evidence."""
    st.subheader("Jakość punktów")
    if data is None:
        st.info("Brak danych ekstrakcji.")
        return
    visibility = data[["time_s", "hip_visibility", "knee_visibility", "ankle_visibility"]].melt(
        "time_s", var_name="point", value_name="visibility"
    )
    st.altair_chart(alt.Chart(visibility).mark_line().encode(
        x=alt.X("time_s:Q", title="Czas [s]"), y=alt.Y("visibility:Q", scale=alt.Scale(domain=[0, 1])), color="point:N"
    ).interactive())
    counts = data["frame_quality_status"].value_counts().rename_axis("status").reset_index(name="frames")
    st.dataframe(counts, hide_index=True)
    st.dataframe(data.loc[data["frame_quality_status"] != "reliable", [
        "frame_number", "time_s", "min_visibility", "angle_jump_deg",
        "point_stability_error", "frame_quality_status", "suspicious_reason",
    ]], hide_index=True)


def _overlay_view(data: pd.DataFrame | None) -> None:
    """Inspect selected-leg points on demand, independently of manual annotation."""
    st.subheader("Podgląd punktów na nagraniu")
    st.caption("Biodro, kolano i kostka wybranej nogi. Nakładka korzysta z zapisanych punktów MediaPipe.")
    source = st.session_state.video_path
    if data is None or data.empty or not source or not Path(source).exists():
        st.info("Wgraj film i wykonaj ekstrakcję albo wczytaj odpowiadający mu CSV.")
        return
    if not st.checkbox("Pokaż punkty na filmie", key="show_overlay"):
        return
    try:
        rows = validate_overlay_data(data)
    except (ValueError, TypeError) as error:
        st.error(str(error))
        return
    # CSV from outside this session does not contain crop/source provenance.
    crop = st.selectbox("Kadrowanie użyte podczas ekstrakcji", ["none", "left", "right", "center"],
                        index=["none", "left", "right", "center"].index(st.session_state.crop_mode),
                        key="overlay_crop")
    st.caption("Dla wczytanego CSV ustaw takie samo kadrowanie jak przy ekstrakcji. "
               "Zgodne wymiary i FPS nie gwarantują, że wybrano właściwy film.")
    frame_numbers = rows["frame_number"].tolist()
    frame_number = st.select_slider("Numer klatki (od 0)", options=frame_numbers, key="overlay_frame")
    row = rows.loc[rows["frame_number"] == frame_number].iloc[0]
    try:
        frame = read_analysis_frame(source, frame_number, crop)
        st.image(draw_overlay(frame, row), channels="BGR")
    except (ValueError, RuntimeError, cv2.error) as error:
        st.error(str(error))
        return
    st.caption("Niebieski: biodro · pomarańczowy: kolano · zielony: kostka. "
               "Brakujące punkty i punkty poza kadrem nie są dorysowywane.")
    st.dataframe(pd.DataFrame([
        {"punkt": point, "x_px": row[f"{point}_x_px"], "y_px": row[f"{point}_y_px"],
         "visibility": row.get(f"{point}_visibility", np.nan)}
        for point in ("hip", "knee", "ankle")
    ]), hide_index=True)
    st.write(f"Status klatki: {row.get('frame_quality_status', 'brak')}")
    if pd.notna(row.get("suspicious_reason")):
        st.write(f"Powód oznaczenia: {row['suspicious_reason']}")
    signature_data = rows.drop(columns=["knee_angle_smoothed"], errors="ignore").to_csv(index=False)
    source_stat = Path(source).stat()
    signature = hashlib.sha256(
        f"{source}|{source_stat.st_size}|{source_stat.st_mtime_ns}|{crop}|{signature_data}".encode()
    ).hexdigest()
    if st.button("Wygeneruj film z punktami", icon=":material/movie:"):
        output = RESULTS_DIR / "overlays" / f"overlay_{uuid4().hex}.mp4"
        try:
            with st.spinner("Nakładanie punktów na kolejne klatki..."):
                export_overlay_video(source, rows, output, crop_mode=crop)
            st.session_state.overlay_video = (signature, str(output))
        except (ValueError, RuntimeError, OSError, cv2.error) as error:
            st.error(f"Nie udało się wygenerować filmu: {error}")
    saved = st.session_state.get("overlay_video")
    if saved and saved[0] == signature and Path(saved[1]).exists():
        st.video(saved[1])
        with Path(saved[1]).open("rb") as video_file:
            st.download_button("Pobierz film z punktami", video_file,
                               file_name=Path(saved[1]).name, mime="video/mp4")
        st.caption("Podgląd MP4 bez dźwięku, maksymalnie 720 px wysokości. "
                   "Zachowuje kolejność klatek i FPS z CSV; przy VFR jest podglądem, nie źródłem czasu pomiaru.")


def _selection_view(data: pd.DataFrame | None) -> None:
    """Select by time only, including frames where MediaPipe failed."""
    st.subheader("Wybór klatek referencyjnych")
    if data is None or data.empty:
        st.info("Brak danych ekstrakcji.")
        return
    st.caption("Próba główna: środki 30 równych przedziałów czasu i najbliższe klatki. "
               "Wynik detekcji, visibility i kąt MediaPipe nie wpływają na wybór.")
    times = pd.to_numeric(data["time_s"], errors="coerce")
    if not np.isfinite(times).all() or (times < 0).any():
        st.error("CSV zawiera nieprawidłowe czasy klatek.")
        return
    lower, upper = float(times.min()), float(times.max())
    start = st.number_input("Początek fragmentu [s]", min_value=lower, max_value=upper,
                            value=lower, format="%.6f", key="selection_start")
    end = st.number_input("Koniec fragmentu [s]", min_value=lower, max_value=upper,
                          value=upper, format="%.6f", key="selection_end")
    count = st.number_input("Liczba znaczników czasu", min_value=1,
                            max_value=max(30, len(data)), value=30, key="selection_count")
    st.caption("Czas pochodzi z kolumny time_s w CSV. Zmiana ustawień wymaga ponownego wyboru klatek.")
    if st.button("Wybierz klatki według czasu", icon=":material/grid_view:", key="select_reference"):
        try:
            selected = select_reference_frames(data, int(count), start_time_s=start, end_time_s=end)
            st.session_state.selected_frames = selected
            write_csv(selected, REFERENCE_DIR / "selected_reference_frames.csv")
        except (ValueError, KeyError, TypeError) as error:
            st.session_state.selected_frames = None
            st.error(str(error))
    selected = st.session_state.selected_frames
    if selected is not None and "selection_method" not in selected:
        st.session_state.selected_frames = None
        st.info("Poprzedni wybór klatek wymaga ponowienia według nowej metody czasowej.")
        return
    if selected is not None:
        requested = int(selected["selection_requested_count"].iloc[0])
        st.write(f"Wybrano {len(selected)} unikalnych klatek dla {requested} znaczników czasu.")
        if len(selected) < requested:
            st.warning("Część znaczników wskazała tę samą klatkę. Duplikaty pominięto bez zastępowania "
                       "innymi klatkami. Rozszerz zakres lub zmniejsz liczbę znaczników.")
        st.dataframe(selected[["frame_number", "time_s", "selection_index", "selection_target_time_s",
                               "selection_time_error_s", "selection_start_s", "selection_end_s"]], hide_index=True)


def _remember_annotation_points(component_key: str) -> None:
    """Copy component events into a draft that survives widget cleanup."""
    state = st.session_state.get(component_key, {})
    points = state.get("points", []) or []
    draft = st.session_state.annotation_drafts.setdefault(component_key, {"points": [], "notes": ""})
    draft["points"] = [dict(point) for point in points]


def _annotation_view() -> None:
    """Render unbiased manual annotation and persist reference records."""
    st.subheader("Ręczna anotacja")
    selected = st.session_state.selected_frames
    video_path = st.session_state.video_path
    if selected is None or selected.empty or not video_path:
        st.info("Wybierz klatki referencyjne i zachowaj dostęp do filmu źródłowego.")
        return
    frame_numbers = selected["frame_number"].astype(int).tolist()
    frame_number = st.selectbox("Klatka", frame_numbers, key="annotation_frame")
    annotator_id = st.text_input("Annotator ID", value="annotator_01", key="annotation_annotator")
    if st.button("Nowa sesja pomiarowa", key="new_annotation_session"):
        st.session_state.annotation_session = f"session_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    session_id = st.text_input("Sesja", key="annotation_session")
    st.caption("Do powtórnego pomiaru użyj nowej sesji. Poprzednie punkty nie są w niej wyświetlane. "
               "Szkice punktów przetrwają zmianę klatki w bieżącym połączeniu; zapis CSV zachowuje pomiary trwale.")
    selected_row = selected.loc[selected["frame_number"] == frame_number].iloc[0]
    try:
        identity = annotation_identity(selected_row, annotator_id, session_id)
        frame = read_analysis_frame(video_path, frame_number, st.session_state.crop_mode)
        image_url = _image_data_url(frame)
        image_hash = hashlib.sha256(base64.b64decode(image_url.split(",", 1)[1])).hexdigest()
        component_key = annotation_widget_key(identity, image_hash)
        path = REFERENCE_DIR / "reference_annotations.csv"
        existing = read_reference_annotations(path) if path.exists() else pd.DataFrame()
        already_saved = annotation_exists(existing, identity)
    except (ValueError, KeyError, RuntimeError, OSError) as error:
        st.error(str(error))
        return
    draft = st.session_state.annotation_drafts.setdefault(component_key, {"points": [], "notes": ""})
    annotate_image(image_url, points=draft["points"], key=component_key,
                   on_points_change=lambda: _remember_annotation_points(component_key))
    points = draft["points"]
    st.caption(f"Zaznaczono {len(points)}/3 punktów.")
    notes = st.text_input("Notatki", value=draft["notes"], key=f"notes_{component_key}")
    draft["notes"] = notes
    if already_saved:
        st.info("Ta klatka jest już zapisana dla tego anotatora i sesji. Kolejny pomiar wymaga nowej sesji.")
    if st.button("Zapisz anotację", icon=":material/save:", key="save_annotation",
                 disabled=len(points) != 3 or already_saved):
        try:
            coordinates = [(float(point["x"]), float(point["y"])) for point in points]
            record = build_reference_annotation(selected_row, coordinates, annotator_id, session_id, notes)
            record.update(frame_image_sha256=image_hash, image_width=frame.shape[1], image_height=frame.shape[0],
                          annotated_at=datetime.now().astimezone().isoformat())
            save_reference_annotation(record, path)
            st.session_state.validation = None
            st.success("Zapisano anotację bez zastępowania wcześniejszych pomiarów.")
        except (ValueError, KeyError, OSError) as error:
            st.error(str(error))


def _validation_view(data: pd.DataFrame | None) -> None:
    """Render reference merge and per-frame validation results."""
    st.subheader("Walidacja")
    st.caption("Błędy dotyczą zgięcia 2D: MediaPipe − referencja. Dodatni bias oznacza większe zgięcie MediaPipe.")
    reference_upload = st.file_uploader("reference_annotations.csv", type=["csv"], key="reference_csv")
    reference_path = REFERENCE_DIR / "reference_annotations.csv"
    reference = read_reference_annotations(reference_upload) if reference_upload else (read_reference_annotations(reference_path) if reference_path.exists() else None)
    if data is None or reference is None:
        st.info("Potrzebne są dane ekstrakcji i anotacje referencyjne.")
        return
    if not {"annotator_id", "annotation_session"}.issubset(reference.columns):
        st.error("Referencja musi zawierać anotatora i sesję pomiarową.")
        return
    pairs = list(reference[["annotator_id", "annotation_session"]].drop_duplicates().itertuples(index=False, name=None))
    if not pairs:
        st.info("Brak anotacji referencyjnych.")
        return
    pair = st.selectbox("Referencja do porównania (anotator / sesja)", pairs,
                        format_func=lambda item: f"{item[0]} / {item[1]}", key="validation_session")
    reference = reference.loc[reference.annotator_id.eq(pair[0]) & reference.annotation_session.eq(pair[1])]
    reference_signature = hashlib.sha256(reference.to_csv(index=False).encode()).hexdigest()
    if st.session_state.get("validation_reference_signature") != reference_signature:
        st.session_state.validation = None
        st.session_state.validation_reference_signature = reference_signature
    if st.button("Połącz i oblicz błędy", icon=":material/compare_arrows:"):
        validation = merge_with_reference(data, reference)
        st.session_state.validation = validation
        write_csv(validation, RESULTS_DIR / "validation_results.csv")
    validation = st.session_state.validation
    if validation is not None:
        st.dataframe(validation[[
            "video_id", "frame_number", "side", "angle_mediapipe", "angle_reference",
            "error_signed", "error_absolute", "hip_distance_px", "knee_distance_px", "ankle_distance_px",
        ]], hide_index=True)


def _statistics_view() -> None:
    """Render error summaries, visibility bins, correlations, and point-error evidence."""
    st.subheader("Analiza statystyczna")
    validation = st.session_state.validation
    if validation is None or validation.empty:
        st.info("Najpierw wykonaj walidację.")
        return
    validation = validation.copy()
    validation["visibility_group"] = validation["min_visibility"].map(classify_visibility)
    summary = grouped_statistics(validation, ["video_id", "person_id", "visibility_group"])
    summary["angle_definition"] = "knee_flexion_2d_deg"
    write_csv(summary, RESULTS_DIR / "summary_statistics.csv")
    st.dataframe(summary, hide_index=True)
    scatter = alt.Chart(validation).mark_circle(size=65).encode(
        x=alt.X("min_visibility:Q", title="Minimalne visibility"),
        y=alt.Y("error_absolute:Q", title="Błąd bezwzględny [°]"),
        color="frame_quality_status:N", tooltip=["video_id:N", "frame_number:Q", "error_absolute:Q"],
    )
    st.altair_chart(scatter.interactive())
    st.dataframe(visibility_correlations(validation), hide_index=True)
    st.warning("Visibility jest wskaźnikiem pewności detekcji MediaPipe, a nie bezpośrednią miarą dokładności punktu.")
    point_summary = pd.DataFrame([
        {"point": point, "mean_error_px": validation[f"{point}_distance_px"].mean(),
         "angle_error_correlation": validation[f"{point}_distance_px"].corr(validation["error_absolute"])}
        for point in ("hip", "knee", "ankle")
    ])
    st.dataframe(point_summary, hide_index=True)


def _export_view(data: pd.DataFrame | None) -> None:
    """Render all research-data downloads and an HTML summary."""
    st.subheader("Eksport")
    _download_csv("Punkty MediaPipe", data, "extracted_landmarks.csv")
    _download_csv("Wybrane klatki", st.session_state.selected_frames, "selected_reference_frames.csv")
    reference_path = REFERENCE_DIR / "reference_annotations.csv"
    _download_csv("Anotacje referencyjne", read_reference_annotations(reference_path) if reference_path.exists() else None, "reference_annotations.csv")
    _download_csv("Wyniki walidacji", st.session_state.validation, "validation_results.csv")
    if data is not None:
        suspicious = data[data["frame_quality_status"] != "reliable"]
        _download_csv("Podejrzane klatki", suspicious, "suspicious_frames.csv")
    summary_path = RESULTS_DIR / "summary_statistics.csv"
    _download_csv("Statystyki", read_csv(summary_path) if summary_path.exists() else None, "summary_statistics.csv")
    html = "<h1>Walidacja kąta stawu kolanowego</h1><p>Raport pomiarowy z nagrania bocznego. System nie jest narzędziem medycznym.</p>"
    st.download_button("Podsumowanie HTML", html.encode("utf-8"), "validation_summary.html", "text/html", icon=":material/description:")


def main() -> None:
    """Run the Streamlit research workspace."""
    st.set_page_config(page_title="Walidacja kąta kolana", page_icon=":material/monitoring:", layout="wide")
    _initialize_state()
    st.title("Walidacja kąta stawu kolanowego")
    st.caption("Pomiar 2D z nagrania bocznego z wykorzystaniem MediaPipe")
    tabs = st.tabs([
        "Nagranie i ekstrakcja", "Kąt kolana", "Jakość punktów", "Wybór klatek",
        "Ręczna anotacja", "Walidacja", "Analiza statystyczna", "Eksport", "Podgląd szkieletu",
    ])
    with tabs[0]: _extraction_view()
    data = st.session_state.extracted
    if data is not None and (
        "angle_geometry" not in data or not data["angle_geometry"].eq("pixel_2d_v1").all()
        or "knee_flexion_deg" not in data or "knee_internal_angle_deg" not in data
    ):
        try:
            data = add_pixel_angle_columns(data)
            data["knee_angle_smoothed"] = smooth_series(data["knee_angle_raw"].to_numpy(), "median", 5)
            data["knee_flexion_smoothed_deg"] = 180.0 - data["knee_angle_smoothed"]
            data = add_quality_columns(data)
            st.session_state.extracted = data
            st.session_state.validation = None
            st.session_state.selected_frames = None
            st.info("Przeliczono kąty i jakość z zapisanych pikseli. Plik źródłowy CSV nie został zmieniony; "
                    "poprawione dane można pobrać w zakładce Eksport.")
        except (ValueError, KeyError) as error:
            st.error(f"Nie można przeliczyć kątów starego CSV: {error}")
            return
    previous_validation = st.session_state.validation
    if previous_validation is not None and "angle_definition" not in previous_validation:
        st.session_state.validation = None
    with tabs[1]: _angle_view(data)
    with tabs[2]: _quality_view(data)
    with tabs[3]: _selection_view(data)
    with tabs[4]: _annotation_view()
    with tabs[5]: _validation_view(data)
    with tabs[6]: _statistics_view()
    with tabs[7]: _export_view(data)
    with tabs[8]: _overlay_view(data)


if __name__ == "__main__":
    main()
