"""Diagnostic selected-leg overlays from saved points, without new inference."""

from pathlib import Path
from uuid import uuid4

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

from .pose_extraction import crop_frame, resize_frame
from .provenance import PROVENANCE_KEYS, RANGE_KEYS, require_source_match, file_sha256, frame_limits


POINTS = {"hip": ((255, 160, 60), "Biodro"),
          "knee": ((40, 150, 255), "Kolano"),
          "ankle": ((70, 220, 90), "Kostka")}


def validate_overlay_data(data: pd.DataFrame) -> pd.DataFrame:
    """Require one recording/side and unique source-frame indexes."""
    required = {"video_id", "person_id", "side", "frame_number", "time_s", "fps",
                "image_width", "image_height"}
    required.update(f"{point}_{axis}_px" for point in POINTS for axis in ("x", "y"))
    missing = required.difference(data.columns)
    if missing or data.empty:
        raise ValueError(f"Brak danych lub kolumn nakładki: {', '.join(sorted(missing))}")
    result = data.copy()
    for column in ("video_id", "person_id", "side", "fps", "image_width", "image_height"):
        if result[column].isna().any() or result[column].nunique() != 1:
            raise ValueError(f"Nakładka wymaga jednej wartości {column}.")
    if result["side"].iloc[0] not in ("left", "right"):
        raise ValueError("Nieprawidłowa strona ciała.")
    frames = pd.to_numeric(result["frame_number"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(frames).all() or (frames < 0).any() or (frames != np.floor(frames)).any():
        raise ValueError("Numery klatek muszą być nieujemnymi liczbami całkowitymi.")
    result["frame_number"] = frames.astype(int)
    if result["frame_number"].duplicated().any():
        raise ValueError("Powielone numery klatek w danych nakładki.")
    for column in ("fps", "image_width", "image_height"):
        value = float(result[column].iloc[0])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Nieprawidłowa wartość {column}.")
    return result.sort_values("frame_number").reset_index(drop=True)


def draw_overlay(frame: np.ndarray, row: pd.Series) -> np.ndarray:
    """Draw finite in-image points only; never fill missing detections."""
    height, width = frame.shape[:2]
    if (width, height) != (int(row["image_width"]), int(row["image_height"])):
        raise ValueError("Wymiary klatki nie zgadzają się z CSV. Sprawdź film i kadrowanie.")
    result = frame.copy()
    positions = {}
    for point in POINTS:
        x, y = float(row[f"{point}_x_px"]), float(row[f"{point}_y_px"])
        if np.isfinite([x, y]).all() and 0 <= x < width and 0 <= y < height:
            positions[point] = (min(round(x), width - 1), min(round(y), height - 1))
    thickness = max(2, round(height / 300))
    for start, end in (("hip", "knee"), ("knee", "ankle")):
        if start in positions and end in positions:
            cv2.line(result, positions[start], positions[end], (0, 230, 255), thickness, cv2.LINE_AA)
    for point, position in positions.items():
        color, label = POINTS[point]
        cv2.circle(result, position, thickness * 2, color, -1, cv2.LINE_AA)
        cv2.putText(result, label, (position[0] + 10, position[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    side = "prawa" if row["side"] == "right" else "lewa"
    label = f"Klatka {int(row['frame_number'])} | {float(row['time_s']):.3f} s | Noga: {side}"
    cv2.rectangle(result, (0, 0), (width, min(80, height)), (20, 20, 20), -1)
    cv2.putText(result, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    status = str(row.get("frame_quality_status", "unknown"))
    cv2.putText(result, f"Status: {status} | Punkty: {len(positions)}/3", (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def export_overlay_video(source: str | Path, data: pd.DataFrame, output: str | Path,
                         crop_mode: str = "none", resize_height: int = 720) -> Path:
    """Export a silent H.264 CFR preview; preserve source frame order and FPS."""
    rows = validate_overlay_data(data)
    interval = None
    if set(PROVENANCE_KEYS).intersection(rows.columns):
        metadata = require_source_match(rows, file_sha256(source))
        if crop_mode != metadata["crop_mode"] or resize_height != metadata["resize_height"]:
            raise ValueError("Kadrowanie i skalowanie muszą odpowiadać metadanym ekstrakcji.")
        if metadata.get("time_source") == "ffmpeg_pts_v1":
            interval = (metadata["analysis_first_frame"], metadata["analysis_stop_frame"])
        elif set(RANGE_KEYS).issubset(metadata):
            interval = frame_limits(metadata["analysis_start_s"], metadata["analysis_end_s"], float(rows.fps.iloc[0]))
    expected = np.arange(*interval) if interval else np.arange(len(rows))
    if not np.array_equal(rows["frame_number"].to_numpy(), expected):
        raise ValueError("Eksport wymaga wszystkich kolejnych klatek wybranego zakresu, bez luk.")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Plik już istnieje: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}_{uuid4().hex}.mp4")
    capture = cv2.VideoCapture(str(source))
    writer = None
    try:
        if not capture.isOpened():
            raise ValueError("Nie można otworzyć nagrania.")
        fps = float(rows["fps"].iloc[0])
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(source_fps) or not np.isclose(source_fps, fps, rtol=0.001):
            raise ValueError("FPS filmu nie zgadza się z ekstrakcją.")
        for _ in range(int(rows.frame_number.iloc[0])):
            if not capture.read()[0]:
                raise ValueError("Film kończy się przed początkiem wybranego fragmentu.")
        for _, row in rows.iterrows():
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Film jest krótszy niż ekstrakcja lub wystąpił błąd odczytu.")
            frame = resize_frame(crop_frame(frame, crop_mode), resize_height)
            frame = draw_overlay(frame, row)
            # Reduce only the diagnostic video; point coordinates stay unchanged.
            h, w = frame.shape[:2]
            if h > 720:
                frame = cv2.resize(frame, (round(w * 720 / h), 720), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
            frame = cv2.copyMakeBorder(frame, 0, h % 2, 0, w % 2, cv2.BORDER_CONSTANT)
            if writer is None:
                writer = imageio_ffmpeg.write_frames(
                    str(temporary), (frame.shape[1], frame.shape[0]), fps=fps,
                    codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                    macro_block_size=1, ffmpeg_log_level="error",
                    output_params=["-movflags", "+faststart"],
                )
                writer.send(None)
            writer.send(np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if interval is None and capture.read()[0]:
            raise ValueError("Film ma więcej klatek niż ekstrakcja. Sprawdź zgodność danych.")
        writer.close()
        writer = None
        temporary.replace(output)
    finally:
        capture.release()
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
    return output
