"""Recording identity and image geometry shared by extraction and annotation."""

import hashlib
import re
import math
from pathlib import Path
from uuid import uuid4

import pandas as pd


PROVENANCE_KEYS = ("analysis_id", "source_sha256", "crop_mode", "resize_height")
RANGE_KEYS = ("analysis_start_s", "analysis_end_s")
TIMING_KEYS = ("time_source", "analysis_first_frame", "analysis_stop_frame")


def validate_time_range(start: float, end: float | None) -> tuple[float, float | None]:
    start = float(start)
    end = float(end) if end is not None else None
    if not math.isfinite(start) or start < 0 or (end is not None and (not math.isfinite(end) or end <= start)):
        raise ValueError("Zakres analizy wymaga 0 ≤ początek < koniec oraz skończonych czasów.")
    return start, end


def frame_limits(start: float, end: float | None, fps: float) -> tuple[int, int | None]:
    """Half-open interval [start, end), using the source frame's nominal time."""
    start, end = validate_time_range(start, end)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Nieprawidłowe FPS; nie można wyznaczyć czasu klatek.")
    return math.ceil(start * fps - 1e-9), math.ceil(end * fps - 1e-9) if end is not None else None


def new_analysis_id() -> str:
    return uuid4().hex


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recording_metadata(data: pd.DataFrame) -> dict | None:
    """Require one complete recording identity; legacy CSV returns None."""
    present = set(PROVENANCE_KEYS).intersection(data.columns)
    if not present:
        return None
    required = (*PROVENANCE_KEYS, "video_id", "person_id", "side")
    if data.empty or not set(required).issubset(data.columns):
        raise ValueError("Niekompletne metadane pochodzenia danych.")
    result = {}
    for key in required:
        if data[key].isna().any() or data[key].nunique() != 1:
            raise ValueError(f"Dane muszą dotyczyć jednej wartości {key}.")
        result[key] = data[key].iloc[0]
    for key, length in (("analysis_id", 32), ("source_sha256", 64)):
        if not re.fullmatch(r"[a-f0-9]{%d}" % length, str(result[key])):
            raise ValueError(f"Nieprawidłowe pole {key}.")
    if result["crop_mode"] not in ("none", "left", "right", "center"):
        raise ValueError("Nieprawidłowe kadrowanie w metadanych.")
    height = float(result["resize_height"])
    if not 0 <= height <= 100000 or not height.is_integer():
        raise ValueError("Nieprawidłowa wysokość przeskalowania w metadanych.")
    result["resize_height"] = int(height)
    for key in ("video_id", "person_id"):
        if not str(result[key]).strip():
            raise ValueError(f"Puste pole {key}.")
        result[key] = str(result[key])
    if result["side"] not in ("left", "right"):
        raise ValueError("Nieprawidłowa strona ciała w metadanych.")
    if "time_source" in data:
        if data.time_source.isna().any() or not data.time_source.eq("ffmpeg_pts_v1").all():
            raise ValueError("Nieprawidłowa informacja o weryfikacji czasu filmu.")
        result["time_source"] = "ffmpeg_pts_v1"
        for key in TIMING_KEYS[1:]:
            if key not in data or data[key].isna().any() or data[key].nunique() != 1:
                raise ValueError(f"Niekompletny zakres klatek PTS: {key}.")
            number = float(data[key].iloc[0])
            if not math.isfinite(number) or number < 0 or not number.is_integer():
                raise ValueError(f"Nieprawidłowy numer klatki w metadanych: {key}.")
            result[key] = int(number)
        if result["analysis_stop_frame"] <= result["analysis_first_frame"]:
            raise ValueError("Pusty zakres klatek PTS.")
    if set(RANGE_KEYS).intersection(data.columns):
        if not set(RANGE_KEYS).issubset(data.columns):
            raise ValueError("Niekompletny zakres analizy w CSV.")
        for key in RANGE_KEYS:
            if data[key].isna().any() or data[key].nunique() != 1:
                raise ValueError(f"Analiza wymaga jednej wartości {key}.")
        start, end = validate_time_range(data.analysis_start_s.iloc[0], data.analysis_end_s.iloc[0])
        result.update(analysis_start_s=start, analysis_end_s=end)
    return result


def require_source_match(data: pd.DataFrame, source_sha256: str | None) -> dict:
    metadata = recording_metadata(data)
    if metadata is None:
        raise ValueError("CSV nie zawiera identyfikacji filmu. Wykonaj ponowną ekstrakcję, aby użyć obrazu i anotacji.")
    if metadata["source_sha256"] != source_sha256:
        raise ValueError("Film nie odpowiada danym CSV (inna suma SHA-256). Wgraj właściwe nagranie.")
    return metadata
