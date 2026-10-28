"""MediaPipe extraction for one controlled side-view knee-angle measurement."""

from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime, timezone

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .frame_quality import add_quality_columns
from .knee_angle import add_pixel_angle_columns
from .smoothing import smooth_series
from .provenance import file_sha256, new_analysis_id, recording_metadata
from .experiment_metadata import normalize_experiment_metadata


SIDE_INDEXES = {
    "left": {"hip": 23, "knee": 25, "ankle": 27},
    "right": {"hip": 24, "knee": 26, "ankle": 28},
}


@dataclass(frozen=True)
class ExtractionResult:
    """Paths and summary values produced by one extraction run."""

    data: pd.DataFrame
    csv_path: Path
    frames: int
    pose_found_pct: float


def crop_frame(frame: np.ndarray, crop_mode: str) -> np.ndarray:
    """Apply a reproducible crop used both for extraction and annotation."""
    height, width = frame.shape[:2]
    if crop_mode == "left":
        return frame[:, : width // 2]
    if crop_mode == "right":
        return frame[:, width // 2:]
    if crop_mode == "center":
        return frame[:, width // 4: 3 * width // 4]
    return frame


def resize_frame(frame: np.ndarray, resize_height: int) -> np.ndarray:
    """Upscale short videos while preserving aspect ratio."""
    height, width = frame.shape[:2]
    if resize_height <= 0 or height >= resize_height:
        return frame
    scale = resize_height / height
    return cv2.resize(frame, (round(width * scale), resize_height), interpolation=cv2.INTER_CUBIC)


def read_analysis_frame(
    video_path: str | Path,
    frame_number: int,
    crop_mode: str = "none",
    resize_height: int = 720,
) -> np.ndarray:
    """Read the same unannotated frame geometry used during pose extraction."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame_number}")
        return resize_frame(crop_frame(frame, crop_mode), resize_height)
    finally:
        capture.release()


def _empty_point_record(point: str) -> dict[str, float]:
    """Return empty normalized, pixel, and visibility fields for one point."""
    return {
        f"{point}_x_norm": np.nan, f"{point}_y_norm": np.nan,
        f"{point}_x_px": np.nan, f"{point}_y_px": np.nan,
        f"{point}_visibility": np.nan,
    }


def _point_record(point: str, landmark: object, width: int, height: int) -> dict[str, float]:
    """Convert a MediaPipe landmark to the canonical point fields."""
    return {
        f"{point}_x_norm": float(landmark.x), f"{point}_y_norm": float(landmark.y),
        f"{point}_x_px": float(landmark.x * width), f"{point}_y_px": float(landmark.y * height),
        f"{point}_visibility": float(landmark.visibility),
    }


def process_video(
    source: str | Path,
    output_csv: str | Path,
    video_id: str,
    person_id: str,
    side: str,
    model_path: str | Path = "models/pose_landmarker_full.task",
    crop_mode: str = "none",
    resize_height: int = 720,
    smoothing_method: str = "median",
    smoothing_window: int = 5,
    analysis_id: str | None = None,
    experiment_metadata: dict | None = None,
) -> ExtractionResult:
    """Extract selected-side hip, knee, ankle points and write canonical CSV data."""
    if side not in SIDE_INDEXES:
        raise ValueError("side must be 'left' or 'right'")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"MediaPipe model not found: {model_path}")
    if Path(output_csv).exists() or Path(output_csv).with_suffix(".metadata.json").exists():
        raise ValueError("Wyniki już istnieją. Wybierz nową ścieżkę dla kolejnej analizy.")
    provenance = {"analysis_id": analysis_id or new_analysis_id(),
                  "source_sha256": file_sha256(source), "crop_mode": crop_mode,
                  "resize_height": resize_height, "video_id": str(video_id).strip(),
                  "person_id": str(person_id).strip(), "side": side}
    provenance = recording_metadata(pd.DataFrame([provenance]))
    study = normalize_experiment_metadata(
        experiment_metadata or {"original_filename": Path(source).name}, provenance["person_id"], side)
    if study["inclusion_status"] != "INCLUDED":
        raise ValueError("Nagranie nie ma statusu INCLUDED; ekstrakcja badawcza została zablokowana.")

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    rows: list[dict[str, object]] = []
    indexes = SIDE_INDEXES[side]
    landmarker = vision.PoseLandmarker.create_from_options(options)
    try:
        frame_number = 0
        while True:
            ok, original = capture.read()
            if not ok:
                break
            frame = resize_frame(crop_frame(original, crop_mode), resize_height)
            height, width = frame.shape[:2]
            timestamp_ms = round(frame_number / fps * 1000)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            detected = landmarker.detect_for_video(image, timestamp_ms)
            landmarks = detected.pose_landmarks[0] if detected.pose_landmarks else None
            row: dict[str, object] = {
                "video_id": video_id, "person_id": person_id, "frame_number": frame_number,
                "time_s": frame_number / fps, "fps": fps, "image_width": width,
                "image_height": height, "side": side, "pose_found": landmarks is not None,
            }
            if landmarks is None:
                for point in indexes:
                    row.update(_empty_point_record(point))
            else:
                for point, index in indexes.items():
                    row.update(_point_record(point, landmarks[index], width, height))
            rows.append(row)
            frame_number += 1
    finally:
        capture.release()
        landmarker.close()

    data = pd.DataFrame(rows)
    if data.empty:
        raise RuntimeError("The source video contains no readable frames")
    for key, value in provenance.items():
        data[key] = value
    for key, value in study.items():
        data[key] = value
    data = add_pixel_angle_columns(data)
    data["knee_angle_smoothed"] = smooth_series(
        data["knee_angle_raw"].to_numpy(), smoothing_method, smoothing_window
    )
    data["knee_flexion_smoothed_deg"] = 180.0 - data["knee_angle_smoothed"]
    data = add_quality_columns(data)
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        data.to_csv(stream, index=False)
    metadata = {**provenance, **study, "source_name": Path(source).name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model_sha256": file_sha256(model_path), "model_name": Path(model_path).name,
                "fps": fps, "frames": len(data),
                "image_width": int(data.image_width.iloc[0]), "image_height": int(data.image_height.iloc[0]),
                "smoothing_method": smoothing_method, "smoothing_window": smoothing_window}
    with csv_path.with_suffix(".metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    return ExtractionResult(
        data=data,
        csv_path=csv_path,
        frames=len(data),
        pose_found_pct=float(data["pose_found"].mean() * 100),
    )
