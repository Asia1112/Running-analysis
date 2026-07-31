"""MediaPipe extraction for one controlled side-view knee-angle measurement."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .frame_quality import add_quality_columns
from .io_utils import write_csv
from .knee_angle import calculate_knee_angle
from .smoothing import smooth_series


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
) -> ExtractionResult:
    """Extract selected-side hip, knee, ankle points and write canonical CSV data."""
    if side not in SIDE_INDEXES:
        raise ValueError("side must be 'left' or 'right'")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"MediaPipe model not found: {model_path}")

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
                row["knee_angle_raw"] = np.nan
            else:
                for point, index in indexes.items():
                    row.update(_point_record(point, landmarks[index], width, height))
                row["knee_angle_raw"] = calculate_knee_angle(
                    (row["hip_x_norm"], row["hip_y_norm"]),
                    (row["knee_x_norm"], row["knee_y_norm"]),
                    (row["ankle_x_norm"], row["ankle_y_norm"]),
                )
            rows.append(row)
            frame_number += 1
    finally:
        capture.release()
        landmarker.close()

    data = pd.DataFrame(rows)
    if data.empty:
        raise RuntimeError("The source video contains no readable frames")
    data["knee_angle_smoothed"] = smooth_series(
        data["knee_angle_raw"].to_numpy(), smoothing_method, smoothing_window
    )
    data = add_quality_columns(data)
    csv_path = write_csv(data, output_csv)
    return ExtractionResult(
        data=data,
        csv_path=csv_path,
        frames=len(data),
        pose_found_pct=float(data["pose_found"].mean() * 100),
    )
