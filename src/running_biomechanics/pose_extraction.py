import argparse
import csv
import os
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .config import (
    CORE_LANDMARKS,
    DEFAULT_CSV_PATH,
    DEFAULT_SOURCE,
    LANDMARKS_TO_SAVE,
    MODEL_PATH,
    POSE_CONNECTIONS,
    VISIBILITY_THRESHOLD,
)


@dataclass
class CropInfo:
    x0: int
    y0: int
    width: int
    height: int


def angle(a, b, c):
    """Angle ABC in degrees using 2D normalized coordinates."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    value = np.degrees(
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )

    value = abs(value)
    if value > 180:
        value = 360 - value

    return value


def apply_crop(frame, crop_mode):
    h, w = frame.shape[:2]

    if crop_mode == "left":
        return frame[:, : w // 2], CropInfo(0, 0, w // 2, h)
    if crop_mode == "right":
        x0 = w // 2
        return frame[:, x0:], CropInfo(x0, 0, w - x0, h)
    if crop_mode == "center":
        x0 = w // 4
        x1 = 3 * w // 4
        return frame[:, x0:x1], CropInfo(x0, 0, x1 - x0, h)

    return frame, CropInfo(0, 0, w, h)


def resize_for_detection(frame, resize_height):
    if resize_height <= 0:
        return frame

    h, w = frame.shape[:2]
    if h >= resize_height:
        return frame

    scale = resize_height / h
    return cv2.resize(frame, (int(w * scale), resize_height), interpolation=cv2.INTER_CUBIC)


def draw_landmarks(frame, landmarks, color=(0, 255, 0)):
    h, w = frame.shape[:2]

    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), 3, color, -1)

    for a, b in POSE_CONNECTIONS:
        ax = int(landmarks[a].x * w)
        ay = int(landmarks[a].y * h)
        bx = int(landmarks[b].x * w)
        by = int(landmarks[b].y * h)
        cv2.line(frame, (ax, ay), (bx, by), color, 2)


def landmark_xy(landmarks, idx, min_vis):
    lm = landmarks[idx]
    if lm.visibility < min_vis:
        return None

    return lm.x, lm.y


def write_empty_landmarks(row):
    for _ in LANDMARKS_TO_SAVE:
        row += [np.nan, np.nan, np.nan, np.nan]


def process_video(
    source,
    model_path=MODEL_PATH,
    csv_path=DEFAULT_CSV_PATH,
    crop_mode="none",
    resize_height=720,
    min_vis=VISIBILITY_THRESHOLD,
    preview=True,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Create models/ and put pose_landmarker_full.task there."
        )

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 30.0

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        header = ["frame_idx", "t_ms", "fps", "pose_found", "core_quality", "core_visible"]

        for name in LANDMARKS_TO_SAVE:
            header += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_vis"]

        header += ["left_knee_deg", "right_knee_deg"]
        writer.writerow(header)

        frame_idx = 0
        found_frames = 0
        good_core_frames = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            cropped, _ = apply_crop(frame, crop_mode)
            detection_frame = resize_for_detection(cropped, resize_height)
            t_ms = int((frame_idx / fps) * 1000)

            rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, t_ms)

            landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
            pose_found = landmarks is not None
            left_knee_deg = np.nan
            right_knee_deg = np.nan
            core_quality = np.nan
            core_visible = False

            row = [frame_idx, t_ms, fps, int(pose_found)]

            if pose_found:
                found_frames += 1
                core_vis = [landmarks[LANDMARKS_TO_SAVE[name]].visibility for name in CORE_LANDMARKS]
                core_quality = float(np.mean(core_vis))
                core_visible = bool(all(v >= min_vis for v in core_vis))
                if core_visible:
                    good_core_frames += 1

                row += [core_quality, int(core_visible)]

                Lh = landmark_xy(landmarks, 23, min_vis)
                Lk = landmark_xy(landmarks, 25, min_vis)
                La = landmark_xy(landmarks, 27, min_vis)
                Rh = landmark_xy(landmarks, 24, min_vis)
                Rk = landmark_xy(landmarks, 26, min_vis)
                Ra = landmark_xy(landmarks, 28, min_vis)

                if Lh is not None and Lk is not None and La is not None:
                    left_knee_deg = angle(Lh, Lk, La)

                if Rh is not None and Rk is not None and Ra is not None:
                    right_knee_deg = angle(Rh, Rk, Ra)

                for _, idx in LANDMARKS_TO_SAVE.items():
                    lm = landmarks[idx]
                    row += [lm.x, lm.y, lm.z, lm.visibility]

                if preview:
                    draw_landmarks(detection_frame, landmarks)
                    cv2.putText(
                        detection_frame,
                        f"quality={core_quality:.2f} core_ok={int(core_visible)}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0) if core_visible else (0, 180, 255),
                        2,
                    )
            else:
                row += [core_quality, int(core_visible)]
                write_empty_landmarks(row)

            row += [left_knee_deg, right_knee_deg]
            writer.writerow(row)

            if preview:
                cv2.imshow("MediaPipe Tasks Pose", detection_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

            frame_idx += 1

    cap.release()
    landmarker.close()
    if preview:
        cv2.destroyAllWindows()

    total = max(1, frame_idx)
    stats = {
        "csv_path": csv_path,
        "frames": frame_idx,
        "pose_found_pct": found_frames / total * 100,
        "good_core_pct": good_core_frames / total * 100,
    }

    print(f"Saved: {csv_path}")
    print(f"Pose found: {stats['pose_found_pct']:.1f}%")
    print(f"Good core landmarks: {stats['good_core_pct']:.1f}%")

    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect running pose landmarks and export biomechanical CSV data."
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Input video path.")
    parser.add_argument("--model", default=MODEL_PATH, help="MediaPipe pose model path.")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Output CSV path.")
    parser.add_argument(
        "--crop",
        choices=["none", "left", "right", "center"],
        default="none",
        help="Crop obvious screen recordings or split-screen videos before detection.",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=720,
        help="Upscale small videos to this height before detection. Use 0 to disable.",
    )
    parser.add_argument(
        "--min-vis",
        type=float,
        default=VISIBILITY_THRESHOLD,
        help="Minimum landmark visibility for angle and quality checks.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not open an OpenCV preview window. Useful for app/server processing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    process_video(
        source=args.source,
        model_path=args.model,
        csv_path=args.csv,
        crop_mode=args.crop,
        resize_height=args.resize_height,
        min_vis=args.min_vis,
        preview=not args.no_preview,
    )
