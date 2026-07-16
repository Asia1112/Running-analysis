import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import imageio_ffmpeg
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
    ROM_JUMP_THRESHOLD_DEG,
    ROM_VISIBILITY_THRESHOLD,
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


def format_debug_value(value, digits=1):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def debug_side_state(landmarks, side, angle_deg, previous_angle_deg):
    indexes = {
        "left": (23, 25, 27),
        "right": (24, 26, 28),
    }
    hip_idx, knee_idx, ankle_idx = indexes[side]
    hip_visibility = landmarks[hip_idx].visibility if landmarks else np.nan
    knee_visibility = landmarks[knee_idx].visibility if landmarks else np.nan
    ankle_visibility = landmarks[ankle_idx].visibility if landmarks else np.nan
    angle_change = (
        abs(angle_deg - previous_angle_deg)
        if np.isfinite(angle_deg) and np.isfinite(previous_angle_deg)
        else np.nan
    )

    reasons = []
    if np.isfinite(angle_change) and angle_change > ROM_JUMP_THRESHOLD_DEG:
        reasons.append("angle jump")
    if not np.isfinite(knee_visibility) or knee_visibility < ROM_VISIBILITY_THRESHOLD:
        reasons.append("low knee visibility")
    if not np.isfinite(ankle_visibility) or ankle_visibility < ROM_VISIBILITY_THRESHOLD:
        reasons.append("low ankle visibility")
    if not np.isfinite(hip_visibility) or hip_visibility < ROM_VISIBILITY_THRESHOLD:
        reasons.append("low hip visibility")
    outside_indexes = []
    if landmarks:
        outside_indexes = [
            idx
            for idx in (hip_idx, knee_idx, ankle_idx)
            if landmarks[idx].x < 0
            or landmarks[idx].x > 1
            or landmarks[idx].y < 0
            or landmarks[idx].y > 1
        ]
        if outside_indexes:
            reasons.append("landmark outside image")

    return {
        "side": side,
        "angle_deg": angle_deg,
        "angle_change_deg": angle_change,
        "hip_visibility": float(hip_visibility),
        "knee_visibility": float(knee_visibility),
        "ankle_visibility": float(ankle_visibility),
        "reasons": reasons,
        "hip_idx": hip_idx,
        "knee_idx": knee_idx,
        "ankle_idx": ankle_idx,
        "outside_indexes": outside_indexes,
    }


def save_debug_frame(frame, landmarks, frame_idx, side_states, output_path):
    annotated = frame.copy()
    if landmarks:
        draw_landmarks(annotated, landmarks)
        h, w = annotated.shape[:2]
        for state in side_states:
            if not state["reasons"]:
                continue
            for idx in (state["hip_idx"], state["knee_idx"], state["ankle_idx"]):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(annotated, (x, y), 8, (0, 0, 255), 2)

    lines = [f"Frame {frame_idx}"]
    for state in side_states:
        if not state["reasons"]:
            continue
        side_label = "L" if state["side"] == "left" else "R"
        lines.append(
            f"{side_label} angle={format_debug_value(state['angle_deg'])} deg "
            f"delta={format_debug_value(state['angle_change_deg'])} deg"
        )
        lines.append(
            f"{side_label} hip_vis={format_debug_value(state['hip_visibility'], 2)} "
            f"knee_vis={format_debug_value(state['knee_visibility'], 2)} "
            f"ankle_vis={format_debug_value(state['ankle_visibility'], 2)}; "
            f"{', '.join(state['reasons'])}"
        )

    overlay_height = 18 + 25 * len(lines)
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], overlay_height), (0, 0, 0), -1)
    for line_idx, line in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (12, 25 + line_idx * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Could not save debug frame: {output_path}")


def annotate_video_frame(frame, landmarks, frame_idx, side_states):
    annotated = frame.copy()
    if landmarks:
        draw_landmarks(annotated, landmarks)

    left = next(state for state in side_states if state["side"] == "left")
    right = next(state for state in side_states if state["side"] == "right")
    lines = [
        f"Frame {frame_idx}",
        f"L knee={format_debug_value(left['angle_deg'])} deg  "
        f"R knee={format_debug_value(right['angle_deg'])} deg",
    ]

    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 68), (0, 0, 0), -1)
    for line_idx, line in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (12, 27 + line_idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    height, width = annotated.shape[:2]
    return annotated[: height - height % 2, : width - width % 2]


def process_video(
    source,
    model_path=MODEL_PATH,
    csv_path=DEFAULT_CSV_PATH,
    crop_mode="none",
    resize_height=720,
    min_vis=VISIBILITY_THRESHOLD,
    preview=True,
    debug_frames_dir="debug_frames",
    annotated_video_path=None,
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

    debug_run_dir = None
    if debug_frames_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_run_dir = Path(debug_frames_dir) / f"{Path(source).stem}_{timestamp}"
        debug_run_dir.mkdir(parents=True, exist_ok=True)

    annotated_path = Path(annotated_video_path) if annotated_video_path else None
    if annotated_path is not None:
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_writer = None

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
        debug_frames_count = 0
        previous_angles = {"left": np.nan, "right": np.nan}

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

            side_states = [
                debug_side_state(landmarks, "left", left_knee_deg, previous_angles["left"]),
                debug_side_state(landmarks, "right", right_knee_deg, previous_angles["right"]),
            ]
            suspicious_sides = [state for state in side_states if state["reasons"]]
            if debug_run_dir is not None and suspicious_sides:
                side_suffix = "".join(
                    "L" if state["side"] == "left" else "R" for state in suspicious_sides
                )
                output_path = debug_run_dir / f"frame_{frame_idx:06d}_{side_suffix}.jpg"
                save_debug_frame(
                    detection_frame,
                    landmarks,
                    frame_idx,
                    side_states,
                    output_path,
                )
                debug_frames_count += 1

            if annotated_path is not None:
                annotated_frame = annotate_video_frame(
                    detection_frame,
                    landmarks,
                    frame_idx,
                    side_states,
                )
                if annotated_writer is None:
                    height, width = annotated_frame.shape[:2]
                    annotated_writer = imageio_ffmpeg.write_frames(
                        str(annotated_path),
                        (width, height),
                        pix_fmt_in="rgb24",
                        pix_fmt_out="yuv420p",
                        fps=fps,
                        codec="libx264",
                        quality=7,
                        macro_block_size=1,
                        ffmpeg_log_level="warning",
                        output_params=["-movflags", "+faststart"],
                    )
                    annotated_writer.send(None)
                rgb_annotated = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                annotated_writer.send(np.ascontiguousarray(rgb_annotated))

            previous_angles["left"] = left_knee_deg
            previous_angles["right"] = right_knee_deg

            if preview:
                cv2.imshow("MediaPipe Tasks Pose", detection_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

            frame_idx += 1

    if annotated_writer is not None:
        annotated_writer.close()

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
        "debug_frames_dir": str(debug_run_dir) if debug_run_dir is not None else None,
        "debug_frames_count": debug_frames_count,
        "annotated_video_path": (
            str(annotated_path) if annotated_writer is not None else None
        ),
    }

    print(f"Saved: {csv_path}")
    print(f"Pose found: {stats['pose_found_pct']:.1f}%")
    print(f"Good core landmarks: {stats['good_core_pct']:.1f}%")
    print(f"Debug frames: {stats['debug_frames_count']} ({stats['debug_frames_dir']})")
    print(f"Annotated video: {stats['annotated_video_path']}")

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
    parser.add_argument(
        "--debug-frames-dir",
        default="debug_frames",
        help="Directory for annotated suspicious frames.",
    )
    parser.add_argument(
        "--annotated-video",
        help="Optional H.264 MP4 output with the detected skeleton and knee angles.",
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
        debug_frames_dir=args.debug_frames_dir,
        annotated_video_path=args.annotated_video,
    )
