"""Command-line entry point for canonical selected-side pose extraction."""

import argparse
from pathlib import Path

from src.knee_angle_validation.pose_extraction import process_video


def main() -> None:
    """Parse extraction parameters and process one side-view video."""
    parser = argparse.ArgumentParser(description="Extract one selected knee from a side-view video.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--csv", default="data/extracted/extracted_landmarks.csv")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--person-id", required=True)
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--crop", choices=["none", "left", "right", "center"], default="none")
    parser.add_argument("--model", default="models/pose_landmarker_full.task")
    args = parser.parse_args()
    result = process_video(
        args.source, args.csv, args.video_id, args.person_id, args.side,
        model_path=args.model, crop_mode=args.crop,
    )
    print(f"Saved {result.frames} frames to {Path(result.csv_path)}")
    print(f"Pose found: {result.pose_found_pct:.1f}%")


if __name__ == "__main__":
    main()
