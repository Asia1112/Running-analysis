"""Command-line entry point for canonical selected-side pose extraction."""

import argparse
from pathlib import Path

from src.knee_angle_validation.pose_extraction import process_video
from src.knee_angle_validation.provenance import new_analysis_id
from src.knee_angle_validation.experiment_metadata import (
    DEFAULT_METADATA_PATH, load_video_metadata, metadata_for_video, normalize_experiment_metadata,
)


def main() -> None:
    """Parse extraction parameters and process one side-view video."""
    parser = argparse.ArgumentParser(description="Extract one selected knee from a side-view video.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--csv", help="New output CSV path; existing files cannot be overwritten")
    parser.add_argument("--video-id", help="Defaults to original filename without extension")
    parser.add_argument("--person-id", help="Required for recordings absent from the metadata table")
    parser.add_argument("--side", choices=["left", "right"])
    parser.add_argument("--speed-kmh", type=float, help="Optional speed for recordings outside the table")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--original-filename", help="Original filename if --source is a hash-named copy")
    parser.add_argument("--crop", choices=["none", "left", "right", "center"], default="none")
    parser.add_argument("--model", default="models/pose_landmarker_full.task")
    args = parser.parse_args()
    original = args.original_filename or Path(args.source).name
    try:
        study = metadata_for_video(original, load_video_metadata(args.metadata))
        person = args.person_id if args.person_id is not None else (study["person_id"] if study else "")
        side = args.side if args.side is not None else (study["side"] if study else "")
        if study and args.speed_kmh is not None and args.speed_kmh != study["speed_kmh"]:
            raise ValueError("Prędkość nie zgadza się z tabelą metadanych.")
        experiment = normalize_experiment_metadata(study or {"original_filename": original, "speed_kmh": args.speed_kmh}, person, side)
        if experiment["inclusion_status"] != "INCLUDED":
            raise ValueError("Nagranie nie ma statusu INCLUDED.")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    analysis_id = new_analysis_id()
    result = process_video(
        args.source, args.csv or Path("data/results/analyses") / analysis_id / "extracted_landmarks.csv",
        args.video_id or Path(original).stem, person, side,
        model_path=args.model, crop_mode=args.crop, analysis_id=analysis_id, experiment_metadata=experiment,
    )
    print(f"Saved {result.frames} frames to {Path(result.csv_path)}")
    print(f"Pose found: {result.pose_found_pct:.1f}%")


if __name__ == "__main__":
    main()
