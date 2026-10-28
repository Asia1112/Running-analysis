"""Command-line validation of extracted measurements against references."""

import argparse
from pathlib import Path

from src.knee_angle_validation.io_utils import read_csv, write_csv
from src.knee_angle_validation.annotation_storage import read_reference_annotations, select_reference_session
from src.knee_angle_validation.statistics import grouped_reference_statistics, reference_quality_summary, grouped_availability, recording_statistics
from src.knee_angle_validation.validation import compare_with_inventory
from src.knee_angle_validation.provenance import recording_metadata, new_analysis_id
from src.knee_angle_validation.experiment_metadata import experiment_values


def main() -> None:
    """Merge extracted and reference CSV files and save validation outputs."""
    parser = argparse.ArgumentParser(description="Validate MediaPipe knee angles.")
    parser.add_argument("--extracted", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--annotator-id", help="Reference annotator ID")
    parser.add_argument("--session", help="Reference annotation session")
    parser.add_argument("--selected-frames", help="Planned reference frames; defaults to adjacent selected_reference_frames.csv when present")
    parser.add_argument("--results", help="Optional explicit validation output path")
    parser.add_argument("--summary", help="Optional explicit summary output path")
    args = parser.parse_args()
    try:
        extracted = read_csv(args.extracted)
        reference = select_reference_session(read_reference_annotations(args.reference), args.annotator_id, args.session)
        selected_path = Path(args.selected_frames) if args.selected_frames else Path(args.extracted).with_name("selected_reference_frames.csv")
        selected = read_csv(selected_path) if args.selected_frames or selected_path.exists() else None
        results, inventory = compare_with_inventory(extracted, reference, selected_frames=selected,
                                       annotator_id=reference.annotator_id.iloc[0],
                                       session=reference.annotation_session.iloc[0])
    except (ValueError, OSError) as error:
        parser.error(str(error))
    summary = grouped_reference_statistics(inventory, ["video_id", "person_id", "speed_kmh"])
    summary["angle_definition"] = "knee_flexion_2d_deg"
    summary["annotator_id"] = reference.annotator_id.iloc[0]
    summary["annotation_session"] = reference.annotation_session.iloc[0]
    metadata = recording_metadata(extracted)
    if metadata:
        for key, value in metadata.items():
            summary[key] = value
    study = experiment_values(extracted)
    if study:
        for key, value in study.items():
            summary[key] = value
    analysis_id = metadata["analysis_id"] if metadata else new_analysis_id()
    folder = Path("data/results/analyses") / analysis_id / "validations" / new_analysis_id()
    result_path = Path(args.results) if args.results else folder / "validation_results.csv"
    summary_path = Path(args.summary) if args.summary else folder / "summary_statistics.csv"
    quality_path = summary_path.with_name(summary_path.stem + "_reference_quality.csv")
    inventory_path = summary_path.with_name(summary_path.stem + "_frame_inventory.csv")
    availability_path = summary_path.with_name(summary_path.stem + "_availability.csv")
    recording_path = summary_path.with_name(summary_path.stem + "_recordings.csv")
    paths = [result_path, summary_path, quality_path, inventory_path, availability_path, recording_path]
    if len({path.resolve() for path in paths}) != len(paths) or any(path.exists() for path in paths):
        parser.error("Wybierz osobne, nowe ścieżki wyników; istniejące pliki nie są zastępowane.")
    write_csv(results, result_path)
    write_csv(summary, summary_path)
    write_csv(reference_quality_summary(inventory), quality_path)
    write_csv(inventory, inventory_path)
    write_csv(grouped_availability(inventory, ["video_id", "person_id", "speed_kmh"]), availability_path)
    write_csv(recording_statistics(inventory), recording_path)
    print(f"Validation: {result_path}")
    print(f"Summary: {summary_path}")
    print(f"Reference quality: {quality_path}")
    print(f"Frame inventory: {inventory_path}")
    if selected is None:
        print("No planned reference selection: completeness of the intended trial is unknown.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
