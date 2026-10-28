"""Command-line validation of extracted measurements against references."""

import argparse
from pathlib import Path

from src.knee_angle_validation.io_utils import read_csv, write_csv
from src.knee_angle_validation.annotation_storage import read_reference_annotations, select_reference_session
from src.knee_angle_validation.statistics import grouped_statistics
from src.knee_angle_validation.validation import merge_with_reference
from src.knee_angle_validation.provenance import recording_metadata, new_analysis_id


def main() -> None:
    """Merge extracted and reference CSV files and save validation outputs."""
    parser = argparse.ArgumentParser(description="Validate MediaPipe knee angles.")
    parser.add_argument("--extracted", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--annotator-id", help="Reference annotator ID")
    parser.add_argument("--session", help="Reference annotation session")
    parser.add_argument("--results", help="Optional explicit validation output path")
    parser.add_argument("--summary", help="Optional explicit summary output path")
    args = parser.parse_args()
    try:
        extracted = read_csv(args.extracted)
        reference = select_reference_session(read_reference_annotations(args.reference), args.annotator_id, args.session)
        results = merge_with_reference(extracted, reference,
                                       annotator_id=reference.annotator_id.iloc[0],
                                       session=reference.annotation_session.iloc[0])
    except ValueError as error:
        parser.error(str(error))
    summary = grouped_statistics(results, ["video_id", "person_id"])
    summary["angle_definition"] = "knee_flexion_2d_deg"
    summary["annotator_id"] = reference.annotator_id.iloc[0]
    summary["annotation_session"] = reference.annotation_session.iloc[0]
    metadata = recording_metadata(extracted)
    if metadata:
        for key, value in metadata.items():
            summary[key] = value
    analysis_id = metadata["analysis_id"] if metadata else new_analysis_id()
    folder = Path("data/results/analyses") / analysis_id / "validations" / new_analysis_id()
    result_path = Path(args.results) if args.results else folder / "validation_results.csv"
    summary_path = Path(args.summary) if args.summary else folder / "summary_statistics.csv"
    if result_path.resolve() == summary_path.resolve() or result_path.exists() or summary_path.exists():
        parser.error("Wybierz osobne, nowe ścieżki wyników; istniejące pliki nie są zastępowane.")
    write_csv(results, result_path)
    write_csv(summary, summary_path)
    print(f"Validation: {result_path}")
    print(f"Summary: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
