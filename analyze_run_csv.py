"""Command-line validation of extracted measurements against references."""

import argparse

from src.knee_angle_validation.io_utils import read_csv, write_csv
from src.knee_angle_validation.statistics import grouped_statistics
from src.knee_angle_validation.validation import merge_with_reference


def main() -> None:
    """Merge extracted and reference CSV files and save validation outputs."""
    parser = argparse.ArgumentParser(description="Validate MediaPipe knee angles.")
    parser.add_argument("--extracted", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--results", default="data/results/validation_results.csv")
    parser.add_argument("--summary", default="data/results/summary_statistics.csv")
    args = parser.parse_args()
    results = merge_with_reference(read_csv(args.extracted), read_csv(args.reference))
    summary = grouped_statistics(results, ["video_id", "person_id"])
    summary["angle_definition"] = "knee_flexion_2d_deg"
    write_csv(results, args.results)
    write_csv(summary, args.summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
