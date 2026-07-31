"""Command-line plot of raw, smoothed, and optional reference knee angles."""

import argparse

import matplotlib.pyplot as plt

from src.knee_angle_validation.io_utils import read_csv


def main() -> None:
    """Plot canonical knee-angle columns from a CSV file."""
    parser = argparse.ArgumentParser(description="Plot knee-angle measurements.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    data = read_csv(args.csv)
    plt.plot(data["time_s"], data["knee_angle_raw"], alpha=0.45, label="raw")
    if "knee_angle_smoothed" in data:
        plt.plot(data["time_s"], data["knee_angle_smoothed"], label="smoothed")
    if "knee_angle_reference" in data:
        plt.scatter(data["time_s"], data["knee_angle_reference"], label="reference")
    plt.xlabel("time [s]")
    plt.ylabel("knee angle [deg]")
    plt.legend()
    if args.output:
        plt.savefig(args.output, dpi=160, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
