"""Command-line plot of raw, smoothed, and optional reference knee angles."""

import argparse

import matplotlib.pyplot as plt

from src.knee_angle_validation.io_utils import read_csv
from src.knee_angle_validation.knee_angle import add_pixel_angle_columns


def main() -> None:
    """Plot canonical knee-angle columns from a CSV file."""
    parser = argparse.ArgumentParser(description="Plot knee-angle measurements.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    data = read_csv(args.csv)
    if "angle_geometry" not in data or not data["angle_geometry"].eq("pixel_2d_v1").all():
        data = add_pixel_angle_columns(data)
    plt.plot(data["time_s"], data["knee_flexion_deg"], alpha=0.45, label="raw flexion")
    if "knee_angle_smoothed" in data:
        plt.plot(data["time_s"], 180.0 - data["knee_angle_smoothed"], label="smoothed flexion")
    if "knee_angle_reference" in data:
        reference = add_pixel_angle_columns(data, reference=True)
        plt.scatter(data["time_s"], reference["knee_flexion_deg"], label="reference flexion")
    plt.xlabel("time [s]")
    plt.ylabel("2D knee flexion [deg] (extension = 0)")
    plt.legend()
    if args.output:
        plt.savefig(args.output, dpi=160, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
