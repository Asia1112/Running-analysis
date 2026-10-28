"""Command-line plot of raw, smoothed, and optional reference knee angles."""

import argparse

import matplotlib.pyplot as plt

from src.knee_angle_validation.io_utils import read_csv
from src.knee_angle_validation.knee_angle import add_pixel_angle_columns
from src.knee_angle_validation.smoothing import smooth_series


def main() -> None:
    """Plot canonical knee-angle columns from a CSV file."""
    parser = argparse.ArgumentParser(description="Plot knee-angle measurements.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output")
    parser.add_argument("--smoothing", choices=["median", "moving_average", "savgol"],
                        help="Optional comparison filter recalculated from raw angles; raw is primary")
    parser.add_argument("--window", type=int, default=5, help="Positive odd frame window")
    args = parser.parse_args()
    data = read_csv(args.csv)
    if "angle_geometry" not in data or not data["angle_geometry"].eq("pixel_2d_v1").all():
        data = add_pixel_angle_columns(data)
    plt.plot(data["time_s"], data["knee_flexion_deg"], label="raw flexion (primary)")
    if args.smoothing:
        filtered = smooth_series(data["knee_flexion_deg"].to_numpy(), args.smoothing, args.window)
        plt.plot(data["time_s"], filtered, alpha=0.65, label=f"{args.smoothing} (optional)")
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
