import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

from .config import DEFAULT_CSV_PATH
from .utils import moving_avg_nan, to_float


def clean_angle(values, lo=40, hi=185):
    values = np.asarray(values, dtype=float).copy()
    values[(values < lo) | (values > hi)] = np.nan
    return values


def load_knee_angles(csv_path):
    t = []
    left = []
    right = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["t_ms"]) / 1000.0)
            left.append(to_float(row["left_knee_deg"]))
            right.append(to_float(row["right_knee_deg"]))

    return np.array(t), np.array(left, dtype=float), np.array(right, dtype=float)


def plot_knee_angles(csv_path=DEFAULT_CSV_PATH):
    t, left, right = load_knee_angles(csv_path)
    left_s = moving_avg_nan(clean_angle(left), k=9)
    right_s = moving_avg_nan(clean_angle(right), k=9)

    plt.figure()
    plt.plot(t, left_s, label="Left knee (smoothed)")
    plt.plot(t, right_s, label="Right knee (smoothed)")
    plt.xlabel("time [s]")
    plt.ylabel("knee angle [deg]")
    plt.title("Knee angle vs time")
    plt.legend()
    plt.grid(True)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot knee angle curves from landmarks CSV.")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Input CSV path.")
    return parser.parse_args()


def main():
    args = parse_args()
    plot_knee_angles(args.csv)
