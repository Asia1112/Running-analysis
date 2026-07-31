"""Generate a compact HTML validation summary from statistics CSV."""

import argparse
from pathlib import Path

from src.knee_angle_validation.io_utils import read_csv


def main() -> None:
    """Convert summary statistics into a non-medical HTML research report."""
    parser = argparse.ArgumentParser(description="Generate an HTML validation summary.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", default="data/results/validation_summary.html")
    args = parser.parse_args()
    summary = read_csv(args.summary)
    html = (
        "<html><meta charset='utf-8'><title>Walidacja kąta kolana</title>"
        "<body><h1>Walidacja kąta stawu kolanowego</h1>"
        "<p>Pomiar 2D z nagrania bocznego z wykorzystaniem MediaPipe.</p>"
        + summary.to_html(index=False, float_format=lambda value: f"{value:.3f}")
        + "<p>System służy do oceny jakości pomiaru i nie jest narzędziem medycznym.</p></body></html>"
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
