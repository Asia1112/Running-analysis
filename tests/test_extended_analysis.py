import csv
import math
import tempfile
import unittest
from pathlib import Path

from src.running_biomechanics.analysis import cadence_status, report_status
from src.running_biomechanics.extended_analysis import _build_reliability, analyze_extended_metrics


LANDMARKS = [
    "L_SHOULDER", "R_SHOULDER", "L_ELBOW", "R_ELBOW", "L_WRIST", "R_WRIST",
    "L_HIP", "R_HIP", "L_KNEE", "R_KNEE", "L_ANKLE", "R_ANKLE", "L_FOOT", "R_FOOT",
]


class ExtendedAnalysisTests(unittest.TestCase):
    def test_cadence_and_report_use_the_same_reliability_thresholds(self):
        result = {
            "cadence": 254.4,
            "cadence_left": 294.8,
            "cadence_right": 213.9,
            "lr_cadence_difference": 80.9,
            "events": 11,
            "coverage": 0.94,
            "core_visible_pct": 94.0,
            "knee_rom_difference_deg": 4.0,
            "trunk_lean_mean_deg": 5.8,
            "report_status": "low_confidence",
        }
        details = {
            "hips": {
                "available": True,
                "left": {"valid_pct": 95.0},
                "right": {"valid_pct": 95.0},
            },
            "trunk": {"valid_pct": 95.0},
            "technique": {"overstriding_available": True},
        }

        reliability = _build_reliability(result, {"tracking_error": False}, details)

        self.assertEqual(cadence_status(result), "low_confidence")
        self.assertEqual(report_status(result), "low_confidence")
        self.assertEqual(reliability["cadence"]["level"], "unreliable")
        self.assertIn("254.4", reliability["cadence"]["reason"])

    def test_large_left_right_difference_requires_review_everywhere(self):
        result = {
            "cadence": 180.0,
            "lr_cadence_difference": 20.0,
            "events": 10,
            "coverage": 0.92,
            "core_visible_pct": 92.0,
            "knee_rom_difference_deg": 4.0,
            "trunk_lean_mean_deg": 5.0,
        }

        self.assertEqual(cadence_status(result), "needs_review")
        self.assertEqual(report_status(result), "needs_review")

    def test_builds_quality_gated_joint_and_technique_metrics(self):
        fieldnames = ["frame_idx", "t_ms", "left_knee_deg", "right_knee_deg"]
        for name in LANDMARKS:
            fieldnames.extend([f"{name}_x", f"{name}_y", f"{name}_vis"])

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "landmarks.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for frame in range(120):
                    phase = 2 * math.pi * frame / 18
                    hip_x = 0.25 + frame * 0.002
                    row = {
                        "frame_idx": frame,
                        "t_ms": frame * (1000 / 30),
                        "left_knee_deg": 130 + 25 * math.sin(phase),
                        "right_knee_deg": 130 + 25 * math.sin(phase + math.pi),
                    }
                    points = {
                        "L_SHOULDER": (hip_x - 0.02, 0.30), "R_SHOULDER": (hip_x + 0.02, 0.30),
                        "L_ELBOW": (hip_x - 0.08, 0.43 + 0.03 * math.sin(phase)),
                        "R_ELBOW": (hip_x + 0.08, 0.43 - 0.03 * math.sin(phase)),
                        "L_WRIST": (hip_x - 0.10, 0.56 + 0.04 * math.sin(phase)),
                        "R_WRIST": (hip_x + 0.10, 0.56 - 0.04 * math.sin(phase)),
                        "L_HIP": (hip_x - 0.02, 0.52), "R_HIP": (hip_x + 0.02, 0.52),
                        "L_KNEE": (hip_x - 0.03 + 0.04 * math.sin(phase), 0.70),
                        "R_KNEE": (hip_x + 0.03 - 0.04 * math.sin(phase), 0.70),
                        "L_ANKLE": (hip_x + 0.07 * math.sin(phase), 0.87 + 0.03 * math.cos(phase)),
                        "R_ANKLE": (hip_x - 0.07 * math.sin(phase), 0.87 - 0.03 * math.cos(phase)),
                        "L_FOOT": (hip_x + 0.10 * math.sin(phase), 0.91 + 0.03 * math.cos(phase)),
                        "R_FOOT": (hip_x - 0.10 * math.sin(phase), 0.91 - 0.03 * math.cos(phase)),
                    }
                    for name, (x, y) in points.items():
                        row[f"{name}_x"] = x
                        row[f"{name}_y"] = y
                        row[f"{name}_vis"] = 0.95
                    writer.writerow(row)

            details = analyze_extended_metrics(
                csv_path,
                result={"core_visible_pct": 95.0},
                validation={"suspicious_frame_numbers": [12, 18]},
            )

        self.assertEqual(details["quality"]["suspicious_frame_count"], 2)
        self.assertGreater(details["quality"]["angle_stability_pct"], 0)
        self.assertTrue(details["hips"]["available"])
        self.assertIsNotNone(details["hips"]["rom_difference_deg"])
        self.assertTrue(details["arms"]["available"])
        self.assertIsNotNone(details["arms"]["symmetry_score"])
        self.assertTrue(details["technique"]["direction_available"])
        self.assertGreater(details["technique"]["contact_count"], 3)
        self.assertIn(details["quality"]["level"], {"good", "review", "unreliable"})
        self.assertIn("overall", details["reliability"])
        self.assertIsNotNone(details["trunk"]["min_deg"])
        self.assertIsNotNone(details["trunk"]["max_deg"])


if __name__ == "__main__":
    unittest.main()
