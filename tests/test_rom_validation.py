import csv
import tempfile
import unittest
from pathlib import Path

from src.running_biomechanics.analysis import validate_knee_rom


class KneeRomValidationTests(unittest.TestCase):
    def test_detects_angle_jumps_low_visibility_and_tracking_error(self):
        fieldnames = [
            "frame_idx",
            "t_ms",
            "left_knee_deg",
            "right_knee_deg",
            "L_KNEE_vis",
            "L_ANKLE_vis",
            "R_KNEE_vis",
            "R_ANKLE_vis",
        ]
        left_angles = [100, 105, 170, 110, 100]
        right_angles = [120, 125, 130, 125, 120]

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "landmarks.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for frame, (left_angle, right_angle) in enumerate(
                    zip(left_angles, right_angles)
                ):
                    writer.writerow(
                        {
                            "frame_idx": frame,
                            "t_ms": frame * 33,
                            "left_knee_deg": left_angle,
                            "right_knee_deg": right_angle,
                            "L_KNEE_vis": 0.5 if frame == 4 else 0.95,
                            "L_ANKLE_vis": 0.95,
                            "R_KNEE_vis": 0.95,
                            "R_ANKLE_vis": 0.95,
                        }
                    )

            validation = validate_knee_rom(csv_path)

        left = validation["sides"]["left"]
        right = validation["sides"]["right"]

        self.assertEqual(left["jump_count"], 2)
        self.assertEqual(left["low_visibility_count"], 1)
        self.assertEqual(left["raw_extrema"]["min"]["frame"], 0)
        self.assertEqual(left["raw_extrema"]["max"]["frame"], 2)
        self.assertGreater(left["raw_rom_deg"], left["smoothed_rom_deg"])
        self.assertTrue(left["tracking_error"])
        self.assertTrue(validation["tracking_error"])

        self.assertEqual(right["jump_count"], 0)
        self.assertFalse(right["tracking_error"])

        suspicious_left_frames = {
            item["frame"]
            for item in validation["suspicious_frames"]
            if item["side"] == "left"
        }
        self.assertEqual(suspicious_left_frames, {2, 3, 4})

    def test_detects_low_hip_visibility_and_landmark_outside_image(self):
        fieldnames = [
            "frame_idx", "t_ms", "left_knee_deg", "right_knee_deg",
            "L_HIP_x", "L_HIP_y", "L_HIP_vis", "L_KNEE_x", "L_KNEE_y",
            "L_KNEE_vis", "L_ANKLE_x", "L_ANKLE_y", "L_ANKLE_vis",
            "R_HIP_x", "R_HIP_y", "R_HIP_vis", "R_KNEE_x", "R_KNEE_y",
            "R_KNEE_vis", "R_ANKLE_x", "R_ANKLE_y", "R_ANKLE_vis",
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "landmarks.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for frame in range(4):
                    row = {
                        "frame_idx": frame, "t_ms": frame * 33,
                        "left_knee_deg": 120 + frame, "right_knee_deg": 125 + frame,
                    }
                    for side in ("L", "R"):
                        for point in ("HIP", "KNEE", "ANKLE"):
                            row[f"{side}_{point}_x"] = 0.5
                            row[f"{side}_{point}_y"] = 0.5
                            row[f"{side}_{point}_vis"] = 0.95
                    if frame == 1:
                        row["L_HIP_vis"] = 0.4
                    if frame == 2:
                        row["R_ANKLE_x"] = 1.1
                    writer.writerow(row)

            validation = validate_knee_rom(csv_path)

        reasons_by_frame = {
            item["frame"]: item["reasons"] for item in validation["suspicious_frames"]
        }
        self.assertTrue(any("visibility biodra" in reason for reason in reasons_by_frame[1]))
        self.assertIn("punkt poza obrazem", reasons_by_frame[2])


if __name__ == "__main__":
    unittest.main()
