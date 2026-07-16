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


if __name__ == "__main__":
    unittest.main()
