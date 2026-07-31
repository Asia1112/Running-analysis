import unittest

import pandas as pd

from src.knee_angle_validation.statistics import error_statistics
from src.knee_angle_validation.validation import merge_with_reference


class ValidationTests(unittest.TestCase):
    def test_join_errors_distances_and_statistics(self):
        extracted = pd.DataFrame([{
            "video_id": "v1", "person_id": "p1", "frame_number": 10, "side": "left",
            "knee_angle_raw": 95.0, "hip_x_px": 0.0, "hip_y_px": 0.0,
            "knee_x_px": 0.0, "knee_y_px": 10.0, "ankle_x_px": 10.0, "ankle_y_px": 10.0,
        }])
        reference = pd.DataFrame([{
            "video_id": "v1", "person_id": "p1", "frame_number": 10, "side": "left",
            "knee_angle_reference": 90.0, "hip_x_ref": 0.0, "hip_y_ref": 0.0,
            "knee_x_ref": 0.0, "knee_y_ref": 10.0, "ankle_x_ref": 10.0, "ankle_y_ref": 10.0,
        }])
        merged = merge_with_reference(extracted, reference)
        self.assertEqual(merged.loc[0, "error_signed"], 5.0)
        self.assertEqual(merged.loc[0, "hip_distance_px"], 0.0)
        stats = error_statistics(merged)
        self.assertEqual(stats["mae"], 5.0)
        self.assertEqual(stats["rmse"], 5.0)
        self.assertEqual(stats["bias"], 5.0)


if __name__ == "__main__":
    unittest.main()
