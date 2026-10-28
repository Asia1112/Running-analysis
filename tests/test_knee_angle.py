import math
import unittest

import numpy as np
import pandas as pd

from src.knee_angle_validation.knee_angle import add_pixel_angle_columns, calculate_knee_angle


class KneeAngleTests(unittest.TestCase):
    def test_known_angles(self):
        self.assertAlmostEqual(calculate_knee_angle((1, 0), (0, 0), (0, 1)), 90.0)
        self.assertAlmostEqual(calculate_knee_angle((-1, 0), (0, 0), (1, 0)), 180.0)
        self.assertAlmostEqual(calculate_knee_angle((1, 0), (0, 0), (1, 1)), 45.0)

    def test_missing_and_zero_length_points_return_nan(self):
        self.assertTrue(math.isnan(calculate_knee_angle(None, (0, 0), (1, 0))))
        self.assertTrue(math.isnan(calculate_knee_angle((0, 0), (0, 0), (1, 0))))

    def test_pixel_angles_are_not_distorted_by_non_square_image(self):
        row = {f"{point}_{axis}_px": value
               for point, coordinates in (("hip", (0., 0.)), ("knee", (100., 100.)), ("ankle", (200., 0.)))
               for axis, value in zip(("x", "y"), coordinates)}
        # Simulate old CSV values: the converter must ignore them, not trust them.
        source = pd.DataFrame([{**row, "knee_angle_raw": 58.7, "knee_angle_smoothed": 60.}])
        corrected = add_pixel_angle_columns(source)
        self.assertAlmostEqual(corrected.loc[0, "knee_internal_angle_deg"], 90.)
        self.assertAlmostEqual(corrected.loc[0, "knee_flexion_deg"], 90.)
        self.assertAlmostEqual(corrected.loc[0, "knee_angle_raw"], 90.)
        self.assertEqual(source.loc[0, "knee_angle_raw"], 58.7)
        self.assertNotIn("knee_angle_smoothed", corrected)
        normalized = calculate_knee_angle((0., 0.), (100/1920, 100/1080), (200/1920, 0.))
        self.assertGreater(abs(normalized - 90.), 30.)

    def test_extension_flexion_and_missing_pixel_points(self):
        source = pd.DataFrame([
            {"hip_x_px": 0., "hip_y_px": 0., "knee_x_px": 0., "knee_y_px": 100.,
             "ankle_x_px": 0., "ankle_y_px": 200.},
            {"hip_x_px": np.nan, "hip_y_px": 0., "knee_x_px": 0., "knee_y_px": 100.,
             "ankle_x_px": 0., "ankle_y_px": 200.},
        ])
        result = add_pixel_angle_columns(source)
        self.assertAlmostEqual(result.loc[0, "knee_internal_angle_deg"], 180.)
        self.assertAlmostEqual(result.loc[0, "knee_flexion_deg"], 0.)
        self.assertTrue(result.loc[1, ["knee_internal_angle_deg", "knee_flexion_deg"]].isna().all())

    def test_translation_and_uniform_scaling_preserve_angle(self):
        points = np.array([[100., 100.], [200., 200.], [300., 100.]])
        self.assertAlmostEqual(calculate_knee_angle(*points), 90.)
        self.assertAlmostEqual(calculate_knee_angle(*(points * 3 + [50., 70.])), 90.)
        self.assertTrue(math.isnan(calculate_knee_angle((0., 0.), (1e-9, 0.), (1., 1.))))


if __name__ == "__main__":
    unittest.main()
