import math
import unittest

from src.knee_angle_validation.knee_angle import calculate_knee_angle


class KneeAngleTests(unittest.TestCase):
    def test_known_angles(self):
        self.assertAlmostEqual(calculate_knee_angle((1, 0), (0, 0), (0, 1)), 90.0)
        self.assertAlmostEqual(calculate_knee_angle((-1, 0), (0, 0), (1, 0)), 180.0)
        self.assertAlmostEqual(calculate_knee_angle((1, 0), (0, 0), (1, 1)), 45.0)

    def test_missing_and_zero_length_points_return_nan(self):
        self.assertTrue(math.isnan(calculate_knee_angle(None, (0, 0), (1, 0))))
        self.assertTrue(math.isnan(calculate_knee_angle((0, 0), (0, 0), (1, 0))))


if __name__ == "__main__":
    unittest.main()
