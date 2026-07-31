import unittest

import numpy as np

from src.knee_angle_validation.smoothing import median_filter, moving_average, savitzky_golay


class SmoothingTests(unittest.TestCase):
    def test_filters_preserve_length_and_reduce_impulse(self):
        values = np.array([100.0, 100.0, 160.0, 100.0, 100.0])
        self.assertEqual(len(moving_average(values, 3)), len(values))
        self.assertEqual(median_filter(values, 3)[2], 100.0)
        self.assertEqual(len(savitzky_golay(values, 5)), len(values))


if __name__ == "__main__":
    unittest.main()
