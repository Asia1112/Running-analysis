import unittest

import numpy as np

from src.knee_angle_validation.frame_quality import (
    assess_frame_quality,
    classify_visibility,
    detect_angle_jumps,
)


class FrameQualityTests(unittest.TestCase):
    def test_visibility_bins(self):
        self.assertEqual(classify_visibility(0.59), "<0.60")
        self.assertEqual(classify_visibility(0.60), "0.60-0.79")
        self.assertEqual(classify_visibility(0.85), "0.80-0.89")
        self.assertEqual(classify_visibility(0.90), ">=0.90")

    def test_jump_detection_and_reasons(self):
        np.testing.assert_array_equal(detect_angle_jumps([100, 105, 140], 20), [False, False, True])
        quality = assess_frame_quality(True, 0.55, 35.0, 2.0, 0.01)
        self.assertEqual(quality.status, "unreliable")
        self.assertGreaterEqual(len(quality.reasons), 2)


if __name__ == "__main__":
    unittest.main()
