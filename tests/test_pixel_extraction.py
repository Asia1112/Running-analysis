import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.knee_angle_validation.pose_extraction import process_video
from src.knee_angle_validation.reference_annotation import build_reference_annotation
from src.knee_angle_validation.validation import merge_with_reference


class PixelExtractionTests(unittest.TestCase):
    def test_extraction_and_reference_agree_for_both_sides_on_1080p_frame(self):
        for side, indexes in (("left", (23, 25, 27)), ("right", (24, 26, 28))):
            with self.subTest(side=side), tempfile.TemporaryDirectory() as directory:
                landmarks = [SimpleNamespace(x=0., y=0., visibility=1.) for _ in range(33)]
                points = ((400., 400.), (600., 600.), (800., 400.))
                for index, (x, y) in zip(indexes, points):
                    landmarks[index] = SimpleNamespace(x=x/1920, y=y/1080, visibility=1.)
                detector = MagicMock()
                detector.detect_for_video.side_effect = [SimpleNamespace(pose_landmarks=[landmarks]),
                                                         SimpleNamespace(pose_landmarks=[])]
                capture = MagicMock()
                capture.isOpened.return_value = True
                capture.get.return_value = 60.
                frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                capture.read.side_effect = [(True, frame), (True, frame), (False, None)]
                model = Path(directory) / "model.task"
                model.touch()
                output = Path(directory) / "extracted.csv"
                with patch('src.knee_angle_validation.pose_extraction.cv2.VideoCapture', return_value=capture), \
                     patch('src.knee_angle_validation.pose_extraction.vision.PoseLandmarker.create_from_options', return_value=detector), \
                     warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    result = process_video('source.mov', output, 'v1', 'p1', side,
                                           model_path=model, resize_height=0)
                saved = pd.read_csv(output)
                self.assertAlmostEqual(saved.loc[0, "knee_internal_angle_deg"], 90.)
                self.assertAlmostEqual(saved.loc[0, "knee_flexion_deg"], 90.)
                self.assertTrue(saved.loc[1, ["knee_internal_angle_deg", "knee_flexion_deg"]].isna().all())
                reference = build_reference_annotation(result.data.iloc[0], points, 'a1', 's1')
                self.assertAlmostEqual(reference["knee_flexion_deg"], 90.)
                merged = merge_with_reference(result.data, pd.DataFrame([reference]))
                self.assertAlmostEqual(merged.loc[0, "error_signed"], 0.)
                self.assertAlmostEqual(merged.loc[0, "error_internal_signed"], 0.)


if __name__ == '__main__':
    unittest.main()
