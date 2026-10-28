"""CFR verification from real decoded PTS, including deliberate VFR input."""

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import imageio_ffmpeg
import numpy as np

from src.knee_angle_validation.video_timing import audit_video_timing, verify_cfr, inspect_timing, TIME_SOURCE
from src.knee_angle_validation.pose_extraction import process_video, read_analysis_frame


class VideoTimingTests(unittest.TestCase):
    def test_millisecond_collision_is_rejected_without_inventing_a_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, model = root / 'video.mov', root / 'model.task'
            source.write_bytes(b'video')
            model.touch()
            capture = MagicMock()
            capture.get.return_value = 60.
            capture.read.return_value = (True, np.zeros((160, 160, 3), dtype=np.uint8))
            detector = MagicMock()
            detector.detect_for_video.return_value.pose_landmarks = []
            timing = inspect_timing([0., 0.0004, 0.017], 1e-6, 60.)
            with patch('src.knee_angle_validation.pose_extraction.cv2.VideoCapture', return_value=capture), \
                 patch('src.knee_angle_validation.pose_extraction.audit_video_timing', return_value=timing), \
                 patch('src.knee_angle_validation.pose_extraction.vision.PoseLandmarker.create_from_options', return_value=detector):
                with self.assertRaisesRegex(ValueError, 'milisekund'):
                    process_video(source, root / 'output.csv', 'v1', 'p1', 'right', model_path=model)
                self.assertEqual(detector.detect_for_video.call_count, 1)
            self.assertFalse((root / 'output.csv').exists())

    def test_constant_fractional_fps_and_nonzero_pts_origin(self):
        for fps in (60., 60000/1001):
            pts = (5. + np.arange(120) / fps).tolist()
            result = verify_cfr(pts, 1e-6, fps)
            self.assertEqual(result.report()['time_source'], TIME_SOURCE)
            self.assertEqual(result.report()['verified_frame_count'], 120)
            self.assertEqual(result.pts_s[0], 5.)

    def test_rejects_variable_intervals_drift_missing_or_duplicate_pts(self):
        for pts, fps in (([0., 1/60, 3/60, 4/60], 60.),
                         ((np.arange(200) / 59.94).tolist(), 60.),
                         ([0., 0., 0.02], 60.), ([0., float('nan')], 60.),
                         ([0.], 60.), ([0., 0.1], float('nan'))):
            with self.subTest(pts=pts[:4], fps=fps), self.assertRaises(ValueError):
                verify_cfr(pts, 1e-6, fps)

    def test_timestamp_quantization_is_bounded(self):
        pts = np.round(np.arange(120) / 60., 3).tolist()
        result = verify_cfr(pts, 0.001, 60.)
        self.assertLessEqual(result.max_time_error_s, 0.001)
        self.assertLessEqual(result.tolerance_s, 0.001)

    def test_real_ffmpeg_and_extraction_use_deliberate_vfr_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'cfr.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 30, (160, 160))
            self.assertTrue(writer.isOpened())
            for i in range(12):
                writer.write(np.full((160, 160, 3), i * 15, dtype=np.uint8))
            writer.release()
            timing = audit_video_timing(source, 30.)
            self.assertEqual(len(timing.pts_s), 12)
            np.testing.assert_allclose(timing.pts_s, np.arange(12) / 30., atol=1e-9)
            target = Path(directory) / 'vfr.mkv'
            command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                       '-i', str(source), '-vf', "setpts=if(lt(N\,6)\,N\,N+3)/(30*TB)",
                       '-fps_mode', 'passthrough', '-c:v', 'ffv1', str(target)]
            subprocess.run(command, check=True, capture_output=True)
            variable = audit_video_timing(target, 30.)
            self.assertFalse(variable.is_cfr)
            self.assertEqual(variable.frame_range(0.2, 0.35), (6, 8))
            model = Path(directory) / 'model.task'
            model.touch()
            detector = MagicMock()
            detector.detect_for_video.return_value.pose_landmarks = []
            with patch('src.knee_angle_validation.pose_extraction.vision.PoseLandmarker.create_from_options', return_value=detector):
                result = process_video(target, Path(directory) / 'points.csv', 'v1', 'p1', 'right', model_path=model,
                                       resize_height=0, start_time_s=0.2, end_time_s=0.35)
            self.assertEqual(result.data.frame_number.tolist(), [6, 7])
            np.testing.assert_allclose(result.data.time_s, [0.3, 1/3], atol=0.001)
            self.assertEqual([call.args[1] for call in detector.detect_for_video.call_args_list], [300, 333])
            self.assertTrue(result.data.time_source.eq(TIME_SOURCE).all())
            self.assertEqual(result.data.mediapipe_timestamp_ms.tolist(), [300, 333])
            self.assertAlmostEqual(float(read_analysis_frame(target, 6, resize_height=0).mean()), 90., delta=5)
            from src.knee_angle_validation.overlay import export_overlay_video
            preview = Path(directory) / 'vfr_preview.mp4'
            export_overlay_video(target, result.data, preview, resize_height=0)
            cap = cv2.VideoCapture(str(preview))
            try:
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            finally:
                cap.release()

    def test_failed_timing_audit_prevents_inference_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, model = root / 'video.mov', root / 'model.task'
            source.write_bytes(b'video')
            model.touch()
            capture = MagicMock()
            capture.get.return_value = 60.
            with patch('src.knee_angle_validation.pose_extraction.cv2.VideoCapture', return_value=capture), \
                 patch('src.knee_angle_validation.pose_extraction.audit_video_timing', side_effect=ValueError('VFR')), \
                 patch('src.knee_angle_validation.pose_extraction.vision.PoseLandmarker.create_from_options') as detector:
                with self.assertRaisesRegex(ValueError, 'VFR'):
                    process_video(source, root / 'output.csv', 'v1', 'p1', 'right', model_path=model)
                detector.assert_not_called()
                capture.release.assert_called_once()
            self.assertFalse((root / 'output.csv').exists())


if __name__ == '__main__':
    unittest.main()
