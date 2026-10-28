"""Temporal interval extraction preserves source indices and timestamps."""

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.knee_angle_validation.pose_extraction import process_video
from src.knee_angle_validation.provenance import frame_limits, validate_time_range, recording_metadata
from src.knee_angle_validation.reference_annotation import select_reference_frames, build_reference_annotation
from src.knee_angle_validation.validation import merge_with_reference
from src.knee_angle_validation.video_timing import verify_cfr


class AnalysisRangeTests(unittest.TestCase):
    def test_cli_defaults_to_twenty_seconds_after_selected_start(self):
        from pose_video import main
        args = ['pose_video.py', '--source', 'IMG_1302.MOV', '--start', '5']
        with patch('sys.argv', args), patch('sys.stdout'), patch('pose_video.process_video') as process:
            process.return_value = SimpleNamespace(frames=1200, csv_path=Path('unused.csv'), pose_found_pct=100.)
            main()
            self.assertEqual(process.call_args.kwargs['start_time_s'], 5.)
            self.assertEqual(process.call_args.kwargs['end_time_s'], 25.)
        with patch('sys.argv', args + ['--end', '5']), patch('sys.stderr'), patch('pose_video.process_video') as process:
            with self.assertRaises(SystemExit):
                main()
            process.assert_not_called()

    def extract(self, directory, start=1., end=3., fps=10., count=50):
        root = Path(directory)
        source, model = root / 'source.mov', root / 'model.task'
        source.write_bytes(b'video')
        model.touch()
        frame = np.zeros((160, 160, 3), dtype=np.uint8)
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.return_value = fps
        capture.read.side_effect = [(True, frame)] * count + [(False, None)]
        detector = MagicMock()
        detector.detect_for_video.return_value = SimpleNamespace(pose_landmarks=[])
        with patch('src.knee_angle_validation.pose_extraction.cv2.VideoCapture', return_value=capture), \
             patch('src.knee_angle_validation.pose_extraction.vision.PoseLandmarker.create_from_options', return_value=detector), \
             patch('src.knee_angle_validation.pose_extraction.audit_video_timing',
                   side_effect=lambda source, actual_fps: verify_cfr([i / actual_fps for i in range(count)], 1e-6, actual_fps)), \
             warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = process_video(source, root / 'points.csv', 'v1', 'p1', 'right', model_path=model,
                                   start_time_s=start, end_time_s=end)
        return result, capture, detector

    def test_only_selected_frames_are_inferred_with_original_time_and_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            result, capture, detector = self.extract(directory)
            self.assertEqual(result.frames, 20)
            self.assertEqual(result.data.frame_number.tolist(), list(range(10, 30)))
            np.testing.assert_allclose(result.data.time_s, np.arange(10, 30) / 10)
            self.assertEqual(detector.detect_for_video.call_count, 20)
            self.assertEqual([call.args[1] for call in detector.detect_for_video.call_args_list], list(range(1000, 3000, 100)))
            self.assertEqual(capture.read.call_count, 30)
            capture.release.assert_called_once()
            detector.close.assert_called_once()
            saved = pd.read_csv(result.csv_path)
            self.assertEqual(saved.frame_number.tolist(), list(range(10, 30)))
            metadata = json.loads(result.csv_path.with_suffix('.metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['analysis_start_s'], 1.)
            self.assertEqual(metadata['analysis_end_s'], 3.)
            selected = select_reference_frames(result.data, count=5)
            self.assertTrue(selected.frame_number.between(10, 29).all())
            self.assertTrue(selected.time_s.between(1., 2.9).all())
            reference = pd.DataFrame([build_reference_annotation(selected.iloc[0], [(0, 0), (0, 10), (10, 10)], 'a1', 's1')])
            merged = merge_with_reference(result.data, reference)
            self.assertEqual(merged.frame_number.tolist(), [int(selected.frame_number.iloc[0])])
            self.assertEqual(merged.analysis_start_s.tolist(), [1.])
            with self.assertRaises(ValueError):
                merge_with_reference(result.data, reference.assign(analysis_end_s=4.))
            with self.assertRaisesRegex(ValueError, 'inny czas klatki'):
                merge_with_reference(result.data, reference.assign(time_s=reference.time_s + 0.01))

    def test_fractional_boundaries_and_sixty_fps_twenty_second_window(self):
        self.assertEqual(frame_limits(0.3, 0.6, 10.), (3, 6))
        self.assertEqual(frame_limits(0.301, 0.601, 10.), (4, 7))
        self.assertEqual(frame_limits(5., 25., 60.), (300, 1500))
        with tempfile.TemporaryDirectory() as directory:
            result, _, detector = self.extract(directory, start=0.31, end=0.61)
            self.assertEqual(result.data.frame_number.tolist(), [4, 5, 6])
            self.assertEqual(detector.detect_for_video.call_count, 3)

    def test_incomplete_or_empty_interval_produces_no_results(self):
        for start, end, count in ((1., 3., 20), (6., None, 50), (0.01, 0.02, 50)):
            with self.subTest(start=start, end=end), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.extract(directory, start=start, end=end, count=count)
                self.assertFalse((Path(directory) / 'points.csv').exists())

    def test_invalid_times_or_fps_are_rejected(self):
        for start, end in ((-1, 2), (1, 1), (2, 1), (float('nan'), 3), (0, float('inf'))):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                validate_time_range(start, end)
        for fps in (0., -1., float('nan'), float('inf')):
            with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
                self.extract(directory, fps=fps)

    def test_unbounded_api_retains_full_recording_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self.extract(directory, start=0., end=None, count=5)
            self.assertEqual(result.data.frame_number.tolist(), [0, 1, 2, 3, 4])
            self.assertEqual(recording_metadata(result.data)['analysis_end_s'], 0.5)

    def test_app_defaults_to_twenty_seconds_and_changing_range_clears_results(self):
        from test_recording_identity import EXTRACTION_SCRIPT
        with tempfile.TemporaryDirectory() as directory:
            at = AppTest.from_string(EXTRACTION_SCRIPT, default_timeout=30)
            at.session_state.test_directory = directory
            at.session_state.video_bytes = b'video A'
            at.run()
            at.text_input(key='input_person_id').set_value('p1').run()
            self.assertEqual(at.number_input(key='input_start_s').value, 0.)
            self.assertEqual(at.number_input(key='input_end_s').value, 20.)
            at.button[0].click().run()
            self.assertFalse(at.exception)
            old_id = at.session_state.active_analysis_id
            self.assertEqual(at.session_state.extracted.analysis_end_s.tolist(), [20.])
            at.number_input(key='input_start_s').set_value(5.).run()
            self.assertFalse(at.exception)
            self.assertIsNone(at.session_state.extracted)
            at.number_input(key='input_end_s').set_value(25.).run()
            at.button[0].click().run()
            self.assertFalse(at.exception)
            self.assertNotEqual(at.session_state.active_analysis_id, old_id)
            self.assertEqual(at.session_state.extracted.analysis_start_s.tolist(), [5.])
            self.assertEqual(at.session_state.extracted.analysis_end_s.tolist(), [25.])


if __name__ == '__main__':
    unittest.main()
