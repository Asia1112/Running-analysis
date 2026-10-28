"""Study table mapping, validation and recording-switch regressions."""

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.knee_angle_validation.experiment_metadata import (
    EXPERIMENT_KEYS, experiment_values, load_video_metadata, metadata_for_video,
    normalize_experiment_metadata,
)
from src.knee_angle_validation.reference_annotation import build_reference_annotation
from src.knee_angle_validation.validation import merge_with_reference

HEADER = 'original_filename;participant_id;speed_kmh;analyzed_side;inclusion_status;notes\n'


class ExperimentMetadataTests(unittest.TestCase):
    def test_excluded_recording_is_blocked_before_model_inference(self):
        from pose_video import main
        from src.knee_angle_validation.pose_extraction import process_video
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / 'table.csv'
            table.write_text(HEADER + 'x.mov;P01;7.5;RIGHT;EXCLUDED;pilot only\n', encoding='utf-8')
            args = ['pose_video.py', '--source', 'x.mov', '--metadata', str(table)]
            with patch('sys.argv', args), patch('sys.stderr'), patch('pose_video.process_video') as process:
                with self.assertRaises(SystemExit):
                    main()
                process.assert_not_called()
            source, model = root / 'x.mov', root / 'model.task'
            source.write_bytes(b'video')
            model.touch()
            with patch('src.knee_angle_validation.pose_extraction.cv2.VideoCapture') as capture:
                with self.assertRaisesRegex(ValueError, 'INCLUDED'):
                    process_video(source, root / 'output.csv', 'v1', 'P01', 'right', model_path=model,
                                  experiment_metadata=metadata_for_video('x.mov', load_video_metadata(table)))
                capture.assert_not_called()
            self.assertFalse((root / 'output.csv').exists())

    def test_actual_table_maps_five_people_and_two_speeds(self):
        table = load_video_metadata()
        self.assertEqual(len(table), 10)
        self.assertEqual(set(table.person_id), {'P01', 'P02', 'P03', 'P04', 'P05'})
        self.assertEqual(set(table.speed_kmh), {7.5, 8.5})
        self.assertTrue(table.side.eq('right').all())
        self.assertEqual(metadata_for_video('img_1302.mov', table)['person_id'], 'P01')
        self.assertEqual(metadata_for_video('IMG_1311.MOV', table)['speed_kmh'], 8.5)
        self.assertIsNone(metadata_for_video('unknown.mov', table))
        self.assertTrue(table.metadata_table_sha256.str.len().eq(64).all())

    def test_invalid_and_duplicate_rows_are_rejected(self):
        invalid = ['x.mov;;7.5;RIGHT;INCLUDED;', 'x.mov;P01;;RIGHT;INCLUDED;',
                   'x.mov;P01;0;RIGHT;INCLUDED;', 'x.mov;P01;inf;RIGHT;INCLUDED;',
                   'x.mov;P01;bad;RIGHT;INCLUDED;', 'x.mov;P01;7.5;front;INCLUDED;',
                   'x.mov;P01;7.5;RIGHT;typo;', ';P01;7.5;RIGHT;INCLUDED;',
                   'x.mov;P01;7.5;RIGHT;INCLUDED;\nX.MOV;P02;8.5;LEFT;INCLUDED;']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'table.csv'
            for row in invalid:
                with self.subTest(row=row):
                    path.write_text(HEADER + row + '\n', encoding='utf-8')
                    with self.assertRaises(ValueError):
                        load_video_metadata(path)
            path.write_text('wrong,header\n1,2', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_video_metadata(path)

    def test_text_ids_decimal_comma_and_unknown_speed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'table.csv'
            path.write_text(HEADER + ' x.mov ;001;7,5; left ;included;note\n', encoding='utf-8-sig')
            row = metadata_for_video('x.mov', load_video_metadata(path))
            self.assertEqual(row['person_id'], '001')
            self.assertEqual(row['speed_kmh'], 7.5)
            self.assertEqual(row['side'], 'left')
        manual = normalize_experiment_metadata({'original_filename': 'other.mov'}, '001', 'left')
        self.assertIsNone(manual['speed_kmh'])
        with self.assertRaises(ValueError):
            normalize_experiment_metadata(row, 'P02', 'left')
        with self.assertRaises(ValueError):
            normalize_experiment_metadata(row, '001', 'right')

    def test_reference_retains_metadata_and_validation_rejects_relabeling(self):
        study = metadata_for_video('IMG_1302.MOV', load_video_metadata())
        extracted = pd.DataFrame([{**study, 'video_id': 'v1', 'frame_number': 10,
                                  'hip_x_px': 0., 'hip_y_px': 0., 'knee_x_px': 0., 'knee_y_px': 10.,
                                  'ankle_x_px': 10., 'ankle_y_px': 10.}])
        reference = pd.DataFrame([build_reference_annotation(extracted.iloc[0], [(0, 0), (0, 10), (10, 10)], 'a1', 's1')])
        merged = merge_with_reference(extracted, reference)
        self.assertEqual(merged.person_id.tolist(), ['P01'])
        for key in EXPERIMENT_KEYS:
            self.assertEqual(merged.iloc[0][key], study[key])
        with self.assertRaises(ValueError):
            merge_with_reference(extracted, reference.assign(speed_kmh=8.5))
        with self.assertRaises(ValueError):
            merge_with_reference(extracted, reference.drop(columns=list(EXPERIMENT_KEYS)))
        loaded = pd.read_csv(io.StringIO(extracted.to_csv(index=False)))
        self.assertEqual(experiment_values(loaded)['metadata_notes'], '')

    def test_cli_maps_metadata_and_rejects_conflicts_before_extraction(self):
        from pose_video import main
        args = ['pose_video.py', '--source', 'IMG_1302.MOV']
        with patch('sys.argv', args), patch('pose_video.process_video') as process, patch('sys.stdout'):
            process.return_value = SimpleNamespace(frames=1, csv_path=Path('unused.csv'), pose_found_pct=100.)
            main()
            self.assertEqual(process.call_args.args[2:5], ('IMG_1302', 'P01', 'right'))
            self.assertEqual(process.call_args.kwargs['experiment_metadata']['speed_kmh'], 7.5)
        for extra in (['--person-id', 'P02'], ['--side', 'left'], ['--speed-kmh', '8.5']):
            with patch('sys.argv', args + extra), patch('sys.stderr'), patch('pose_video.process_video') as process:
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                process.assert_not_called()

    def test_streamlit_switch_maps_person_speed_side_and_resets_previous_results(self):
        from test_recording_identity import EXTRACTION_SCRIPT
        script = EXTRACTION_SCRIPT.replace("'same_name.MOV'", "st.session_state.test_filename")
        with tempfile.TemporaryDirectory() as directory:
            at = AppTest.from_string(script, default_timeout=30)
            at.session_state.test_directory = directory
            at.session_state.test_filename = 'IMG_1302.MOV'
            at.session_state.video_bytes = b'first'
            at.run()
            self.assertFalse(at.exception)
            self.assertEqual(at.text_input(key='input_person_id').value, 'P01')
            self.assertTrue(at.text_input(key='input_person_id').disabled)
            self.assertEqual(at.number_input(key='input_speed').value, 7.5)
            at.button[0].click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.session_state.extracted.speed_kmh.tolist(), [7.5])
            at.session_state.test_filename = 'IMG_1305.MOV'
            at.session_state.video_bytes = b'second'
            at.session_state.video_token = 'B'
            at.run()
            self.assertFalse(at.exception)
            self.assertEqual(at.text_input(key='input_person_id').value, 'P02')
            self.assertEqual(at.number_input(key='input_speed').value, 8.5)
            self.assertEqual(at.session_state.input_side, 'right')
            self.assertIsNone(at.session_state.extracted)
            at.button[0].click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.session_state.extracted.person_id.tolist(), ['P02'])


if __name__ == '__main__':
    unittest.main()
