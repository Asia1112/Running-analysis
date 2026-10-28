"""Manual uncertainty is retained and separated from primary accuracy metrics."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.knee_angle_validation.reference_annotation import build_reference_annotation
from src.knee_angle_validation.annotation_storage import save_reference_annotation, read_reference_annotations
from src.knee_angle_validation.reference_quality import normalize_reference_quality
from src.knee_angle_validation.statistics import grouped_reference_statistics, reference_quality_summary, error_statistics
from src.knee_angle_validation.validation import merge_with_reference


def reference(points=None, **qualities):
    frame = pd.Series(dict(video_id='v1', person_id='p1', frame_number=10, side='right'))
    flags = dict(hip='CLEAR', knee='CLEAR', ankle='CLEAR')
    flags.update(qualities)
    return build_reference_annotation(frame, [(0, 0), (0, 10), (10, 10)] if points is None else points,
                                      'a1', 's1', point_qualities=flags)


class ReferenceQualityTests(unittest.TestCase):
    def test_uncertain_point_preserves_measurement_and_frame_classification(self):
        record = reference(knee='UNCERTAIN')
        self.assertEqual(record['frame_quality'], 'UNCERTAIN')
        self.assertEqual(record['knee_flexion_deg'], 90.)
        self.assertEqual(record['hip_quality'], 'CLEAR')

    def test_unmarkable_point_is_empty_even_if_coordinates_were_provided(self):
        record = reference(knee='UNMARKABLE')
        self.assertEqual(record['frame_quality'], 'UNMARKABLE')
        self.assertTrue(np.isnan(record['knee_x_ref']))
        self.assertTrue(np.isnan(record['knee_flexion_deg']))
        self.assertEqual(record['ankle_x_ref'], 10.)

    def test_no_click_annotation_round_trips_and_does_not_disappear_in_validation(self):
        record = reference([], hip='UNMARKABLE', knee='UNMARKABLE', ankle='UNMARKABLE')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.csv'
            save_reference_annotation(record, path)
            loaded = read_reference_annotations(path)
            self.assertEqual(loaded.frame_quality.tolist(), ['UNMARKABLE'])
            self.assertEqual(loaded.knee_x_ref.tolist(), [''])
            extracted = pd.DataFrame([dict(video_id='v1', person_id='p1', frame_number=10, side='right',
                                          hip_x_px=0., hip_y_px=0., knee_x_px=0., knee_y_px=10., ankle_x_px=10., ankle_y_px=10.)])
            merged = merge_with_reference(extracted, loaded)
            self.assertEqual(len(merged), 1)
            self.assertTrue(merged.angle_reference.isna().all())
            self.assertTrue(merged.error_signed.isna().all())
            self.assertEqual(reference_quality_summary(merged).set_index('frame_quality').loc['UNMARKABLE', 'n_reference'], 1)

    def test_invalid_flags_missing_visible_points_and_inconsistent_frame_are_rejected(self):
        with self.assertRaises(ValueError):
            reference(knee='bad')
        with self.assertRaises(ValueError):
            reference([(0, 0), None, (10, 10)])
        with self.assertRaises(ValueError):
            normalize_reference_quality(pd.DataFrame([reference()]).assign(frame_quality='UNCERTAIN'))
        with self.assertRaises(ValueError):
            normalize_reference_quality(pd.DataFrame([reference()]).drop(columns='knee_quality'))

    def test_legacy_measurements_are_never_silently_assigned_clear(self):
        record = reference()
        for key in ('hip_quality', 'knee_quality', 'ankle_quality', 'frame_quality'):
            record.pop(key)
        normalized = normalize_reference_quality(pd.DataFrame([record]))
        self.assertEqual(normalized.frame_quality.tolist(), ['UNASSESSED'])

    def test_primary_sensitivity_and_quality_denominators_are_separate(self):
        data = pd.DataFrame(dict(error_signed=[2., 20., np.nan, 100., np.inf],
                                 frame_quality=['CLEAR', 'UNCERTAIN', 'UNMARKABLE', 'UNASSESSED', 'CLEAR']))
        overall = grouped_reference_statistics(data, []).set_index('analysis_subset')
        self.assertEqual(overall.loc['primary_clear', 'n'], 1)
        self.assertEqual(overall.loc['primary_clear', 'mae'], 2.)
        self.assertEqual(overall.loc['sensitivity_clear_uncertain', 'n'], 2)
        self.assertEqual(overall.loc['sensitivity_clear_uncertain', 'mae'], 11.)
        self.assertEqual(overall.loc['legacy_unassessed', 'mae'], 100.)
        self.assertEqual(reference_quality_summary(data).n_reference.sum(), 5)
        self.assertEqual(reference_quality_summary(data).n_valid_pairs.sum(), 3)
        self.assertEqual(error_statistics(data)['n'], 3)

    def test_ui_can_save_zero_clicks_and_missing_knee_without_mislabeling_ankle(self):
        from test_annotation_ui import ANNOTATION_SCRIPT, PROVENANCE
        def quality(at, point):
            return next(box for box in at.selectbox if box.key and box.key.startswith('quality_') and box.key.endswith('_' + point))
        with tempfile.TemporaryDirectory() as directory:
            at = AppTest.from_string(ANNOTATION_SCRIPT, default_timeout=30)
            at.session_state.test_directory = directory
            at.session_state.video_path = 'test.mov'
            at.session_state.annotation_session = 's1'
            at.session_state.selected_frames = pd.DataFrame([dict(video_id='v1', person_id='p1', frame_number=10,
                side='right', image_width=160, image_height=160, **PROVENANCE)])
            at.run()
            for point in ('hip', 'knee', 'ankle'):
                quality(at, point).set_value('UNMARKABLE').run()
                self.assertFalse(at.exception)
            self.assertFalse(at.button(key='save_annotation').disabled)
            at.button(key='save_annotation').click().run()
            self.assertFalse(at.exception)
            at.text_input(key='annotation_session').set_value('s2').run()
            quality(at, 'knee').set_value('UNMARKABLE').run()
            at.session_state.points_event = [dict(x=10., y=20.), dict(x=100., y=120.)]
            at.run()
            at.button(key='save_annotation').click().run()
            self.assertFalse(at.exception)
            saved = read_reference_annotations(Path(directory) / 'analyses' / ('a' * 32) / 'reference_annotations.csv')
            self.assertEqual(saved.annotation_session.tolist(), ['s1', 's2'])
            self.assertEqual(saved.frame_quality.tolist(), ['UNMARKABLE', 'UNMARKABLE'])
            self.assertEqual(float(saved.iloc[1].ankle_x_ref), 100.)
            self.assertEqual(saved.iloc[1].knee_x_ref, '')


if __name__ == '__main__':
    unittest.main()
