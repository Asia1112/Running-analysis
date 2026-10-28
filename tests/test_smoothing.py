import unittest

import numpy as np

from src.knee_angle_validation.smoothing import median_filter, moving_average, savitzky_golay, smooth_series
from src.knee_angle_validation.statistics import error_statistics
from src.knee_angle_validation.validation import merge_with_reference
from src.knee_angle_validation.reference_annotation import build_reference_annotation
import pandas as pd
from streamlit.testing.v1 import AppTest


class SmoothingTests(unittest.TestCase):
    def test_app_accepts_window_longer_than_data_and_preserves_missing_angles(self):
        script = '''
import app
import streamlit as st
app._angle_view(st.session_state.test_data)
'''
        source = pd.DataFrame({'time_s': [0., 1/60, 2/60], 'frame_number': [0, 1, 2],
                               'knee_angle_raw': [90., np.nan, 130.], 'knee_flexion_deg': [90., np.nan, 50.],
                               'pose_found': [True, False, True]})
        for point in ('hip', 'knee', 'ankle'):
            source[f'{point}_visibility'] = [1., np.nan, 1.]
            source[f'{point}_x_norm'] = [0.1, np.nan, 0.2]
            source[f'{point}_y_norm'] = [0.1, np.nan, 0.2]
        at = AppTest.from_string(script, default_timeout=30)
        at.session_state.test_data = source.copy()
        at.run()
        self.assertFalse(at.exception)
        at.segmented_control[0].set_value('moving_average').run()
        self.assertFalse(at.exception)
        result = at.session_state.test_data
        np.testing.assert_allclose(result.knee_angle_smoothed, [90., np.nan, 130.], equal_nan=True)
        np.testing.assert_array_equal(result.knee_angle_raw, source.knee_angle_raw)
        self.assertTrue(result.smoothing_gap_policy.eq('finite_segments_v1').all())

    def test_all_filters_preserve_missing_positions_and_do_not_mix_across_gaps(self):
        source = np.array([np.nan, 10., 10., np.nan, 100., 100., np.inf, 200., np.nan])
        original = source.copy()
        for method in ('median', 'moving_average', 'savgol'):
            with self.subTest(method=method):
                result = smooth_series(source, method, 7)
                self.assertEqual(result.shape, source.shape)
                np.testing.assert_array_equal(np.isnan(result), ~np.isfinite(source))
                np.testing.assert_allclose(result[np.isfinite(source)], source[np.isfinite(source)], atol=1e-10)
        np.testing.assert_array_equal(source, original)

    def test_empty_missing_and_short_series_keep_exact_length(self):
        for source in (np.array([]), np.array([np.nan]), np.array([np.nan, np.nan]),
                       np.array([4.]), np.array([4., 8.]), np.array([4., np.nan])):
            for method in ('median', 'moving_average', 'savgol'):
                with self.subTest(source=source, method=method):
                    result = smooth_series(source, method, 9)
                    self.assertEqual(result.shape, source.shape)
                    self.assertTrue(np.array_equal(np.isfinite(result), np.isfinite(source)))
        np.testing.assert_allclose(moving_average(np.array([4., 8.]), 9), [6., 6.])

    def test_changing_data_after_gap_cannot_change_filtered_data_before_gap(self):
        before = np.array([1., 2., 5., 3., 1.])
        for method in ('median', 'moving_average', 'savgol'):
            a = smooth_series(np.r_[before, np.nan, [100., 200., 300.]], method, 7)
            b = smooth_series(np.r_[before, np.nan, [-1000., -2000., -3000.]], method, 7)
            np.testing.assert_allclose(a[:5], b[:5], atol=1e-10)

    def test_constant_and_polynomial_curves_have_expected_behavior(self):
        for method in ('median', 'moving_average', 'savgol'):
            np.testing.assert_allclose(smooth_series(np.full(9, 90.), method, 5), 90., atol=1e-10)
        values = np.arange(9, dtype=float) ** 2 + 3
        np.testing.assert_allclose(savitzky_golay(values, 5), values, atol=1e-10)

    def test_invalid_windows_orders_and_dimensions_are_rejected(self):
        for method in ('median', 'moving_average', 'savgol'):
            for window in (0, -1, 2, 4, 3.5, True):
                with self.subTest(method=method, window=window), self.assertRaises(ValueError):
                    smooth_series(np.array([1., 2., 3.]), method, window)
            with self.assertRaises(ValueError):
                smooth_series(np.ones((2, 2)), method, 3)
        for order in (-1, 3, 1.5, True):
            with self.assertRaises(ValueError):
                savitzky_golay(np.arange(5.), 3, order)

    def test_filtering_changes_extrema_but_cannot_change_raw_validation(self):
        self.assertLess(median_filter(np.array([90., 90., 160., 90., 90.]), 3).max(), 160.)
        frame = pd.Series(dict(video_id='v1', person_id='p1', frame_number=10, side='right'))
        extracted = pd.DataFrame([{**frame.to_dict(), 'hip_x_px': 0., 'hip_y_px': 0.,
            'knee_x_px': 0., 'knee_y_px': 10., 'ankle_x_px': 10., 'ankle_y_px': 10.,
            'knee_angle_smoothed': 0., 'knee_flexion_smoothed_deg': 180.}])
        reference = pd.DataFrame([build_reference_annotation(frame, [(0., 0.), (0., 10.), (10., 10.)], 'a1', 's1')])
        first = merge_with_reference(extracted, reference)
        second = merge_with_reference(extracted.assign(knee_angle_smoothed=180., knee_flexion_smoothed_deg=0.), reference)
        self.assertEqual(error_statistics(first)['mae'], error_statistics(second)['mae'])
        pd.testing.assert_series_equal(first.error_signed, second.error_signed)
        self.assertEqual(first.angle_mediapipe.tolist(), [90.])

    def test_filters_preserve_length_and_reduce_impulse(self):
        values = np.array([100.0, 100.0, 160.0, 100.0, 100.0])
        self.assertEqual(len(moving_average(values, 3)), len(values))
        self.assertEqual(median_filter(values, 3)[2], 100.0)
        self.assertEqual(len(savitzky_golay(values, 5)), len(values))


if __name__ == "__main__":
    unittest.main()
