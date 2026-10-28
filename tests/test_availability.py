"""Coverage denominators include unpaired, unannotated and unavailable frames."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.knee_angle_validation.reference_annotation import build_reference_annotation
from src.knee_angle_validation.validation import compare_with_inventory
from src.knee_angle_validation.statistics import (
    availability_statistics, grouped_availability, grouped_reference_statistics,
    reference_quality_summary, recording_statistics,
)


def inputs():
    identities = [dict(video_id='v1', person_id='p1', side='right', frame_number=i) for i in range(8)]
    source = pd.DataFrame([{**identity, 'hip_x_px': 0. if i in (0, 2, 4) else np.nan, 'hip_y_px': 0.,
                           'knee_x_px': 0., 'knee_y_px': 10., 'ankle_x_px': 10., 'ankle_y_px': 10.}
                          for i, identity in enumerate(identities[:6])])
    reference = []
    for i in (0, 1, 2, 3, 6, 7):
        quality = dict(hip='CLEAR', knee='UNMARKABLE' if i == 2 else 'UNCERTAIN' if i == 3 else 'CLEAR', ankle='CLEAR')
        reference.append(build_reference_annotation(pd.Series(identities[i]), [(0, 0), (0, 10), (10, 10)],
                                                     'a1', 's1', point_qualities=quality))
    selected = pd.DataFrame([identities[i] for i in (0, 1, 2, 3, 4, 6)])
    return source, pd.DataFrame(reference), selected


class AvailabilityTests(unittest.TestCase):
    def test_app_reports_coverage_when_no_reference_key_has_a_pair(self):
        script = '''
from pathlib import Path
import app
import streamlit as st
app.RESULTS_DIR = Path(st.session_state.test_directory)
app._initialize_state()
st.session_state.active_analysis_id = 'a' * 32
app._validation_view(st.session_state.test_data)
app._statistics_view()
'''
        source, reference, selected = inputs()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'analyses' / ('a' * 32)
            folder.mkdir(parents=True)
            reference.loc[reference.frame_number.eq(6)].to_csv(folder / 'reference_annotations.csv', index=False)
            selected.loc[selected.frame_number.eq(6)].to_csv(folder / 'selected_reference_frames.csv', index=False)
            at = AppTest.from_string(script, default_timeout=30)
            at.session_state.test_directory = directory
            at.session_state.test_data = source
            at.run()
            self.assertFalse(at.exception)
            at.button[0].click().run()
            self.assertFalse(at.exception)
            self.assertTrue(at.session_state.validation.empty)
            coverage = pd.read_csv(folder / 'availability_summary.csv')
            self.assertEqual(coverage.iloc[0].n_selected_reference, 1)
            self.assertEqual(coverage.iloc[0].n_reference_without_extraction, 1)
            self.assertEqual(coverage.iloc[0].n_paired_available, 0)

    def test_planned_only_frame_retains_recording_metadata_and_wrong_plan_is_rejected(self):
        source, reference, selected = inputs()
        metadata = dict(analysis_id='a' * 32, source_sha256='b' * 64, crop_mode='none', resize_height=0)
        source = source.assign(**metadata)
        reference = reference.assign(**metadata)
        selected = pd.concat([selected, selected.iloc[[0]].assign(frame_number=999)], ignore_index=True).assign(**metadata)
        _, ledger = compare_with_inventory(source, reference, selected_frames=selected)
        self.assertEqual(ledger.set_index('frame_number').loc[999, 'analysis_id'], 'a' * 32)
        with self.assertRaises(ValueError):
            compare_with_inventory(source, reference, selected_frames=selected.assign(source_sha256='c' * 64))

    def test_all_populations_and_missing_measurements_have_explicit_counts(self):
        source, reference, selected = inputs()
        pairs, ledger = compare_with_inventory(source, reference, selected_frames=selected)
        self.assertEqual(len(ledger), 8)
        self.assertEqual(len(pairs), 4)
        report = availability_statistics(ledger)
        expected = dict(n_source_frames=6, n_mediapipe_available=3, n_mediapipe_missing=3,
                        n_selected_reference=6, n_reference_scope=6, n_reference_annotations=5,
                        n_not_annotated=1, n_reference_available=4, n_reference_unavailable=2,
                        n_mediapipe_available_in_scope=3, n_mediapipe_unavailable_in_scope=3,
                        n_paired_available=1, n_primary_pairs=1, n_only_mediapipe=2,
                        n_only_reference=3, n_both_unavailable=0,
                        n_reference_without_extraction=2, n_selected_without_extraction=1,
                        n_reference_outside_selection=1)
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(report[key], value)
        self.assertEqual(report['mediapipe_available_pct'], 50.)
        self.assertAlmostEqual(report['paired_available_pct'], 100/6)
        self.assertEqual(report['n_paired_available'] + report['n_only_mediapipe'] +
                         report['n_only_reference'] + report['n_both_unavailable'], report['n_reference_scope'])
        counts = reference_quality_summary(ledger).set_index('frame_quality')
        self.assertEqual(counts.n_reference.sum(), 5)
        self.assertEqual(counts.loc['CLEAR', 'n_reference'], 3)
        self.assertEqual(ledger.set_index('frame_number').loc[6, 'reference_available'], True)

    def test_absent_plan_is_unknown_instead_of_assumed_complete(self):
        source, reference, _ = inputs()
        _, ledger = compare_with_inventory(source, reference)
        report = availability_statistics(ledger)
        self.assertTrue(np.isnan(report['n_selected_reference']))
        self.assertTrue(np.isnan(report['n_not_annotated']))
        self.assertEqual(report['n_reference_scope'], 6)
        self.assertEqual(report['scope_definition'], 'annotation_rows_or_mixed')

    def test_unplanned_reference_does_not_increase_primary_metrics(self):
        source, reference, selected = inputs()
        # Both frame 0 and frame 2 have a valid pair, but only 0 is planned here.
        reference.loc[reference.frame_number.eq(2), ['knee_quality', 'frame_quality']] = 'CLEAR'
        reference.loc[reference.frame_number.eq(2), ['knee_x_ref', 'knee_y_ref']] = [0., 10.]
        planned = selected.loc[selected.frame_number.eq(0)]
        pairs, ledger = compare_with_inventory(source, reference, selected_frames=planned)
        summary = grouped_reference_statistics(ledger, []).set_index('analysis_subset')
        self.assertEqual(summary.loc['primary_clear', 'n'], 1)
        self.assertEqual(pairs.reference_scope.sum(), 1)
        self.assertEqual(availability_statistics(ledger)['n_reference_outside_selection'], 5)

    def test_selected_only_key_is_preserved(self):
        source, reference, selected = inputs()
        selected = pd.concat([selected, selected.iloc[[0]].assign(frame_number=999)], ignore_index=True)
        _, ledger = compare_with_inventory(source, reference, selected_frames=selected)
        row = ledger.set_index('frame_number').loc[999]
        self.assertFalse(row.in_extraction)
        self.assertFalse(row.has_reference_row)
        self.assertTrue(row.reference_scope)
        self.assertFalse(row.paired_available)

    def test_speed_and_recording_tables_keep_frame_and_recording_units_separate(self):
        source, reference, selected = inputs()
        _, first = compare_with_inventory(source, reference, selected_frames=selected)
        first['speed_kmh'] = 7.5
        second = first.assign(video_id='v2', person_id='p2', speed_kmh=8.5)
        both = pd.concat([first, second], ignore_index=True)
        availability = grouped_availability(both, ['video_id', 'person_id', 'speed_kmh'])
        self.assertEqual(len(availability.loc[availability.group_type.eq('speed_kmh')]), 2)
        overall = availability.loc[availability.group_type.eq('overall')].iloc[0]
        self.assertEqual(overall.n_recordings, 2)
        self.assertEqual(overall.n_persons, 2)
        per_recording = recording_statistics(both)
        self.assertEqual(len(per_recording), 4)
        self.assertEqual(set(per_recording.speed_kmh), {7.5, 8.5})
        self.assertEqual(per_recording.loc[per_recording.analysis_subset.eq('primary_clear'), 'n'].tolist(), [1, 1])

    def test_cli_exports_inventory_and_coverage_including_unmatched_rows(self):
        from analyze_run_csv import main
        source, reference, selected = inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source.to_csv(root / 'source.csv', index=False)
            reference.to_csv(root / 'reference.csv', index=False)
            selected.to_csv(root / 'selected_reference_frames.csv', index=False)
            args = ['analyze_run_csv.py', '--extracted', str(root / 'source.csv'), '--reference', str(root / 'reference.csv'),
                    '--results', str(root / 'pairs.csv'), '--summary', str(root / 'summary.csv')]
            with patch('sys.argv', args), patch('sys.stdout'):
                main()
            ledger = pd.read_csv(root / 'summary_frame_inventory.csv')
            coverage = pd.read_csv(root / 'summary_availability.csv')
            self.assertEqual(len(ledger), 8)
            self.assertEqual(coverage.iloc[0].n_selected_reference, 6)
            self.assertTrue((root / 'summary_recordings.csv').exists())

    def test_study_reports_combine_recordings_without_double_counting_a_reanalysis(self):
        from summarize_experiments import combine_inventories, main
        source, reference, selected = inputs()
        _, ledger = compare_with_inventory(source, reference, selected_frames=selected)
        ledger = ledger.assign(analysis_id='a' * 32, source_sha256='b' * 64, crop_mode='none', resize_height=0, speed_kmh=7.5)
        second = ledger.assign(analysis_id='c' * 32, source_sha256='d' * 64, video_id='v2', person_id='p2', speed_kmh=8.5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path, second_path = root / 'first.csv', root / 'second.csv'
            ledger.to_csv(first_path, index=False)
            second.to_csv(second_path, index=False)
            combined = combine_inventories([first_path, second_path])
            self.assertEqual(availability_statistics(combined)['n_recordings'], 2)
            with self.assertRaisesRegex(ValueError, 'Powtórzone nagranie'):
                combine_inventories([first_path, first_path])
            args = ['summarize_experiments.py', '--inventories', str(first_path), str(second_path), '--output-dir', str(root / 'report')]
            with patch('sys.argv', args), patch('sys.stdout'):
                main()
            summary = pd.read_csv(root / 'report' / 'summary_statistics.csv')
            self.assertEqual(set(summary.loc[summary.group_type.eq('speed_kmh'), 'group_value'].astype(str)), {'7.5', '8.5'})
            second.loc[0, 'mediapipe_available'] = False
            second.to_csv(second_path, index=False)
            with self.assertRaisesRegex(ValueError, 'Flagi dostępności'):
                combine_inventories([first_path, second_path])


if __name__ == '__main__':
    unittest.main()
