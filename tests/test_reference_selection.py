import unittest

import numpy as np
import pandas as pd

from src.knee_angle_validation.reference_annotation import select_reference_frames


def frames(times):
    return pd.DataFrame({"frame_number": np.arange(len(times)), "time_s": times,
                         "video_id": "v1", "person_id": "p1", "side": "right"})


class ReferenceSelectionTests(unittest.TestCase):
    def test_30_midpoints_and_source_indexes_are_preserved(self):
        data = frames(np.arange(61, dtype=float))
        data.index = np.arange(800, 861)
        selected = select_reference_frames(data)
        np.testing.assert_array_equal(selected.frame_number, np.arange(1, 60, 2))
        np.testing.assert_array_equal(selected.selection_target_time_s, np.arange(1, 60, 2))
        self.assertEqual(len(selected), 30)
        self.assertTrue(selected.frame_number.is_unique)
        self.assertTrue(selected.selection_time_error_s.eq(0).all())
        self.assertEqual(selected.selection_method.unique().tolist(), ["time_bin_midpoints_v1"])
        self.assertNotIn("selection_method", data)

    def test_selection_is_independent_of_all_mediapipe_results(self):
        data = frames(np.arange(61, dtype=float))
        expected = select_reference_frames(data).frame_number.tolist()
        data["pose_found"] = False
        data["knee_angle_raw"] = np.nan
        data["min_visibility"] = np.nan
        selected = select_reference_frames(data)
        self.assertEqual(selected.frame_number.tolist(), expected)
        self.assertFalse(selected.pose_found.any())
        self.assertTrue(selected.knee_angle_raw.isna().all())
        data["pose_found"] = True
        data["knee_angle_raw"] = 180.
        data["min_visibility"] = 1.
        self.assertEqual(select_reference_frames(data).frame_number.tolist(), expected)

    def test_interval_uses_source_numbers_without_resetting_time(self):
        selected = select_reference_frames(frames(np.arange(61, dtype=float)), 5,
                                           start_time_s=10., end_time_s=20.)
        self.assertEqual(selected.frame_number.tolist(), [11, 13, 15, 17, 19])
        self.assertEqual(selected.time_s.tolist(), [11., 13., 15., 17., 19.])
        self.assertTrue(selected.selection_start_s.eq(10.).all())
        self.assertTrue(selected.selection_end_s.eq(20.).all())

    def test_irregular_times_use_nearest_time_and_earlier_frame_at_tie(self):
        data = frames([0., 0.2, 0.9, 2., 2.1, 5.])
        selected = select_reference_frames(data, 3)
        self.assertEqual(selected.frame_number.tolist(), [2, 4, 5])
        np.testing.assert_allclose(selected.selection_target_time_s, [5/6, 2.5, 25/6])
        tie = select_reference_frames(frames([0., 1.]), 1)
        self.assertEqual(tie.frame_number.tolist(), [0])

    def test_short_and_single_frame_inputs_do_not_duplicate_frames(self):
        selected = select_reference_frames(frames([0., 1., 2.]), 30)
        self.assertEqual(selected.frame_number.tolist(), [0, 1, 2])
        self.assertTrue(selected.selection_requested_count.eq(30).all())
        single = select_reference_frames(frames([4.]), 30)
        self.assertEqual(single.frame_number.tolist(), [0])
        self.assertEqual(single.time_s.tolist(), [4.])

    def test_sparse_times_do_not_fill_collisions_with_other_frames(self):
        selected = select_reference_frames(frames([0., 0.01, 10.]), 4)
        self.assertEqual(selected.frame_number.tolist(), [1, 2])
        self.assertEqual(selected.selection_target_time_s.tolist(), [1.25, 8.75])

    def test_order_and_duplicate_dataframe_indexes_do_not_change_selection(self):
        data = frames(np.arange(61, dtype=float)).iloc[::-1]
        data.index = [0] * len(data)
        selected = select_reference_frames(data)
        self.assertEqual(selected.frame_number.tolist(), list(range(1, 60, 2)))

    def test_invalid_data_and_ranges_are_rejected(self):
        data = frames([0., 1., 2.])
        for count in (0, -1, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                select_reference_frames(data, count)
        for start, end in ((2., 1.), (-1., 1.), (0., 3.), (np.nan, 1.)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                select_reference_frames(data, start_time_s=start, end_time_s=end)
        for invalid in (data.iloc[:0], pd.concat([data, data.iloc[[0]]]),
                        frames([0., 0., 1.]), frames([0., np.nan, 2.]), frames([0., 2., 1.])):
            with self.assertRaises(ValueError):
                select_reference_frames(invalid)
        data.loc[1, "video_id"] = "other"
        with self.assertRaises(ValueError):
            select_reference_frames(data)
        with self.assertRaises(ValueError):
            select_reference_frames(frames([0., 1.]), start_time_s=0.1, end_time_s=0.2)


if __name__ == "__main__":
    unittest.main()
