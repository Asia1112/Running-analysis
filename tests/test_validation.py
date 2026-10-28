import unittest
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.knee_angle_validation.statistics import error_statistics
from src.knee_angle_validation.validation import merge_with_reference
from src.knee_angle_validation.reference_annotation import build_reference_annotation


def measurements():
    identity = dict(video_id="v1", person_id="p1", frame_number=10, side="left")
    points = [(0., 0.), (0., 10.), (10., 10.)]
    extracted = pd.DataFrame([{**identity, **{
        f"{point}_{axis}_px": value for point, xy in zip(("hip", "knee", "ankle"), points)
        for axis, value in zip(("x", "y"), xy)}}])
    reference = pd.DataFrame([build_reference_annotation(pd.Series(identity), points, "a1", "s1")])
    return extracted, reference


class ValidationTests(unittest.TestCase):
    def test_duplicates_on_either_side_cannot_multiply_observations(self):
        extracted, reference = measurements()
        for left_count, right_count, label in ((2, 1, "MediaPipe"), (1, 2, "Referencja"), (2, 2, "MediaPipe")):
            with self.subTest(left=left_count, right=right_count):
                with self.assertRaisesRegex(ValueError, label + ": powielone klucze"):
                    merge_with_reference(pd.concat([extracted] * left_count), pd.concat([reference] * right_count))

    def test_reference_requires_one_annotator_and_session_in_core_api(self):
        extracted, reference = measurements()
        references = pd.concat([reference, reference.assign(annotation_session="s2"),
                                reference.assign(annotator_id="a2")], ignore_index=True)
        for kwargs in ({}, {"annotator_id": "a1"}, {"session": "s1"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                merge_with_reference(extracted, references, **kwargs)
        result = merge_with_reference(extracted, references, annotator_id="a1", session="s2")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.annotation_session.tolist(), ["s2"])
        self.assertEqual(result.annotator_id.tolist(), ["a1"])
        with self.assertRaises(ValueError):
            merge_with_reference(extracted, references, annotator_id="a1", session="missing")

    def test_empty_or_missing_reference_identity_is_rejected(self):
        extracted, reference = measurements()
        for key in ("annotator_id", "annotation_session"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    merge_with_reference(extracted, reference.drop(columns=key))
                for value in (None, "", "  "):
                    with self.assertRaises(ValueError):
                        merge_with_reference(extracted, reference.assign(**{key: value}))

    def test_invalid_join_keys_are_rejected_on_either_side(self):
        extracted, reference = measurements()
        changes = [(key, value) for key in ("video_id", "person_id", "side") for value in (None, " ")]
        changes += [("frame_number", value) for value in (None, float("inf"), -1, 1.5, "bad")]
        changes += [("side", "unknown")]
        for key, value in changes:
            for side in ("extracted", "reference"):
                with self.subTest(key=key, value=value, side=side), self.assertRaises(ValueError):
                    merge_with_reference(extracted.assign(**{key: value}) if side == "extracted" else extracted,
                                         reference.assign(**{key: value}) if side == "reference" else reference)

    def test_normalization_does_not_hide_duplicates(self):
        extracted, reference = measurements()
        duplicate = extracted.assign(video_id=" v1 ", frame_number="10")
        with self.assertRaisesRegex(ValueError, "MediaPipe: powielone klucze"):
            merge_with_reference(pd.concat([extracted, duplicate]), reference)
        result = merge_with_reference(duplicate, reference)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.frame_number.tolist(), [10])
        self.assertEqual(duplicate.video_id.tolist(), [" v1 "])

    def test_all_join_keys_distinguish_independent_measurements(self):
        extracted, reference = measurements()
        variants = [{}, {"video_id": "v2"}, {"person_id": "p2"}, {"frame_number": 11}, {"side": "right"}]
        result = merge_with_reference(pd.concat([extracted.assign(**v) for v in variants]),
                                      pd.concat([reference.assign(**v) for v in variants]))
        self.assertEqual(len(result), 5)
        self.assertFalse(result.duplicated(["video_id", "person_id", "frame_number", "side"]).any())

    def test_duplicate_outside_reference_sample_is_also_rejected(self):
        extracted, reference = measurements()
        extra = extracted.assign(frame_number=999)
        with self.assertRaisesRegex(ValueError, "MediaPipe: powielone klucze"):
            merge_with_reference(pd.concat([extracted, extra, extra]), reference)

    def test_cli_rejects_duplicates_without_replacing_output(self):
        from analyze_run_csv import main
        extracted, reference = measurements()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.concat([extracted, extracted]).to_csv(root / "extracted.csv", index=False)
            reference.to_csv(root / "reference.csv", index=False)
            output, summary = root / "results.csv", root / "summary.csv"
            output.write_text("previous results", encoding="utf-8")
            summary.write_text("previous summary", encoding="utf-8")
            argv = ["analyze_run_csv.py", "--extracted", str(root / "extracted.csv"), "--reference",
                    str(root / "reference.csv"), "--results", str(output), "--summary", str(summary)]
            with patch("sys.argv", argv), patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "previous results")
            self.assertEqual(summary.read_text(encoding="utf-8"), "previous summary")

    def test_join_errors_distances_and_statistics(self):
        extracted = pd.DataFrame([{
            "video_id": "v1", "person_id": "p1", "frame_number": 10, "side": "left",
            "knee_angle_raw": 123.0, "hip_x_px": 0.0, "hip_y_px": 0.0,
            "knee_x_px": 0.0, "knee_y_px": 10.0,
            "ankle_x_px": 10.0, "ankle_y_px": 10.0 + 10.0 * math.tan(math.radians(5)),
        }])
        reference = pd.DataFrame([{
            "video_id": "v1", "person_id": "p1", "frame_number": 10, "side": "left",
            "knee_angle_reference": 90.0, "hip_x_ref": 0.0, "hip_y_ref": 0.0,
            "annotator_id": "a1", "annotation_session": "s1",
            "knee_x_ref": 0.0, "knee_y_ref": 10.0, "ankle_x_ref": 10.0, "ankle_y_ref": 10.0,
        }])
        merged = merge_with_reference(extracted, reference)
        self.assertAlmostEqual(merged.loc[0, "angle_mediapipe"], 85.0)
        self.assertAlmostEqual(merged.loc[0, "angle_reference"], 90.0)
        self.assertAlmostEqual(merged.loc[0, "error_signed"], -5.0)
        self.assertAlmostEqual(merged.loc[0, "error_internal_signed"], 5.0)
        self.assertEqual(merged.loc[0, "angle_definition"], "knee_flexion_2d_deg")
        self.assertEqual(merged.loc[0, "hip_distance_px"], 0.0)
        stats = error_statistics(merged)
        self.assertAlmostEqual(stats["mae"], 5.0)
        self.assertAlmostEqual(stats["rmse"], 5.0)
        self.assertAlmostEqual(stats["bias"], -5.0)


if __name__ == "__main__":
    unittest.main()
