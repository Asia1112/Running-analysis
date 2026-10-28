import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.knee_angle_validation.annotation_storage import (
    annotation_exists, annotation_identity, annotation_widget_key,
    read_reference_annotations, save_reference_annotation,
    select_reference_session,
)
from src.knee_angle_validation.reference_annotation import build_reference_annotation


def record(session="001", annotator="a1", **changes):
    frame = pd.Series({"video_id": "v1", "person_id": "p1", "frame_number": 10, "side": "right", **changes})
    return build_reference_annotation(frame, [(0., 0.), (0., 10.), (10., 10.)], annotator, session)


class AnnotationStorageTests(unittest.TestCase):
    def test_two_sessions_are_preserved_and_duplicate_cannot_replace_first(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.csv"
            save_reference_annotation(record(), path)
            first = path.read_bytes()
            changed = record()
            changed["hip_x_ref"] = 100.
            with self.assertRaises(ValueError):
                save_reference_annotation(changed, path)
            self.assertEqual(path.read_bytes(), first)
            save_reference_annotation(record("002"), path)
            loaded = read_reference_annotations(path)
            self.assertEqual(loaded.annotation_session.tolist(), ["001", "002"])
            self.assertEqual(loaded.hip_x_ref.tolist(), [0., 0.])

    def test_person_side_and_annotator_are_part_of_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.csv"
            records = [record(), record(annotator="a2"), record(person_id="p2"), record(side="left")]
            for item in records:
                save_reference_annotation(item, path)
            loaded = read_reference_annotations(path)
            self.assertEqual(len(loaded), 4)
            self.assertTrue(all(annotation_exists(loaded, item) for item in records))

    def test_atomic_failure_keeps_existing_file_and_cleans_up_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.csv"
            save_reference_annotation(record(), path)
            before = path.read_bytes()
            with patch('src.knee_angle_validation.annotation_storage.os.replace', side_effect=OSError("disk")):
                with self.assertRaises(OSError):
                    save_reference_annotation(record("002"), path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["reference.csv"])

    def test_concurrent_write_is_not_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.csv"
            lock = path.with_name(path.name + '.lock')
            lock.touch()
            with self.assertRaises(ValueError):
                save_reference_annotation(record(), path)
            self.assertFalse(path.exists())
            self.assertTrue(lock.exists())

    def test_widget_key_is_separate_for_session_side_person_and_image(self):
        identity = annotation_identity(record(), "a1", "001")
        key = annotation_widget_key(identity, "image1")
        self.assertEqual(annotation_widget_key(identity, "image1"), key)
        for field, value in (("annotation_session", "002"), ("annotator_id", "a2"),
                             ("person_id", "p2"), ("side", "left"), ("video_id", "v2"), ("frame_number", 11)):
            self.assertNotEqual(annotation_widget_key({**identity, field: value}, "image1"), key)
        self.assertNotEqual(annotation_widget_key(identity, "image2"), key)

    def test_invalid_identity_and_degenerate_points_are_rejected(self):
        frame = pd.Series(dict(video_id="v1", person_id="p1", frame_number=0, side="left"))
        for annotator, session in (("", "s1"), ("a1", "  ")):
            with self.assertRaises(ValueError):
                build_reference_annotation(frame, [(0, 0), (1, 1), (2, 0)], annotator, session)
        with self.assertRaises(ValueError):
            build_reference_annotation(frame, [(0, 0), (0, 0), (2, 0)], "a1", "s1")

    def test_primary_validation_cannot_pool_repeated_sessions(self):
        data = pd.DataFrame([record("001"), record("002")])
        with self.assertRaises(ValueError):
            select_reference_session(data)
        selected = select_reference_session(data, 'a1', '002')
        self.assertEqual(selected.annotation_session.tolist(), ['002'])
        self.assertEqual(len(select_reference_session(selected)), 1)
        with self.assertRaises(ValueError):
            select_reference_session(data, 'missing', '002')


if __name__ == '__main__':
    unittest.main()
