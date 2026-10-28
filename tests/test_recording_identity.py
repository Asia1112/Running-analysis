"""Recording provenance and isolated Streamlit storage regression tests."""

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.knee_angle_validation.annotation_storage import save_reference_annotation
from src.knee_angle_validation.provenance import (
    PROVENANCE_KEYS, recording_metadata, require_source_match,
)
from src.knee_angle_validation.reference_annotation import build_reference_annotation
from src.knee_angle_validation.validation import merge_with_reference


def data_for(content=b"video A", analysis_id="a" * 32):
    return pd.DataFrame([{
        "video_id": "video_001", "person_id": "001", "side": "left", "frame_number": 0,
        "analysis_id": analysis_id, "source_sha256": hashlib.sha256(content).hexdigest(),
        "crop_mode": "none", "resize_height": 720, "image_width": 160, "image_height": 160,
        "hip_x_px": 0., "hip_y_px": 0., "knee_x_px": 0., "knee_y_px": 10.,
        "ankle_x_px": 10., "ankle_y_px": 10.,
    }])


EXTRACTION_SCRIPT = '''
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import streamlit as st
import app
from src.knee_angle_validation.provenance import file_sha256
app.RAW_DIR = Path(st.session_state.test_directory) / 'raw'
app.RAW_DIR.mkdir(exist_ok=True)
app.RESULTS_DIR = Path(st.session_state.test_directory) / 'results'
app._initialize_state()
class Upload(io.BytesIO):
    def __init__(self, content, name, file_id):
        super().__init__(content)
        self.name, self.file_id, self.size = name, file_id, len(content)
def upload(label, **kwargs):
    if kwargs.get('key') == 'extracted_csv':
        value = st.session_state.get('csv_bytes')
        return Upload(value, 'points.csv', 'csv') if value else None
    content = st.session_state.get('video_bytes')
    return Upload(content, 'same_name.MOV', st.session_state.get('video_token', 'A')) if content else None
def process(source, output, video_id, person_id, side, *, crop_mode, analysis_id):
    data = pd.DataFrame([dict(video_id=video_id, person_id=person_id, side=side, frame_number=0,
        analysis_id=analysis_id, source_sha256=file_sha256(source), crop_mode=crop_mode, resize_height=720)])
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    return SimpleNamespace(data=data, frames=1, pose_found_pct=100.)
with patch('app.st.file_uploader', side_effect=upload), patch('app.process_video', side_effect=process):
    app._extraction_view()
st.session_state.geometry_error = None
if st.session_state.extracted is not None:
    try:
        app._verified_geometry(st.session_state.extracted)
    except ValueError as error:
        st.session_state.geometry_error = str(error)
'''


class RecordingIdentityTests(unittest.TestCase):
    def test_image_views_block_unverified_csv_before_reading_any_frame(self):
        script = '''
from pathlib import Path
from unittest.mock import patch
import streamlit as st
import app
app.RESULTS_DIR = Path(st.session_state.test_directory)
app._initialize_state()
st.session_state.active_analysis_id = 'a' * 32
st.session_state.source_sha256 = 'e' * 64
st.session_state.selected_frames = st.session_state.test_data
st.session_state.video_path = st.session_state.test_video
with patch('app.read_analysis_frame', side_effect=AssertionError('wrong frame read')):
    app._annotation_view()
    app._overlay_view(st.session_state.test_data)
'''
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'video.mov'
            source.write_bytes(b'wrong video')
            for data in (data_for(), data_for().drop(columns=list(PROVENANCE_KEYS))):
                at = AppTest.from_string(script, default_timeout=30)
                at.session_state.test_directory = directory
                at.session_state.test_video = str(source)
                at.session_state.test_data = data
                at.run()
                self.assertFalse(at.exception)
                self.assertEqual(len(at.error), 2)

    def test_app_starts_with_no_active_analysis(self):
        at = AppTest.from_file('app.py', default_timeout=30).run()
        self.assertFalse(at.exception)
        self.assertIsNone(at.session_state.extracted)
        self.assertIsNone(at.session_state.summary)

    def test_provenance_is_retained_in_reference_and_comparison(self):
        extracted = data_for()
        record = build_reference_annotation(extracted.iloc[0], [(0, 0), (0, 10), (10, 10)], "a1", "s1")
        result = merge_with_reference(extracted, pd.DataFrame([record]))
        for key in PROVENANCE_KEYS:
            self.assertEqual(record[key], extracted.iloc[0][key])
            self.assertEqual(result.iloc[0][key], record[key])

    def test_matching_frame_ids_cannot_hide_a_different_video_or_geometry(self):
        extracted = data_for()
        reference = pd.DataFrame([build_reference_annotation(extracted.iloc[0], [(0, 0), (0, 10), (10, 10)], "a1", "s1")])
        for key, value in (("source_sha256", "c" * 64), ("analysis_id", "b" * 32),
                           ("crop_mode", "right"), ("resize_height", 1080)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                merge_with_reference(extracted, reference.assign(**{key: value}))
        with self.assertRaises(ValueError):
            merge_with_reference(extracted, reference.drop(columns=list(PROVENANCE_KEYS)))

    def test_partial_metadata_legacy_and_unsafe_ids(self):
        extracted = data_for()
        self.assertIsNone(recording_metadata(extracted.drop(columns=list(PROVENANCE_KEYS))))
        with self.assertRaises(ValueError):
            require_source_match(extracted.drop(columns=list(PROVENANCE_KEYS)), extracted.source_sha256.iloc[0])
        for invalid in (extracted.drop(columns="crop_mode"), extracted.assign(analysis_id="../outside"),
                        extracted.assign(resize_height=-1), extracted.assign(source_sha256="")):
            with self.assertRaises(ValueError):
                recording_metadata(invalid)
        with self.assertRaises(ValueError):
            require_source_match(extracted, "d" * 64)

    def test_annotation_file_cannot_mix_analyses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.csv"
            record = build_reference_annotation(data_for().iloc[0], [(0, 0), (0, 10), (10, 10)], "a1", "s1")
            save_reference_annotation(record, path)
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                save_reference_annotation({**record, "analysis_id": "b" * 32, "annotation_session": "s2"}, path)
            self.assertEqual(path.read_bytes(), before)

    def test_same_named_uploads_get_different_paths_without_overwriting(self):
        import app
        class Upload(io.BytesIO):
            name = "same_name.MOV"
        with tempfile.TemporaryDirectory() as directory, patch("app.RAW_DIR", Path(directory)):
            first = app._uploaded_path(Upload(b"video A"))
            second = app._uploaded_path(Upload(b"video B"))
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), b"video A")
            self.assertEqual(second.read_bytes(), b"video B")
            self.assertEqual(app._uploaded_path(Upload(b"video A")), first)
            first.write_bytes(b"damaged")
            with self.assertRaises(ValueError):
                app._uploaded_path(Upload(b"video A"))

    def test_switching_source_or_controls_clears_all_dependent_state(self):
        with tempfile.TemporaryDirectory() as directory:
            at = AppTest.from_string(EXTRACTION_SCRIPT, default_timeout=30)
            at.session_state.test_directory = directory
            at.session_state.video_bytes = b"video A"
            at.run()
            self.assertFalse(at.exception)
            at.button[0].click().run()
            self.assertFalse(at.exception)
            first_id = at.session_state.active_analysis_id
            path_a = Path(at.session_state.video_path)
            output_a = Path(directory) / "results" / "analyses" / first_id / "extracted_landmarks.csv"
            before = output_a.read_bytes()
            for key in ('selected_frames', 'validation', 'summary'):
                at.session_state[key] = pd.DataFrame([{'old': 1}])
            at.session_state.annotation_drafts = {'old': {'points': [1]}}
            at.session_state.overlay_video = ('old', 'old.mp4')
            at.run()
            self.assertIsNotNone(at.session_state.validation)
            at.session_state.video_bytes = b"video B"
            at.session_state.video_token = 'B'
            at.run()
            self.assertFalse(at.exception)
            for key in ('extracted', 'selected_frames', 'validation', 'summary', 'active_analysis_id'):
                self.assertIsNone(at.session_state[key])
            self.assertEqual(at.session_state.annotation_drafts, {})
            self.assertNotIn('overlay_video', at.session_state)
            self.assertEqual(path_a.read_bytes(), b"video A")
            at.button[0].click().run()
            second_id = at.session_state.active_analysis_id
            self.assertNotEqual(second_id, first_id)
            self.assertEqual(output_a.read_bytes(), before)
            self.assertEqual(len(list((Path(directory) / 'results' / 'analyses').iterdir())), 2)
            for key, value in (('input_video_id', 'video_002'), ('input_person_id', '002')):
                at.text_input(key=key).set_value(value).run()
                self.assertIsNone(at.session_state.extracted)
                at.button[0].click().run()
                self.assertFalse(at.exception)
            at.selectbox(key='input_crop').set_value('right').run()
            self.assertIsNone(at.session_state.extracted)

    def test_csv_import_does_not_authorize_another_video_and_is_stable_on_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            at = AppTest.from_string(EXTRACTION_SCRIPT, default_timeout=30)
            at.session_state.test_directory = directory
            at.session_state.video_bytes = b"video B"
            at.session_state.csv_bytes = data_for().to_csv(index=False).encode()
            at.run()
            self.assertFalse(at.exception)
            self.assertIn('SHA-256', at.session_state.geometry_error)
            self.assertEqual(at.session_state.active_analysis_id, 'a' * 32)
            at.session_state.selected_frames = data_for()
            at.run()
            self.assertIsNotNone(at.session_state.selected_frames)
            self.assertFalse(at.exception)
            # A modified CSV cannot overwrite the original under the same analysis ID.
            before = (Path(directory) / 'results' / 'analyses' / ('a' * 32) / 'extracted_landmarks.csv').read_bytes()
            at.session_state.csv_bytes = data_for().assign(hip_x_px=99.).to_csv(index=False).encode()
            at.run()
            self.assertFalse(at.exception)
            self.assertIsNone(at.session_state.extracted)
            self.assertEqual((Path(directory) / 'results' / 'analyses' / ('a' * 32) / 'extracted_landmarks.csv').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
