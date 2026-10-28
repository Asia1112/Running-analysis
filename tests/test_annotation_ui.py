"""Streamlit session-flow tests; JS interaction is tested separately in Node."""
import tempfile
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.knee_angle_validation.annotation_storage import read_reference_annotations, save_reference_annotation
from src.knee_angle_validation.reference_annotation import build_reference_annotation


ANNOTATION_SCRIPT = '''
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import streamlit as st
import app
app.REFERENCE_DIR = Path(st.session_state.test_directory)
app._initialize_state()
def component(image, *, points, key, on_points_change):
    st.session_state.rendered_key = key
    st.session_state.rendered_points = [dict(p) for p in points]
    event = st.session_state.pop('points_event', None)
    if event is not None:
        st.session_state[key] = {'points': event}
        on_points_change()
    return SimpleNamespace(points=points)
with patch('app.annotate_image', side_effect=component), patch('app.read_analysis_frame', return_value=np.zeros((160,160,3), dtype=np.uint8)):
    app._annotation_view()
'''


class AnnotationUiTests(unittest.TestCase):
    def test_drafts_survive_reruns_and_frame_switches_but_new_session_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            at = AppTest.from_string(ANNOTATION_SCRIPT, default_timeout=30)
            at.session_state.test_directory = directory
            at.session_state.video_path = 'test.mov'
            at.session_state.annotation_session = 's1'
            at.session_state.selected_frames = pd.DataFrame([
                dict(video_id='v1', person_id='p1', frame_number=i, side='right') for i in (0, 1)])
            at.run()
            self.assertFalse(at.exception)
            first_key = at.session_state.rendered_key
            points = [dict(x=10., y=10.), dict(x=10., y=100.), dict(x=100., y=100.)]
            at.session_state.points_event = points[:1]
            at.run()
            at.run()
            self.assertEqual(at.session_state.rendered_points, points[:1])
            at.selectbox(key='annotation_frame').set_value(1).run()
            self.assertEqual(at.session_state.rendered_points, [])
            at.selectbox(key='annotation_frame').set_value(0).run()
            self.assertEqual(at.session_state.rendered_key, first_key)
            self.assertEqual(at.session_state.rendered_points, points[:1])
            at.session_state.points_event = points
            at.run()
            at.button(key='save_annotation').click().run()
            self.assertFalse(at.exception)
            at.run()
            self.assertTrue(at.button(key='save_annotation').disabled)
            at.text_input(key='annotation_session').set_value('s2').run()
            self.assertEqual(at.session_state.rendered_points, [])
            self.assertNotEqual(at.session_state.rendered_key, first_key)
            at.session_state.points_event = [dict(x=20., y=10.), *points[1:]]
            at.run()
            at.button(key='save_annotation').click().run()
            self.assertFalse(at.exception)
            from pathlib import Path
            saved = read_reference_annotations(Path(directory) / 'reference_annotations.csv')
            self.assertEqual(saved.annotation_session.tolist(), ['s1', 's2'])
            self.assertEqual(saved.hip_x_ref.tolist(), [10., 20.])
            self.assertTrue(saved.frame_image_sha256.str.len().eq(64).all())
            at.button(key='new_annotation_session').click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.session_state.rendered_points, [])
            self.assertTrue(at.text_input(key='annotation_session').value.startswith('session_'))

    def test_validation_uses_one_session_and_invalidates_previous_result(self):
        from pathlib import Path
        script = '''
from pathlib import Path
import streamlit as st
import app
app.REFERENCE_DIR = Path(st.session_state.test_directory)
app.RESULTS_DIR = Path(st.session_state.test_directory)
app._initialize_state()
app._validation_view(st.session_state.test_data)
'''
        with tempfile.TemporaryDirectory() as directory:
            frame = pd.Series(dict(video_id='v1', person_id='p1', frame_number=0, side='right'))
            points = [(0.,0.), (0.,10.), (10.,10.)]
            for session in ('s1', 's2'):
                save_reference_annotation(build_reference_annotation(frame, points, 'a1', session),
                                          Path(directory) / 'reference_annotations.csv')
            at = AppTest.from_string(script, default_timeout=30)
            at.session_state.test_directory = directory
            at.session_state.test_data = pd.DataFrame([{**frame.to_dict(),
                **{f'{p}_{a}_px': v for p, xy in zip(('hip','knee','ankle'), points) for a,v in zip(('x','y'),xy)}}])
            at.run()
            self.assertFalse(at.exception)
            at.button[0].click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.session_state.validation.annotation_session.tolist(), ['s1'])
            at.selectbox(key='validation_session').set_value(('a1','s2')).run()
            self.assertIsNone(at.session_state.validation)
            at.button[0].click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.session_state.validation.annotation_session.tolist(), ['s2'])


if __name__ == '__main__':
    unittest.main()
