import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.knee_angle_validation.overlay import draw_overlay, export_overlay_video, validate_overlay_data


def measurements(count=4):
    return pd.DataFrame([
        {"video_id": "v1", "person_id": "p1", "side": "right", "frame_number": i,
         "time_s": i / 30, "fps": 30, "image_width": 160, "image_height": 160,
         "hip_x_px": 40, "hip_y_px": 100, "knee_x_px": 80, "knee_y_px": 120,
         "ankle_x_px": 120, "ankle_y_px": 140}
        for i in range(count)
    ])


class OverlayTests(unittest.TestCase):
    def test_points_use_pixel_positions_without_mutating_source(self):
        frame = np.zeros((160, 160, 3), dtype=np.uint8)
        result = draw_overlay(frame, measurements().iloc[0])
        self.assertTrue(result[120, 80].any())
        self.assertFalse(frame.any())

    def test_missing_and_outside_points_are_not_connected(self):
        row = measurements().iloc[0].copy()
        row["knee_x_px"] = np.nan
        row["ankle_x_px"] = 200
        result = draw_overlay(np.zeros((160, 160, 3), dtype=np.uint8), row)
        self.assertFalse(result[120, 80].any())

    def test_rejects_ambiguous_frames_and_wrong_geometry(self):
        rows = measurements()
        with self.assertRaises(ValueError):
            validate_overlay_data(pd.concat([rows, rows.iloc[[0]]]))
        rows.loc[1, "side"] = "left"
        with self.assertRaises(ValueError):
            validate_overlay_data(rows)
        with self.assertRaises(ValueError):
            draw_overlay(np.zeros((120, 160, 3), dtype=np.uint8), measurements().iloc[0])

    def test_export_preserves_frames_and_rejects_truncated_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 160))
            self.assertTrue(writer.isOpened())
            for _ in range(4):
                writer.write(np.zeros((160, 160, 3), dtype=np.uint8))
            writer.release()
            target = Path(directory) / "overlay.mp4"
            export_overlay_video(source, measurements(), target, resize_height=0)
            capture = cv2.VideoCapture(str(target))
            try:
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 30, places=2)
                frames = []
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frames.append(frame)
                self.assertEqual(len(frames), 4)
                self.assertGreater(int(frames[0][120, 80].sum()), 100)
            finally:
                capture.release()
            with self.assertRaises(FileExistsError):
                export_overlay_video(source, measurements(), target, resize_height=0)
            incomplete = Path(directory) / "incomplete.mp4"
            with self.assertRaises(ValueError):
                export_overlay_video(source, measurements(3), incomplete, resize_height=0)
            self.assertFalse(incomplete.exists())


if __name__ == "__main__":
    unittest.main()
