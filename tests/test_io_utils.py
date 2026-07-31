import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.knee_angle_validation.io_utils import read_csv, write_csv


class IoUtilsTests(unittest.TestCase):
    def test_csv_round_trip(self):
        source = pd.DataFrame({"frame_number": [1, 2], "angle": [90.0, 91.5]})
        with tempfile.TemporaryDirectory() as directory:
            path = write_csv(source, Path(directory) / "data.csv")
            loaded = read_csv(path)
        pd.testing.assert_frame_equal(source, loaded)


if __name__ == "__main__":
    unittest.main()
