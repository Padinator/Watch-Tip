import sys
import unittest

from pathlib import Path

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[2]
sys.path.append(str(project_dir))


class TestApiRequester(unittest.TestCase):

    # TODO: This class need still some tests
    def test_parallelize_task_and_return_results(self):
        pass
