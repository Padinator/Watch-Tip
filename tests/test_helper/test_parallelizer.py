import sys
import unittest

from pathlib import Path

# ---------- Import own python files ----------
project_dir = Path(__file__).parents[2]
sys.path.append(str(project_dir))

from helper.parallelizer import parallelize_task_and_return_results


class TestApiRequester(unittest.TestCase):
    
    # TODO: This class need still some tests
    def test_parallelize_task_and_return_results(self):
        pass