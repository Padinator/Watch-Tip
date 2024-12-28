import json
import pickle
import sys
import unittest

from pathlib import Path
from tempfile import NamedTemporaryFile

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[2]
sys.path.append(str(project_dir))

from helper.file_system_interaction import (
    save_object_in_file,
    load_object_from_file,
    save_json_objects_in_file,
    load_json_objects_from_file,
)


# Define constants
BASIC_PATH = project_dir / "tests/jsons_files/test_file_system_interaction_jsons"


class TestFileSystemInteraction(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment for each test.

        Attributes
        ----------
        __test_object_path : Path
            Path to the test object JSON file.
        __test_object_in_separate_line : Path
            Path to the test object JSON file stored in a separate line.
        """

        self.__test_object_path = BASIC_PATH / "test_object.json"
        self.__test_object_in_separate_line = BASIC_PATH / "test_object_in_separate_line.json"

    def test_save_object_in_file(self) -> None:
        """
        Test the `save_object_in_file` function.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        with open(self.__test_object_path) as json_file:
            test_json_file = json.load(json_file)

        with NamedTemporaryFile(delete=False, mode="wb") as temp_file:
            temp_path = Path(temp_file.name)

        try:
            save_object_in_file(temp_path, test_json_file)

            with open(temp_path, "rb") as file:
                loaded_data = pickle.load(file)

            self.assertEqual(loaded_data, test_json_file)
        finally:
            temp_path.unlink()

    def test_load_object_from_file(self) -> None:
        """
        Tests the `load_object_from_file` function.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        with open(self.__test_object_path) as json_file:
            test_json_file = json.load(json_file)

        with NamedTemporaryFile(delete=False, mode="wb") as temp_file:
            temp_path = Path(temp_file.name)
            pickle.dump(test_json_file, temp_file)

        try:
            loaded_data = load_object_from_file(temp_path)
            self.assertEqual(loaded_data, test_json_file)
        finally:
            temp_path.unlink()

    def test_save_json_objects_in_file(self) -> None:
        """
        Test the `save_json_objects_in_file` function.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        with open(self.__test_object_path) as json_file:
            test_json_file = json.load(json_file)

        with NamedTemporaryFile(delete=False, mode="w", suffix=".json") as temp_file:
            temp_path = Path(temp_file.name)

        try:
            save_json_objects_in_file(temp_path, test_json_file)

            with open(temp_path, "r", encoding="utf-8") as file:
                lines = file.readlines()

            result = [json.loads(line.strip()) for line in lines]
            self.assertEqual(result, test_json_file)
        finally:
            temp_path.unlink()

    def test_load_json_objects_from_file(self) -> None:
        """
        Test the `load_json_objects_from_file` function.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        expected_result = [
            {"title": "Iron Man", "genre": "Sci-Fi"},
            {"title": "Inception", "genre": "Sci-Fi"},
            {
                "title": "The Lord of the Rings: The fellowship of the Ring",
                "genre": "Fantasy",
            },
            {"title": "The Godfather", "genre": "Crime"},
        ]

        result = load_json_objects_from_file(self.__test_object_in_separate_line)
        self.assertEqual(result, expected_result)
