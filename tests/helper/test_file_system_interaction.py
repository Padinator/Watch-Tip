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
    save_one_json_object_in_file,
    load_one_json_object_from_file,
)


# Define constants
BASIC_PATH = project_dir / "tests/jsons_files/test_file_system_interaction_jsons"


class TestFileSystemInteraction(unittest.TestCase):
    """
    Tests python file "file_system_interaction.py".

    Attributes
    ----------
    __test_object_path : Path
        Path to the test object JSON file.
    __test_object_in_separate_line : Path
        Path to the test object JSON file stored in a separate line.
    """

    def setUp(self):
        """
        Sets up the test environment for each test.
        """

        self.__test_object_path = BASIC_PATH / "test_object.json"
        self.__test_object_in_separate_line_path = BASIC_PATH / "test_object_in_separate_line.json"
        self.__test_one_prettied_json_object_path = BASIC_PATH / "test_one_prettied_json_object.json"

    def test_save_object_in_file(self) -> None:
        """
        Test saving a python object with pickle into file (correct case).
        """

        # Read expected JSON
        with open(self.__test_object_path) as json_file:
            test_json_file = json.load(json_file)

        # Create path for temporary file to store JSON into
        with NamedTemporaryFile(delete=False, mode="wb") as temp_file:
            temp_path = Path(temp_file.name)

        # Execute function and assert/check result
        try:
            save_object_in_file(temp_path, test_json_file)

            with open(temp_path, "rb") as file:
                loaded_data = pickle.load(file)

            self.assertEqual(loaded_data, test_json_file)
        finally:  # Delete temporary file
            temp_path.unlink()

    # ------------ Test function "load_object_from_file" ------------
    def test_load_object_from_file(self) -> None:
        """
        Tests loading a saved python object with pickle from file (ccorect case).
        """

        # Read expected JSON
        with open(self.__test_object_path) as json_file:
            test_json_file = json.load(json_file)

        # Create path for temporary file to store JSON into
        with NamedTemporaryFile(delete=False, mode="wb") as temp_file:
            temp_path = Path(temp_file.name)
            pickle.dump(test_json_file, temp_file)

        # Execute function and assert/check result
        try:
            loaded_data = load_object_from_file(temp_path)
            self.assertEqual(loaded_data, test_json_file)
        finally:  # Delete temporary file
            temp_path.unlink()

    # ------------ Test function "save_json_objects_in_file" ------------
    def test_save_json_objects_in_file(self) -> None:
        """
        Tests saving multiple "," separated JSON objects in file (correct case).
        """

        # Read expected JSON
        with open(self.__test_object_path) as json_file:
            test_json_file = json.load(json_file)

        # Create path for temporary file to store JSON into
        with NamedTemporaryFile(delete=False, mode="w", suffix=".json") as temp_file:
            temp_path = Path(temp_file.name)

        # Execute function and assert/check result
        try:
            save_json_objects_in_file(temp_path, test_json_file)

            with open(temp_path, "r", encoding="utf-8") as file:
                lines = file.readlines()

            result = [json.loads(line.strip()) for line in lines]
            self.assertEqual(result, test_json_file)
        finally:  # Delete temporary file
            temp_path.unlink()

    # ------------ Test function "load_json_objects_from_file" ------------
    def test_load_json_objects_from_file(self) -> None:
        """
        Tests loading JSON objects from file separated by "," (correct case).
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

        result = load_json_objects_from_file(self.__test_object_in_separate_line_path)
        self.assertEqual(result, expected_result)

    # ------------ Test function "save_json_one_object_in_file" ------------
    def test_save_one_json_object_in_file_1(self) -> None:
        """
        Tests saving a single JSON object (correct case).
        """

        # Read expected JSON
        with open(self.__test_one_prettied_json_object_path) as json_file:
            test_json_file = json.load(json_file)

        # Create path for temporary file to store JSON into
        with NamedTemporaryFile(delete=False, mode="w", suffix=".json") as temp_file:
            temp_path = Path(temp_file.name)

        # Execute function and assert/check result
        try:
            save_one_json_object_in_file(temp_path, test_json_file)

            with open(temp_path, "r", encoding="utf-8") as file:
                result = json.load(file)
            self.assertEqual(result, test_json_file)
        finally:  # Delete temporary file
            temp_path.unlink()

    # ------------ Test function "load_one_json_object_from_file" ------------
    def test_load_one_json_object_from_file_1(self):
        """
        Tests reading an object in JSON format (correct case).
        """

        expected_result = {"forename": "Jo", "last name": "Ghurt", "friends": ["Anna Nass", "Cam Eras", "Chris Tentum"]}

        result = load_one_json_object_from_file(self.__test_one_prettied_json_object_path)
        self.assertEqual(result, expected_result)
