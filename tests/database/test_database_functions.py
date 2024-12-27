import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pymongo import MongoClient
from pymongo.collection import Collection

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[2]
sys.path.append(str(project_dir))

from database.database_functions import (
    DBConnector,
    DBModifier,
    delete_one_by_attr,
    delete_one_by_id,
    get_entries_by_attr_from_database,
    get_one_by_attr,
    get_one_by_id,
    get_table_from_database,
    insert_one_element,
    read_all_entries_from_database_as_dict,
    update_one_by_attr,
    update_one_by_id,
)


class DatabaseFunctions(unittest.TestCase):

    def setUp(self):
        self.mock_db_modifier = MagicMock(spec=DBModifier)

    # @patch("database.database_functions.DBConnector.")
    def test_get_table_from_database(self) -> None:
        """
        Test the method 'get_table_from_database'
        in the file database_functions.py
        """

        # Define variables
        collection_name = "test_collection"
        database_name = "test_database"

        # Create mocks
        mock_db_connector = MagicMock()
        mongo_client = MagicMock(spec=MongoClient)
        mock_database = MagicMock()
        mock_collection = MagicMock()

        # Connect mocks
        mock_db_connector.__connector_object = mongo_client
        mongo_client.__getitem__.return_value = mock_database
        mock_database.__getitem__.return_value = mock_collection
        mock_db_connector.get_table_from_database.return_value = self.mock_db_modifier

        # Call function to test
        result = get_table_from_database(mock_db_connector, collection_name, database_name)

        # Assert passed arguments
        mock_db_connector.get_table_from_database.assert_called_with(
            collection_name=collection_name, database_name=database_name
        )

        # Assert result
        self.assertEqual(result, self.mock_db_modifier)

    def test_insert_one_element(self) -> None:
        """
        Test the method 'insert_one_element'
        in the file database_functions.py
        """

        entity = {"id": 1, "title": "Iron Man"}

        insert_one_element(self.mock_db_modifier, entity)

        self.mock_db_modifier.return_value.insert_one.assert_called_once_with(entity)

    def test_read_all_entries_from_database_as_dict(self) -> None:
        """
        Test the method 'read_all_entries_from_database_as_dict'
        in the file database_functions.py
        """

        self.mock_db_modifier.return_value.find.return_value = [
            {"id": 1, "title": "Iron Man", "_id": "mock_id1"},
            {"id": 2, "title": "The Lord of the Rings: The Two Towers", "_id": "mock_id2"},
        ]

        result = read_all_entries_from_database_as_dict(self.mock_db_modifier)

        expected_result = {
            1: {"title": "Iron Man"},
            2: {"title": "The Lord of the Rings: The Two Towers"},
        }

        self.assertEqual(result, expected_result)

        self.mock_db_modifier.return_value.find.assert_called_once()

    def test_get_entries_by_attr_from_database(self) -> None:
        """
        Test the method 'get_entries_by_attr_from_database'
        in the file database_functions.py
        """

        attr = "type"
        attr_value = "movie"
        self.mock_db_modifier.return_value.find.return_value = [
            {"id": 1, "type": "movie", "name": "Inception"},
            {"id": 2, "type": "movie", "name": "The Matrix"},
        ]

        result = get_entries_by_attr_from_database(self.mock_db_modifier, attr, attr_value)

        expected_result = [
            {"id": 1, "type": "movie", "name": "Inception"},
            {"id": 2, "type": "movie", "name": "The Matrix"},
        ]

        self.mock_db_modifier.return_value.find.assert_called_once_with({attr: attr_value})

        self.assertEqual(result, expected_result)

    def test_get_one_by_attr(self) -> None:
        """
        Test the method 'get_one_by_attr'
        in the file database_functions.py
        """

        self.mock_db_modifier.return_value.find_one.return_value = {
            "movie": "Iron Man",
            "genre": "Sci-Fi",
        }

        result = get_one_by_attr(self.mock_db_modifier, "movie", "Iron Man")

        self.assertEqual(result, {"movie": "Iron Man", "genre": "Sci-Fi"})

    def test_get_one_by_id(self) -> None:
        """
        Test the method 'get_one_by_id'
        in the file database_functions.py
        """

        self.mock_db_modifier.return_value.find_one.return_value = {
            "id": 1,
            "type": "movie",
            "name": "Inception",
        }

        result = get_one_by_id(self.mock_db_modifier, 1)

        self.assertEqual(result, {"id": 1, "type": "movie", "name": "Inception"})

    def test_update_one_by_attr(self) -> None:
        """
        Test the method 'update_one_by_attr'
        in the file database_functions.py
        """

        self.mock_db_modifier.return_value.find_one_and_update.return_value = {
            "id": 1,
            "type": "movie",
            "name": "Interstellar",
        }

        result = update_one_by_attr(self.mock_db_modifier, "type", "movie", "name", "Interstellar")

        self.mock_db_modifier.return_value.find_one_and_update.assert_called_with(
            {"type": "movie"}, {"$set": {"name": "Interstellar"}}, return_document=True
        )

        self.assertEqual(result, {"id": 1, "type": "movie", "name": "Interstellar"})

    def test_update_one_by_id(self) -> None:
        """
        Test the method 'update_one_by_id'
        in the file database_functions.py
        """

        self.mock_db_modifier.return_value.find_one_and_update.return_value = {
            "id": 1,
            "type": "movie",
            "name": "Interstellar",
        }

        result = update_one_by_id(self.mock_db_modifier, 1, "name", "Interstellar")

        self.mock_db_modifier.return_value.find_one_and_update.assert_called_with(
            {"id": 1}, {"$set": {"name": "Interstellar"}}, return_document=True
        )

        self.assertEqual(result, {"id": 1, "type": "movie", "name": "Interstellar"})

    def test_delete_one_by_attr(self) -> None:
        """
        Test the method 'delete_one_by_attr'
        in the file database_functions.py
        """

        self.mock_db_modifier.return_value.find_one_and_delete.return_value = {
            "id": 1,
            "type": "movie",
            "name": "Inception",
        }

        result = delete_one_by_attr(self.mock_db_modifier, "year", "2010")

        self.mock_db_modifier.return_value.find_one_and_delete.assert_called_with({"year": "2010"})

        self.assertEqual(result, {"id": 1, "type": "movie", "name": "Inception"})

    def test_delete_one_by_id(self) -> None:
        """
        Test the method 'delete_one_by_id'
        in the file database_functions.py
        """

        self.mock_db_modifier.return_value.find_one_and_delete.return_value = {
            "id": 1,
            "type": "movie",
            "name": "Inception",
        }

        result = delete_one_by_id(self.mock_db_modifier, 1)

        self.mock_db_modifier.return_value.find_one_and_delete.assert_called_with({"id": 1})

        self.assertEqual(result, {"id": 1, "type": "movie", "name": "Inception"})
