import unittest
from unittest.mock import MagicMock

from pymongo import MongoClient
from pymongo.collection import Collection

from database.database_functions import (
    delete_one_by_attr,
    delete_one_by_id,
    get_entries_by_attr_from_database,
    get_mongo_db_specific_collection,
    get_one_by_attr,
    get_one_by_id,
    insert_one_element,
    read_all_entries_from_database_as_dict,
    update_one_by_attr,
    update_one_by_id,
)


class DatabaseFunctions(unittest.TestCase):

    def test_insert_one_element(self) -> None:
        """
        Test the method 'insert_one_element'
        in the file database_functions.py
        """

        mock_table = MagicMock()

        entity = {"_id": 1, "title": "Iron Man"}

        insert_one_element(mock_table, entity)

        mock_table.insert_one.assert_called_once_with(entity)

    def test_get_mongo_db_specific_collection(self) -> None:
        """
        Test the method 'get_mongo_db_specific_collection'
        in the file database_functions.py
        """

        mongo_client = MagicMock(spec=MongoClient)

        mock_database = MagicMock()
        mock_collection = MagicMock()

        mongo_client.__getitem__.return_value = mock_database
        mock_database.__getitem__.return_value = mock_collection

        collection_name = "test_collection"
        database_name = "test_database"

        result = get_mongo_db_specific_collection(
            mongo_client, collection_name, database_name
        )

        mongo_client.__getitem__.assert_called_with(database_name)
        mock_database.__getitem__.assert_called_with(collection_name)

        self.assertEqual(result, mock_collection)

    def test_read_all_entries_from_database_as_dict(self) -> None:
        """
        Test the method 'read_all_entries_from_database_as_dict'
        in the file database_functions.py
        """

        mock_table = MagicMock(spec=Collection)

        mock_table.find.return_value = [
            {"id": 1, "title": "Iron Man", "_id": "mock_id1"},
            {
                "id": 2,
                "title": "The Lord of the Rings: The two Towers",
                "_id": "mock_id2",
            },
        ]

        result = read_all_entries_from_database_as_dict(mock_table)

        expected_result = {
            1: {"title": "Iron Man"},
            2: {"title": "The Lord of the Rings: The two Towers"},
        }

        mock_table.find.assert_called_once()

        self.assertEqual(result, expected_result)

    def test_get_entries_by_attr_from_database(self) -> None:
        """
        Test the method 'get_entries_by_attr_from_database'
        in the file database_functions.py
        """

        mock_table = MagicMock(spec=Collection)

        attr = "type"
        attr_value = "movie"
        mock_table.find.return_value = [
            {"_id": 1, "type": "movie", "name": "Inception"},
            {"_id": 2, "type": "movie", "name": "The Matrix"},
        ]

        result = get_entries_by_attr_from_database(mock_table, attr, attr_value)

        expected_result = [
            {"_id": 1, "type": "movie", "name": "Inception"},
            {"_id": 2, "type": "movie", "name": "The Matrix"},
        ]

        mock_table.find.assert_called_once_with({attr: attr_value})

        self.assertEqual(result, expected_result)

    def test_get_one_by_attr(self) -> None:
        """
        Test the method 'get_one_by_attr'
        in the file database_functions.py
        """

        mock_table = MagicMock()

        mock_table.find_one.return_value = {
            "movie": "Iron Man",
            "genre": "Sci-Fi",
        }

        result = get_one_by_attr(mock_table, "movie", "Iron Man")

        self.assertEqual(result, {"movie": "Iron Man", "genre": "Sci-Fi"})

    def test_get_one_by_id(self) -> None:
        """
        Test the method 'get_one_by_id'
        in the file database_functions.py
        """

        mock_collection = MagicMock(spec=Collection)

        mock_collection.find_one.return_value = {
            "_id": 1,
            "type": "movie",
            "name": "Inception",
        }

        result = get_one_by_id(mock_collection, 1)

        self.assertEqual(
            result, {"_id": 1, "type": "movie", "name": "Inception"}
        )

    def test_update_one_by_attr(self) -> None:
        """
        Test the method 'update_one_by_attr'
        in the file database_functions.py
        """

        mock_collection = MagicMock(spec=Collection)

        mock_collection.find_one_and_update.return_value = {
            "_id": 1,
            "type": "movie",
            "name": "Interstellar",
        }

        result = update_one_by_attr(
            mock_collection, "type", "movie", "name", "Interstellar"
        )

        mock_collection.find_one_and_update.assert_called_with(
            {"type": "movie"}, {"$set": {"name": "Interstellar"}}
        )

        self.assertEqual(
            result, {"_id": 1, "type": "movie", "name": "Interstellar"}
        )

    def test_update_one_by_id(self) -> None:
        """
        Test the method 'update_one_by_id'
        in the file database_functions.py
        """

        mock_collection = MagicMock(spec=Collection)

        mock_collection.find_one_and_update.return_value = {
            "_id": 1,
            "type": "movie",
            "name": "Interstellar",
        }

        result = update_one_by_id(mock_collection, 1, "name", "Interstellar")

        mock_collection.find_one_and_update.assert_called_with(
            {"_id": 1}, {"$set": {"name": "Interstellar"}}
        )

        self.assertEqual(
            result, {"_id": 1, "type": "movie", "name": "Interstellar"}
        )

    def test_delete_one_by_attr(self) -> None:
        """
        Test the method 'delete_one_by_attr'
        in the file database_functions.py
        """

        mock_collection = MagicMock(spec=Collection)

        mock_collection.find_one_and_delete.return_value = {
            "_id": 1,
            "type": "movie",
            "name": "Inception",
        }

        result = delete_one_by_attr(mock_collection, "year", "2010")

        mock_collection.find_one_and_delete.assert_called_with({"year": "2010"})

        self.assertEqual(
            result, {"_id": 1, "type": "movie", "name": "Inception"}
        )

    def test_delete_one_by_id(self) -> None:
        """
        Test the method 'delete_one_by_id'
        in the file database_functions.py
        """

        mock_collection = MagicMock(spec=Collection)

        mock_collection.find_one_and_delete.return_value = {
            "_id": 1,
            "type": "movie",
            "name": "Inception",
        }

        result = delete_one_by_id(mock_collection, 1)

        mock_collection.find_one_and_delete.assert_called_with({"_id": 1})

        self.assertEqual(
            result, {"_id": 1, "type": "movie", "name": "Inception"}
        )
