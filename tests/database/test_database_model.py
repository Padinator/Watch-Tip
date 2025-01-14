import unittest
import sys

from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[2]
sys.path.append(str(project_dir))

from database.database_functions import DBModifier
from database.model import DatabaseModel


class TestDatabaseModel(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment for DatabaseModel tests.

        This method is called before each test to create a mock DBModifier and
        initialize a DatabaseModel instance with a test database and collection.
        The mock DBModifier is assigned to the model's _db_table_modifier attribute
        to simulate database operations.

        Attributes
        ----------
        mock_db_modifier : unittest.mock.MagicMock
            A mock object for simulating DBModifier behavior.
        model : DatabaseModel
            An instance of DatabaseModel initialized with test database and collection.
        """

        self.mock_db_modifier = MagicMock(spec=DBModifier)
        self.model = DatabaseModel("test_db", "test_collection")
        self.model._db_table_modifier = self.mock_db_modifier

    @patch("database.database_functions.insert_one_element")
    def test_insert_one(self, insert_one_element) -> None:
        """
        Test the insert_one method of the database model.

        Parameters
        ----------
        insert_one_element : unittest.mock.MagicMock
            Mock object for the 'insert_one' method.

        Returns
        -------
        None
        """

        insert_one_element.return_value = {
            "id": 1,
            "title": "Iron Man",
            "_id": "mock_id1",
        }

        entity = ({"id": 1, "title": "Iron Man", "_id": "mock_id1"},)

        result = self.model.insert_one(entity)

        insert_one_element.assert_called_once_with(db_table_modifier=self.mock_db_modifier, enitity=entity)

        self.assertEqual(result, {"id": 1, "title": "Iron Man", "_id": "mock_id1"})

    @patch("database.database_functions.get_all_entries_from_database")
    def test_get_all(self, get_all_entries_from_database) -> None:
        """
        Test the 'get_all' method in the model.py file.

        Parameters
        ----------
        get_all_entries_from_database : unittest.mock.MagicMock
            Mock object for the 'get_all' method.

        Returns
        -------
        None
        """

        get_all_entries_from_database.return_value = {
            1: {"name": "Actor 1", "age": 30},
            2: {"name": "Actor 2", "age": 40},
        }

        result = self.model.get_all()

        get_all_entries_from_database.assert_called_once_with(db_table_modifier=self.mock_db_modifier)

        self.assertEqual(
            result,
            {1: {"name": "Actor 1", "age": 30}, 2: {"name": "Actor 2", "age": 40}},
        )

    @patch("database.database_functions.get_one_by_attr")
    def test_get_one_by_attr(self, get_one_by_attr) -> None:
        """
        Test the method 'get_one_by_attr' in the file model.py.

        Parameters
        ----------
        get_one_by_attr : unittest.mock.MagicMock
            Mock object for the 'get_one_by_attr' method.

        Returns
        -------
        None
        """

        get_one_by_attr.return_value = {"name": "Actor 1", "age": 30}

        attr = "name"
        attr_value = "Actor 1"

        result = self.model.get_one_by_attr(attr, attr_value)

        get_one_by_attr.assert_called_once_with(db_table_modifier=self.mock_db_modifier, attr=attr, attr_value=attr_value)

        self.assertEqual(result, {"name": "Actor 1", "age": 30})

    @patch("database.database_functions.get_one_by_id")
    def test_get_one_by_id(self, get_one_by_id) -> None:
        """
        Test the 'get_one_by_id' method in the model.py file.

        Parameters
        ----------
        get_one_by_id : unittest.mock.MagicMock
            Mock object for the 'get_one_by_id' method.

        Returns
        -------
        None
        """

        get_one_by_id.return_value = {"_id": 1, "name": "Actor 1", "age": 30}

        entity_id = 1

        result = self.model.get_one_by_id(entity_id)

        get_one_by_id.assert_called_once_with(db_table_modifier=self.mock_db_modifier, id=entity_id)

        self.assertEqual(result, {"_id": 1, "name": "Actor 1", "age": 30})

    @patch("database.database_functions.update_one_by_attr")
    def test_update_one_by_attr(self, update_one_by_attr) -> None:
        """
        Test the method 'update_one_by_attr' in the file model.py.

        Parameters
        ----------
        update_one_by_attr : unittest.mock.MagicMock
            Mock object for the 'update_one_by_attr' method.

        Returns
        -------
        None
        """

        update_one_by_attr.return_value = {
            "_id": 1,
            "name": "Updated Actor",
            "age": 35,
        }

        attr = "name"
        attr_value = "Actor 1"
        attr_to_update = "age"
        attr_to_update_value = 35

        result = self.model.update_one_by_attr(attr, attr_value, attr_to_update, attr_to_update_value)

        update_one_by_attr.assert_called_once_with(
            db_table_modifier=self.mock_db_modifier,
            attr=attr,
            attr_value=attr_value,
            attr_to_update=attr_to_update,
            attr_to_update_value=attr_to_update_value,
        )

        self.assertEqual(result, {"_id": 1, "name": "Updated Actor", "age": 35})

    @patch("database.database_functions.update_one_by_id")
    def test_update_one_by_id(self, update_one_by_id) -> None:
        """
        Test the method 'update_one_by_id' in the file model.py.

        Parameters
        ----------
        update_one_by_id : unittest.mock.MagicMock
            Mock object for the 'update_one_by_id' method.

        Returns
        -------
        None
        """

        update_one_by_id.return_value = {
            "_id": 1,
            "name": "Updated Actor",
            "age": 35,
        }

        entity_id = 1
        attr_to_update = "age"
        attr_to_update_value = 35

        result = self.model.update_one_by_id(entity_id, attr_to_update, attr_to_update_value)

        update_one_by_id.assert_called_once_with(
            db_table_modifier=self.mock_db_modifier,
            id=entity_id,
            attr_to_update=attr_to_update,
            attr_to_update_value=attr_to_update_value,
        )

        self.assertEqual(result, {"_id": 1, "name": "Updated Actor", "age": 35})

    @patch("database.database_functions.delete_one_by_attr")
    def test_delete_one_by_attr(self, delete_one_by_attr) -> None:
        """
        Test the method 'delete_one_by_attr' in the file model.py.

        Parameters
        ----------
        delete_one_by_attr : unittest.mock.MagicMock
            Mock object for the 'delete_one_by_attr' method.

        Returns
        -------
        None
        """

        delete_one_by_attr.return_value = {"_id": 1, "name": "Actor 1", "age": 30}

        attr = "name"
        attr_value = "Actor 1"

        result = self.model.delete_one_by_attr(attr, attr_value)

        delete_one_by_attr.assert_called_once_with(
            db_table_modifier=self.mock_db_modifier, attr=attr, attr_value=attr_value
        )

        self.assertEqual(result, {"_id": 1, "name": "Actor 1", "age": 30})

    @patch("database.database_functions.delete_one_by_id")
    def test_delete_one_by_id(self, delete_one_by_id) -> None:
        """
        Test the method 'delete_one_by_id' in the file model.py.

        Parameters
        ----------
        delete_one_by_id : unittest.mock.MagicMock
            Mock object for the'delete_one_by_id' method.

        Returns
        -------
        None
        """

        delete_one_by_id.return_value = {"_id": 1, "name": "Actor 1", "age": 30}

        entity_id = 1

        result = self.model.delete_one_by_id(entity_id)

        delete_one_by_id.assert_called_once_with(db_table_modifier=self.mock_db_modifier, id=entity_id)

        self.assertEqual(result, {"_id": 1, "name": "Actor 1", "age": 30})
