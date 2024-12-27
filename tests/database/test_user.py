import sys
import unittest

from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[2]
sys.path.append(str(project_dir))

from database.database_functions import DBModifier
from database.user import Users


class TestUser(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment for each test.

        This method is called before each test to set up any state that is shared
        between tests. It creates a mock object for the DBModifier class and assigns
        it to the _db_table_modifier attribute of the Users instance.

        Attributes
        ----------
        mock_db_modifier : unittest.mock.MagicMock
            A mock object for the DBModifier class.
        user : Users
            An instance of the Users class with the _db_table_modifier attribute
            set to the mock_db_modifier.
        """
        self.mock_db_modifier = MagicMock(spec=DBModifier)
        self.user = Users()
        self.user._db_table_modifier = self.mock_db_modifier

    @patch("database.database_functions.get_entries_by_attr_from_database")
    def test_get_all(self, get_entries_by_attr_from_database) -> None:
        """
        Test the method 'get_all' in the file user.py.

        This test verifies that the 'get_all' method correctly retrieves all user entries
        from the database and returns them in the expected format.

        Parameters
        ----------
        get_entries_by_attr_from_database : MagicMock
            A mock function that simulates retrieving entries from the database based on attributes.

        Returns
        -------
        None
        """

        get_entries_by_attr_from_database.return_value = [
            {"_id": 1, "user": "user1", "name": "First User", "age": 22},
            {"_id": 2, "user": "user2", "name": "Second User", "age": 24},
        ]

        expected_result = {
            "user1": {"name": "First User", "age": 22},
            "user2": {"name": "Second User", "age": 24},
        }

        result = self.user.get_all()

        get_entries_by_attr_from_database.assert_called_once_with(
            db_table_modifier=self.mock_db_modifier, attr="", attr_value=""
        )

        self.assertEqual(result, expected_result)
