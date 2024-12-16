import unittest

from unittest.mock import patch

from database.model import DatabaseModel
from database.user import Users

model = DatabaseModel("test_db", "test_collection")
model._table = "mock_table"


class TestUser(unittest.TestCase):

    @patch("database.database_functions.get_entries_by_attr_from_database")
    def test_get_all(self, get_entries_by_attr_from_database) -> None:
        """Test the method 'get_all' in the file user.py"""

        get_entries_by_attr_from_database.return_value = [
            {"_id": 1, "user": "user1", "name": "First User", "age": 22},
            {"_id": 2, "user": "user2", "name": "Second User", "age": 24},
        ]

        expected_result = {
            "user1": {"name": "First User", "age": 22},
            "user2": {"name": "Second User", "age": 24},
        }

        result = Users.get_all()

        get_entries_by_attr_from_database.assert_called_once_with(
            table=model._table, attr="", attr_value=""
        )

        self.assertEqual(result, expected_result)
