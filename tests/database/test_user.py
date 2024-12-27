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
        self.mock_db_modifier = MagicMock(spec=DBModifier)
        self.user = Users()
        self.user._db_table_modifier = self.mock_db_modifier

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

        result = self.user.get_all()

        get_entries_by_attr_from_database.assert_called_once_with(
            db_table_modifier=self.mock_db_modifier, attr="", attr_value=""
        )

        self.assertEqual(result, expected_result)
