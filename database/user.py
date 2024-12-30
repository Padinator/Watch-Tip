import copy as cp
import sys

from pathlib import Path
from typing import Any, Dict
# from typing import Any, Dict, override

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[1]
sys.path.append(str(project_dir))

import database.model as m
import database.database_functions as dbf


class Users(m.DatabaseModel):
    """
    Subclass of DatabaseModel for interacting with the table for all
    users.
    """

    def __init__(self):
        """
        Creates an user object for modifying users in database
        """

        super().__init__(self._database_name, "all_users")

    # Read functions
    # @override
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Finds and returns all existing entites. This method uses the method
        "get_entries_by_attr_from_database", because the table user does
        not have an attribute "id" per user, only a username under the key
        "user".

        Returns
        -------
        Dict[str, Dict[str, Any]]
            All existing user review entities with username as key of this
            dict. The properties are listed in the corresponding inner dicts.
        """

        id_key = "user"
        all_users = dbf.get_entries_by_attr_from_database(db_table_modifier=self._db_table_modifier, attr="", attr_value="")
        all_users_formatted = {}

        # Format all users to not have the username as key -> it will be used as dict ID
        for user in all_users:
            username = user[id_key]
            user_copied = cp.copy(user)
            del user_copied["_id"]
            del user_copied[id_key]
            all_users_formatted[username] = user_copied

        return all_users_formatted
