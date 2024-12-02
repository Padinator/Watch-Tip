import database.model as m


class Users(m.DatabaseModel):
    """
        Sub class of DatabaseModel for interacting with the table for all
        users.
    """

    def __init__(self):
        """"Creates an user object for modifying users in database"""
        super().__init__(self._database_name, "all_users")
