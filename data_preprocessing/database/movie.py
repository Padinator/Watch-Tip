import database.model as m

class Movies(m.DatabaseModel):
    """
        Sub class of DatabaseModel for interacting with the table for all
        movies.
    """

    def __init__(self):
        """"Creates an genre object for modifying genres in database"""
        super().__init__(self._database_name, "all_movies")
