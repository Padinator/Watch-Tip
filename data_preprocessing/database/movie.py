import database.model as m

class Movies(m.DatabaseModel):
    """
        Sub class of DatabaseModel for interacting with the table for all
        movies.
    """

    def __init__(self):
        """"Creates a movie object for modifying movies in database"""
        super().__init__(self._database_name, "all_movies")
