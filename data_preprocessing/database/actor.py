import database.model as m


class Actors(m.DatabaseModel):
    """
        Sub class of DatabaseModel for interacting with the table for all
        actors.
    """

    def __init__(self):
        """Creates an actor object for modifying actors in database"""
        super().__init__(self._database_name, "all_actors")
