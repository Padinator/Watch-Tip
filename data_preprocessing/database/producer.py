import database.model as m


class Producers(m.DatabaseModel):
    """
        Sub class of DatabaseModel for interacting with the table for all
        producers.
    """

    def __init__(self):
        """"Creates a producer object object for modifying producers in database"""
        super().__init__(self._database_name, "all_producers")
