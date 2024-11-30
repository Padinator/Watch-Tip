import database.model as m


class ProductionCompanies(m.DatabaseModel):
    """
        Sub class of DatabaseModel for interacting with the table for all
        production companies.
    """

    def __init__(self):
        """"Creates an genre object for modifying genres in database"""
        super().__init__(self._database_name, "all_production_companies")
