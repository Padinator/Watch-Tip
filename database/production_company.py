import sys

from pathlib import Path

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[1]
sys.path.append(str(project_dir))

import database.model as m


class ProductionCompanies(m.DatabaseModel):
    """
        Sub class of DatabaseModel for interacting with the table for all
        production companies.
    """

    def __init__(self):
        """"Creates a production company object for modifying production companies in database"""
        super().__init__(self._database_name, "all_production_companies")
