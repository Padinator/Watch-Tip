import sys

from pathlib import Path

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[1]
sys.path.append(str(project_dir))

import database.model as m


class Producers(m.DatabaseModel):
    """
    Subclass of DatabaseModel for interacting with the table for all
    producers.
    """

    def __init__(self):
        """
        Creates a producer object object for modifying producers in database
        """

        super().__init__(self._database_name, "all_producers")
