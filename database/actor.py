import sys

from pathlib import Path

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[1]
sys.path.append(str(project_dir))

import database.model as m


class Actors(m.DatabaseModel):
    """
    Subclass of DatabaseModel for interacting with the table for all
    actors.
    """

    def __init__(self):
        """
        Creates an actor object for modifying actors in database
        """

        super().__init__(self._database_name, "all_actors")
