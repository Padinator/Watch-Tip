import sys

from pathlib import Path

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[1]
sys.path.append(str(project_dir))

import database.model as m


class Genres(m.DatabaseModel):
    """
    Subclass of DatabaseModel for interacting with the table for all
    genres.
    """

    def __init__(self):
        """
        Creates an genre object for modifying genres in database
        """

        super().__init__(self._database_name, "all_genres")
