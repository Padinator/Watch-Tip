import sys

from abc import ABC
from pathlib import Path
from typing import Any, Dict

# from typing import Any, Dict, override

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[1]
sys.path.append(str(project_dir))

import database.database_functions as dbf


class DatabaseModel(ABC):
    """
    Abstract base class for sub classes interacting with database.
    This abstract base class contains interactino function for interacting
    with the database

    Attributes
    ----------
    _db_table_modifier : DBModifier
        Object for accessing/modifying a specific table in database
    """

    __db_connector = dbf.connect_with_database()
    _database_name = "watch_tip"

    def __init__(self, database_name: str, collection_name: str):
        """
        Creates an object for modifying data in database

        Parameters
        ----------
        database_name : str
            Name of database to connect to
        collection_name: str
            Name of table to get access to
        """

        self._db_table_modifier = dbf.get_table_from_database(
            DatabaseModel.__db_connector, collection_name=collection_name, database_name=database_name
        )

    # ------------------------- CRUD functions -------------------------
    # Create functions
    def insert_one(self, entity: Dict[str, Any]) -> str:
        """
        Insert passed entity and returns it, if operation was successful.\n
        Parallel execution is possible.

        Parameter
        ---------
        enitity: Dict
            Entity to insert into database

        Returns
        -------
        str
            The mongoDB intern ID
        """

        return dbf.insert_one_element(db_table_modifier=self._db_table_modifier, enitity=entity)

    # Read functions
    def get_all(self) -> Dict[int, Dict[str, Any]]:
        """
        Returns all existing entites.

        Returns
        -------
        Dict[int, Dict[str, Any]]
            Returns a dict containing all entries of a table. The ID is the key
            and the properties are stored in the values.
        """

        return dbf.get_all_entries_from_database(db_table_modifier=self._db_table_modifier)

    def get_one_by_attr(self, attr: str, attr_value: Any) -> Dict[str, Any]:
        """
        Finds a specific entity by the value 'attribute_value' of the
        attribute 'attribute' and returns it.

        Parameters
        ----------
        attr : str
            Filter entites by attribute "attr"
        attr_value : Any
            Filter entites by value "attr_value" as value for attribute "attr"

        Returns
        -------
        Dict[str, Any]
            Returns one entity, if one was found, else None.
        """

        return dbf.get_one_by_attr(db_table_modifier=self._db_table_modifier, attr=attr, attr_value=attr_value)

    def get_one_by_id(self, id: int) -> Dict[str, Any]:
        """
        Finds a specific entity by ID and returns it.

        Parameters
        ----------
        id : Any
            Filter entites by value "id" as value for attribute "id"

        Returns
        -------
        Dict[str, Any]
            Returns one entity, if one was found, else None.
        """

        return dbf.get_one_by_id(db_table_modifier=self._db_table_modifier, id=id)

    # Update functions
    def update_one_by_attr(
        self, attr: str, attr_value: Any, attr_to_update: str, attr_to_update_value: Any
    ) -> Dict[str, Any]:
        """
        Updates entity with passed id 'id' by changes 'changes_on_entity'.
        Returns saved object, if operation was successful
        (parallel execution possible).

        Parameters
        ----------
        attr : str
            Filter entites by attribute "attr"
        attr_value : Any
            Filter entites by value "attr_value" as value for attribute "attr"
        attr_to_update : str
            Attribute to update
        attr_to_update_value : Any
            Value of attribute "attr_to_update" to update

        Returns
        -------
        Dict[str, Any]
            Returns the modified/updated entity.
        """

        return dbf.update_one_by_attr(
            db_table_modifier=self._db_table_modifier,
            attr=attr,
            attr_value=attr_value,
            attr_to_update=attr_to_update,
            attr_to_update_value=attr_to_update_value,
        )

    def update_one_by_id(self, id: int, attr_to_update: str, attr_to_update_value: Any) -> Dict[str, Any]:
        """
        Updates entity with passed id 'id' by changes 'changes_on_entity'.
        Returns saved object, if operation was successful
        (parallel execution possible).

        Parameters
        ----------
        db_table_modifier : "DBModifier"
            Database modifier object for updating one entity from database
            reachable with current DBModifier object
        id : Any
            Filter entites by value "id" as value for attribute "id"
        attr_to_update : str
            Attribute to update
        attr_to_update_value : Any
            Value of attribute "attr_to_update" to update

        Returns
        -------
        Dict[str, Any]
            Returns the modified/updated entity.
        """

        return dbf.update_one_by_id(
            db_table_modifier=self._db_table_modifier,
            id=id,
            attr_to_update=attr_to_update,
            attr_to_update_value=attr_to_update_value,
        )

    # Delete functions
    def delete_one_by_attr(self, attr: str, attr_value: Any) -> Dict[str, Any]:
        """
        Deletes entity with passed id 'id' and returns it, if
        operation was successful (parallel execution possible).

        Parameters
        ----------
        attr : str
            Filter entites by attribute "attr"
        attr_value : Any
            Filter entites by value "attr_value" as value for attribute "attr"

        Returns
        -------
        Dict[str, Any]
            Returns the deleted entity, if one to delete was found, else
            returns None.
        """

        return dbf.delete_one_by_attr(db_table_modifier=self._db_table_modifier, attr=attr, attr_value=attr_value)

    def delete_one_by_id(self, id: int) -> Dict[str, Any]:
        """
        Deletes entity with passed id 'id' and returns it, if
        operation was successful (parallel execution possible).

        Parameters
        ----------
        id : Any
            Filter entites by value "id" as value for attribute "id"

        Returns
        -------
        Dict[str, Any]
            Returns the deleted entity, if one to delete was found, else
            returns None.
        """

        return dbf.delete_one_by_id(db_table_modifier=self._db_table_modifier, id=id)
