from abc import ABC, abstractmethod
import database.database_functions as dbf

from typing import Any, Dict


class DatabaseModel(ABC):
    """
        Abstract base class for sub classes interacting with database.
        This abstract base class contains interactino function for interacting
        with the database
    """

    __database = dbf.connect_with_database()
    _database_name = "watch_tip"

    def __init__(self, database_name: str, collection_name: str):
        """
            Creates an object for modifying data in database
        """

        self._table = dbf.get_mongo_db_specific_collection(
            DatabaseModel.__database, database_name=database_name,
                collection_name=collection_name)

    # ------------------------- CRUD functions -------------------------
    # Create functions
    def insert_one(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """
            Insert passed entity and returns it, if operation was successful
            (parallel execution possible).
        """
        return dbf.insert_one_element(table=self._table, enitity=entity)

    # Read functions
    def get_all(self) -> Dict[int, Dict[str, Any]]:
        """
            Returns all existing entites.
        """

        return dbf.get_all_entries_from_database(table=self._table)

    def get_one_by_attr(self, attr: str, attr_value: Any) -> Dict[str, Any]:
        """
            Finds a specific entity by the value 'attribute_value' of the
            attribute 'attribute' and returns it.
        """

        return dbf.get_one_by_attr(table=self._table, attr=attr, attr_value=attr_value)

    def get_one_by_id(self, id: int) -> Dict[str, Any]:
        """
            Finds a specific entity by ID and returns it.
        """

        return dbf.get_one_by_id(table=self._table, id=id)

    # Update functions
    def update_one_by_attr(self, attr: str, attr_value: Any, attr_to_update: str, attr_to_update_value: Any) -> Dict[str, Any]:
        """
            Updates entity with passed id 'id' by changes 'changes_on_entity'.
            Returns saved object, if operation was successful
            (parallel execution possible).
        """

        return dbf.update_one_by_attr(table=self._table, attr=attr,
            attr_value=attr_value, attr_to_update=attr_to_update,
            attr_to_update_value=attr_to_update_value)

    def update_one_by_id(self, id: int, attr_to_update: str, attr_to_update_value: Any) -> Dict[str, Any]:
        """
            Updates entity with passed id 'id' by changes 'changes_on_entity'.
            Returns saved object, if operation was successful
            (parallel execution possible).
        """

        return dbf.update_one_by_id(table=self._table, id=id,
            attr_to_update=attr_to_update,
            attr_to_update_value=attr_to_update_value)

    # Delete functions
    def delete_one_by_attr(self, attr: str, attr_value: Any) -> Dict[str, Any]:
        """
            Deletes entity with passed id 'id' and returns it, if
            operation was successful (parallel execution possible).
        """

        return dbf.delete_one_by_attr(table=self._table, attr=attr, attr_value=attr_value)
    
    def delete_one_by_id(self, id: int) -> Dict[str, Any]:
        """
            Deletes entity with passed id 'id' and returns it, if
            operation was successful (parallel execution possible).
        """

        return dbf.delete_one_by_id(table=self._table, id=id)
