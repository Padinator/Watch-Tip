import copy as cp

from pymongo import MongoClient, ReturnDocument
from pymongo.collection import Collection
from typing import Any, Dict


# Define basic URL
basic_database_url = "mongodb://localhost"
default_port = 27017
default_database_name = "watch_tip"


class DBModifier:
    """
    Wrapper class for modifiying database contens
    """

    def __init__(self, collection: Collection):
        """
        Creates an object for modifying a specific table from database.

        Parameters
        ----------
        collection : Collection
            Collection with which table from database can be accessed/
            modified. It will be stored.
        """

        self.__collection = collection

    def __call__(self) -> Collection:
        """
        Returns the collection for modifying a collection/table in MongoDB.

        Returns
        -------
        Collection
            Returns the collection for modifying a collection/table in MongoDB.
        """

        return self.__collection


class DBConnector:
    """
    Wrapper class for connecting with database and requesting/modifiying data
    from database. The purpose is that in database/model.py no database
    specific parts are included and of course not the exact database connector
    object.
    """

    def __init__(self, connector_object: MongoClient):
        """
        Creates a connector object, which establishes a connection to
        database.

        Parameters
        ----------
        connector_object : MongoClient
            Client for communicating with MongoDB. It will be stored.
        """

        self.__connector_object = connector_object

    def get_table_from_database(self, collection_name: str, database_name: str = default_database_name) -> "DBModifier":
        """
        Requests database for a specific table.

        Parameters
        ----------
        collection_name : str,
            Name of collection/table to request/access from MongoDB
        database_name : str, default default_database_name
            Name of database, which contains collections/tables like
            "collection_name", which will be requested.

        Returns
        -------
        DBModifier
            Returns an database modifier object for modifying passed
            collection/table "collection_name".
        """

        mongodb = self.__connector_object[database_name]
        collection = mongodb[collection_name]
        return DBModifier(collection=collection)


# Connect function
def connect_with_database(url: str = basic_database_url, port: int = default_port) -> "DBConnector":
    """
    Connects to database and returns a database object for establishing a
    connection.

    url : str, default basic_database_url
        URL under which the database is accessable
    port : int, default default_port
        Port which will be inserted in the passed URL "url" for connecting to
        database
    """

    mongo_client = MongoClient(f"{url}:{port}/")
    return DBConnector(mongo_client)


# Access collection from database
def get_table_from_database(
    db_connector: "DBConnector",
    collection_name: str,
    database_name: str = default_database_name,
) -> "DBModifier":
    """
    Returns requested collection of MongoDB.

    Parameters
    ----------
    mongo_client: MongoClient
        Client for communicating to MongoDB
    collection_name: str
        Name of table to get access to
    database_name : str
        Name of database to connect to
    """

    return db_connector.get_table_from_database(collection_name=collection_name, database_name=database_name)


# ------------------------- CRUD functions -------------------------
# Crud functions will bee database specific (here MongoDB-specific)
# Create functions
def insert_one_element(db_table_modifier: "DBModifier", enitity: Dict) -> str:
    """
    Inserts the passed dictionary with any format into the database.
    The entity must have the JSON format!\n
    Parallel execution possible.

    Parameter
    ---------
    db_table_modifier: "DBModifier"
        Database modifier object for inserting one entity into table into
        database reachable with current DBModifier object
    enitity: Dict
        Entity to insert into database

    Returns
    -------
    str
        The mongoDB intern ID
    """

    return db_table_modifier().insert_one(enitity)


# Read functions
def read_all_entries_from_database_as_dict(
    db_table_modifier: "DBModifier",
) -> Dict[int, Dict[str, Any]]:
    """
    Reads a collection from database and passes user a deep
    copy for not manipulating the data by accident.\n
    Deletes intern key "_id" of MongoDB and uses self defined key "id".

    Parameter
    ---------
    db_table_modifier: "DBModifier"
        Database modifier object for requesting all entities reachable with
        current DBModifier object

    Returns
    -------
    Dict[int, Dict[str, Any]]
        Returns a dict containing all entries of a table. The ID is the key
        and the properties are stored in the values.
    """

    data = {}

    for entity in db_table_modifier().find():  # Cursor for iterating over all genres
        entity_id = entity["id"]
        entity_copied = cp.copy(entity)
        del entity_copied["_id"]
        del entity_copied["id"]
        data[entity_id] = entity_copied

    return data


def get_all_entries_from_database(db_table_modifier: "DBModifier") -> Dict[int, Dict[str, Any]]:
    """
    Reads all entries from passed collection from database and returns
    user a deep copy for not manipulating the data by accident.

    Parameter
    ---------
    db_table_modifier: "DBModifier"
        Database modifier object for requesting all entities reachable with
        current DBModifier object

    Returns
    -------
    Dict[int, Dict[str, Any]]
        Returns a dict containing all entries of a table. The ID is the key
        and the properties are stored in the values.
    """

    return read_all_entries_from_database_as_dict(db_table_modifier=db_table_modifier)


def get_entries_by_attr_from_database(
    db_table_modifier: "DBModifier", attr: str, attr_value: str
) -> Dict[int, Dict[str, Any]]:
    """
    Reads all entries from passed collection from database matching the
    value "attr_value" of the attribute "attr" and returns user a deep
    copy for not manipulating the data by accident.
    """

    if attr == "" and attr_value == "":
        return [cp.copy(entity) for entity in db_table_modifier().find()]
    return [cp.copy(entity) for entity in db_table_modifier().find({attr: attr_value})]


def get_one_by_attr(db_table_modifier: "DBModifier", attr: str, attr_value: Any) -> Dict[str, Any]:
    """
    Searches in database for a specific entity by the value 'attr_value' of the
    attribute 'attr' and returns it.

    Parameters
    ----------
    db_table_modifier : "DBModifier"
        Database modifier object for requesting one entity from database
        reachable with current DBModifier object
    attr : str
        Filter entites by attribute "attr"
    attr_value : Any
        Filter entites by value "attr_value" as value for attribute "attr"

    Returns
    -------
    Dict[str, Any]
        Returns one entity, if one was found, else None.
    """

    return db_table_modifier().find_one({attr: attr_value})


def get_one_by_id(db_table_modifier: "DBModifier", id: int) -> Dict[str, Any]:
    """
    Searches in database for a specific entity by the passed ID and
    returns found entity. This function calls function "get_one_by_attr".

    Parameters
    ----------
    db_table_modifier : "DBModifier"
        Database modifier object for requesting one entity from database
        reachable with current DBModifier object
    id : Any
        Filter entites by value "id" as value for attribute "id"

    Returns
    -------
    Dict[str, Any]
        Returns one entity, if one was found, else None.
    """

    return get_one_by_attr(db_table_modifier=db_table_modifier, attr="id", attr_value=id)


# Update functions
def update_one_by_attr(
    db_table_modifier: "DBModifier", attr: str, attr_value: Any, attr_to_update: str, attr_to_update_value: Any
) -> Dict[str, Any]:
    """
    Searches in database for a specific entity (first one found) by the value
    'attr_value' of the attribute 'attr', updates the attribute
    'attr_to_update' with the value 'attr_to_update_value' and returns the
    modified entity.\n
    If entity does not exist in database, nothing will be done.

    Parameters
    ----------
    db_table_modifier : "DBModifier"
        Database modifier object for updating one entity from database
        reachable with current DBModifier object
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

    return db_table_modifier().find_one_and_update(
        {attr: attr_value}, {"$set": {attr_to_update: attr_to_update_value}}, return_document=ReturnDocument.AFTER
    )


def update_one_by_id(
    db_table_modifier: "DBModifier", id: int, attr_to_update: str, attr_to_update_value: Any
) -> Dict[str, Any]:
    """
    Searches in database for a specific entity (first one found) by the passed
    ID, updates the attribute 'attr_to_update' with the value
    'attr_to_update_value' and returns the modified entity. If entity does not
    exist in database, nothing will be done. This function calls
    "update_one_by_attr".

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

    return update_one_by_attr(
        db_table_modifier=db_table_modifier,
        attr="id",
        attr_value=id,
        attr_to_update=attr_to_update,
        attr_to_update_value=attr_to_update_value,
    )


# Delete functions
def delete_one_by_attr(db_table_modifier: "DBModifier", attr: str, attr_value: Any) -> Dict[str, Any]:
    """
    Searches in database for a specific entity by the value 'attr_value'
    of the attribute 'attr', deletes the first found entity and returns
    the modified entity.

    Parameters
    ----------
    db_table_modifier : "DBModifier"
        Database modifier object for deleting one entity from database
        reachable with current DBModifier object
    attr : str
        Filter entites by attribute "attr"
    attr_value : Any
        Filter entites by value "attr_value" as value for attribute "attr"

    Returns
    -------
    Dict[str, Any]
        Returns the deleted entity, if one to delete was found, else returns
        None.
    """

    return db_table_modifier().find_one_and_delete({attr: attr_value})


def delete_one_by_id(db_table_modifier: "DBModifier", id: int) -> Dict[str, Any]:
    """
    Searches in database for a specific entity by the passed ID, deletes
    the first found entity and returns the modified entity.

    Parameters
    ----------
    db_table_modifier : "DBModifier"
        Database modifier object for deleting one entity from database
        reachable with current DBModifier object
    id : Any
        Filter entites by value "id" as value for attribute "id"

    Returns
    -------
    Dict[str, Any]
        Returns the deleted entity, if one to delete was found, else returns
        None.
    """

    return delete_one_by_attr(db_table_modifier=db_table_modifier, attr="id", attr_value=id)
