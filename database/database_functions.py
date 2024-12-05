import copy as cp
import pymongo

from typing import Any, Dict, List


# Define basic URL
basic_database_url = "mongodb://localhost"
default_port = 27017
default_database_name = "watch_tip"


# Connect function
def connect_with_database(url: str=basic_database_url, port: int = default_port) -> pymongo.MongoClient:
    """
        Connects to database and returns a database object
    """

    return pymongo.MongoClient(f"{url}:{port}/")


# ------------------------- CRUD functions -------------------------
# Crud functions will bee database specific (here MongoDB-specific)
# Create functions
def insert_one_element(table: pymongo.synchronous.collection.Collection, enitity: Dict) -> None:
    """
        Inserts the passed dictionary with any format into the database.
        The entity must have the correct format!
    """

    table.insert_one(enitity)


# Read functions
def get_mongo_db_specific_collection(mongo_client: pymongo.MongoClient, collection_name: str, database_name: str=default_database_name):
    """
        Returns requested collection of MongoDB.
    """

    mongodb = mongo_client[database_name]
    collection = mongodb[collection_name]

    return collection


def read_all_entries_from_database_as_dict(table: pymongo.synchronous.collection.Collection) -> Dict[int, Dict[str, Any]]:
    """
        Reads a collection from database and passes user a deep
        copy for not manipulating the data by accident.
    """

    data = {}

    for entity in table.find():  # Cursor for iterating over all genres
        entity_id = entity["id"]
        entity_copied = cp.copy(entity)
        del entity_copied["_id"]
        del entity_copied["id"]
        data[entity_id] = entity_copied

    return data


def get_all_entries_from_database(table: pymongo.synchronous.collection.Collection) -> Dict[int, Dict[str, Any]]:
    """
        Reads all entries from passed collection from database and returns
        user a deep copy for not manipulating the data by accident.
    """

    return read_all_entries_from_database_as_dict(table=table)


def get_entries_by_attr_from_database(table: pymongo.synchronous.collection.Collection, attr: str, attr_value: str) -> Dict[int, Dict[str, Any]]:
    """
        Reads all entries from passed collection from database matching the
        value "attr_value" of the attribute "attr" and returns user a deep
        copy for not manipulating the data by accident.
    """

    if attr == "" and attr_value == "":
        return [cp.copy(entity) for entity in table.find()]
    return [cp.copy(entity) for entity in table.find({attr: attr_value})]


def get_one_by_attr(table: pymongo.synchronous.collection.Collection, attr: str, attr_value: Any) -> Dict[str, Any]:
    """
        Searches in database for a specific entity by the value 'attr_value' of the
        attribute 'attr' and returns it.
    """

    return table.find_one({attr: attr_value})


def get_one_by_id(table: pymongo.synchronous.collection.Collection, id: int) -> Dict[str, Any]:
    """
        Searches in database for a specific entity by the passed ID and
        returns found entity.
    """

    return get_one_by_attr(table=table, attr="_id", attr_value=id)


# Update functions
def update_one_by_attr(table: pymongo.synchronous.collection.Collection, attr: str, attr_value: Any,
                       attr_to_update: str, attr_to_update_value: Any) -> Dict[str, Any]:
    """
        Searches in database for a specific entity by the value 'attr_value' of the
        attribute 'attr', updates the attribute 'attr_to_update' with the value
        'attr_to_update_value' and returns the modified entity.
    """

    return table.find_one_and_update({attr: attr_value},
        {"$set" : {attr_to_update: attr_to_update_value }})


def update_one_by_id(table: pymongo.synchronous.collection.Collection, id: int,
                     attr_to_update: str, attr_to_update_value: Any) -> Dict[str, Any]:
    """
        Searches in database for a specific entity by the passed ID, updates
        the attribute 'attr_to_update' with the value 'attr_to_update_value'
        and returns the modified entity.
    """

    return update_one_by_attr(table=table, attr="_id", attr_value=id,
                attr_to_update=attr_to_update,
                attr_to_update_value=attr_to_update_value)


# Delete functions
# Update functions
def delete_one_by_attr(table: pymongo.synchronous.collection.Collection, attr: str, attr_value: Any) -> Dict[str, Any]:
    """
        Searches in database for a specific entity by the value 'attr_value'
        of the attribute 'attr', deletes the found entity and returns
        the modified entity.
    """

    return table.find_one_and_delete({attr: attr_value})


def delete_one_by_id(table: pymongo.synchronous.collection.Collection, id: int) -> Dict[str, Any]:
    """
        Searches in database for a specific entity by the passed ID, deletes
        the found entity and returns the modified entity.
    """

    return delete_one_by_attr(table=table, attr="_id", attr_value=id)
