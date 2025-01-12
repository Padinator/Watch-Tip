import json
import pickle

from pathlib import Path
from typing import Any, Dict, List


def save_object_in_file(path: Path, object_to_store: Any) -> None:
    """
    Store passed object "object_to_store" in file laying at passed path
    "path". Use for storing python module "pickle" and write object in
    binary mode.

    Parameters
    ----------
    path : Path
        Path where passed object will be stored
    object_to_store : Any
        Object to store/save
    """

    with open(path, "wb") as file:
        pickle.dump(object_to_store, file)


def load_object_from_file(path: Path) -> Any:
    """
    Loads object from file laying under path "path" and returns load
    object with the help of the python module "pickle".

    Parameters
    ----------
    path : Path
        Path from which an object will be loaded

    Returns
    -------
    Any
        Returns loaded object.
    """

    object_to_load = None

    with open(path, "rb") as file:
        object_to_load = pickle.load(file)

    return object_to_load


def save_json_objects_in_file(path: Path, json_objects: List[Any]) -> None:
    """
    Saves JSON objects in a file. Each line will contain one JSON object.
    This will be done by the predefined python module "json".

    Parameter
    ---------
    path : Path
        Path where passed JSON objects will be stored
    json_objects : List[Any]
        List of JSON objects to store/save under passed path
    """

    with open(path, "w", encoding="utf-8") as file:
        for obj in json_objects:
            json.dump(obj, file)
            file.write("\n")  # Each object will be saved on its own line


def load_json_objects_from_file(path: Path, keys: List[Any] = None) -> List[Any]:
    """
    Loads all JSON object from file, where each line contains exactly one
    JSON object. This will be done by the predefined python module "json".
    Read only necessary keys and create one dict, if no passed read all
    keys.

    Parameters
    ----------
    path : Path
        Path from which JSON object will be loaded
    keys : List[Any], default None
        Read only requested keys from JSON objects

    Returns
    -------
    List[Any]
        Returns a list of loaded JSON formatted objects
    """

    all_objects = []

    with open(path, "rb") as file:
        all_objects = [json.loads(line.decode("utf-8")) for line in file.readlines()]

        if keys:  # Only returns keys, which are requested
            for obj in all_objects:
                obj = dict([(key, obj[key]) for key in keys])

    return all_objects


def save_one_json_object_in_file(path: Path, obj: Dict[Any, Any]) -> None:
    """
    Saves a JSON object in a file. For saving the prettied
    format (RFC 7159 with 4 spaces as tab) will be used. This will be done by
    the predefined python module "json".

    Parameter
    ---------
    path : Path
        Path where passed JSON objects will be stored
    json_objects : List[Any]
        List of JSON objects to store/save under passed path
    """

    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, indent=4)


def load_one_json_object_from_file(path: Path, keys=None) -> List[Any]:
    """
    Loads JSON object from file, where the JSON object is stored in prettied
    format (RFC 7159 with 4 spaces as tab). This will be done by the
    predefined python module "json". Read only necessary keys and create one
    dict, if no passed read all keys.

    Parameters
    ----------
    path : Path
        Path from which JSON object will be loaded
    keys : List[Any], default None
        Read only requested keys from JSON objects

    Returns
    -------
    List[Any]
        Returns a list of loaded JSON formatted objects
    """

    with open(path, "r") as file:
        obj = json.load(file)

        if keys:  # Only returns keys, which are requested
            obj = dict([(key, obj[key]) for key in keys])

        return obj
