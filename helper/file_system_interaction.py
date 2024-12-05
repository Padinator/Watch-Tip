import json
import pickle

from pathlib import Path
from typing import Any, List


def save_object_in_file(path: Path, object_to_store: Any) -> None:
    """
        Store passed object "object_to_store" in file laying at passed path
        "path". Use for storing python module "pickle" and write object in
        binary mode.
    """

    with open(path, "wb") as file:
        pickle.dump(object_to_store, file)

def load_object_from_file(path: Path) -> Any:
    """
        Loads object from file laying under path "path" and returns load
        object with the help of the python module "pickle".
    """

    object_to_load = None

    with open(path, "rb") as file:
        object_to_load = pickle.load(file)

    return object_to_load


def save_json_objects_in_file(path: Path, json_objects: List[Any]) -> None:
    """
        Saves json objects in a file. Each line will contain one json object.
        This will be done by the predefined python module "json".
    """

    with open(path, "w", encoding="utf-8") as file:
        for obj in json_objects:
            json.dump(obj)
            file.write("\n")  # Each object will be saved on its own line


def load_json_objects_from_file(path: Path, keys=None) -> List[Any]:
    """
        Loads all json object from file, where each line contains exactly one
        json object. This will be done by the predefined python module "json".
        Read only necessary keys and create one dict, if no passed read all
        keys.
    """

    all_objects = []

    with open(path, "rb") as file:
        all_objects = [json.loads(line.decode("utf-8")) for line in file.readlines()]

        if keys:  # keys != None or keys != []
            for obj in all_objects:
                obj = dict([(key, obj[key]) for key in keys])

    return all_objects
