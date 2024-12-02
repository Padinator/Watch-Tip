import pickle

from pathlib import Path
from typing import Any


def save_object_in_file(path: Path, object_to_store: Any) -> None:
    """
        Store passed object "object_to_store" in file laying at passed path
        "path". Use for storing pickle and write object in binary mode
    """

    with open(path, "wb") as file:
        pickle.dump(object_to_store, file)

def load_object_from_file(path: Path) -> Any:
    """
        Loads object from file laying under path "path" and returns load
        object.
    """

    object_to_load = None

    with open(path, "rb") as file:
        object_to_load = pickle.load(file)

    return object_to_load
