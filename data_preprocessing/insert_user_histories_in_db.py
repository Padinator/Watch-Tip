import collections
import operator
import pickle

from difflib import SequenceMatcher
from multiprocessing.pool import ThreadPool
from threading import Thread, Semaphore
from typing import Any, Callable, Dict, List, Iterable

import data_preprocessing.helper.variables as vars

from database.movie import Movies


# Define some constants
COUNT_OF_MAX_RUNNING_THREADS = 16


def compareStrings(s1: str, s2: str, min_ratio: float=0.8) -> bool:
    """
        Returns if both strings are similar with comparing them like
        Levenshtein distance, but here:
        2 * <#matching chars> / ([len(s1) + len(s2)]^2)
    """

    return min_ratio < SequenceMatcher(None, s1, s2).ratio()


def find_netflixmovie_in_database(line: str, all_movies: Dict[int, Any]) -> Dict[str, Any]:
    id, year, name = line.split(",", 2)
    movie_in_database = {}

    # Search netflix movie in database
    for movie_id, movie in all_movies.items():
        if compareStrings(name, movie["original_title"]) or compareStrings(name, movie["title"]):
        # if name == movie["original_title"] or name == movie["title"]:
            movie_in_database[movie_id] = movie

    # Check, if no movie, one movie or multiple movies were found
    if len(movie_in_database) == 0:
        return None
    elif len(movie_in_database) == 1:  # Only one matching movie found
        movie_in_database = list(movie_in_database.keys())[0]
    elif 1 < len(movie_in_database):  # Find movie with closest release_date
        movie_in_database = max(movie_in_database.items(), key=abs(int(year) - int(operator.itemgetter(1)["release_date"])))
        print("here")

    return {"id": movie_in_database, "netflix_movie_id": id,"name": name.strip(), "year": year}



def parallelize_task_and_return_results(task: Callable, args: List[Any], thread_sem: Semaphore, result_storage_sem: Semaphore,
                                        thread_number: int, result_storage: Dict[int, Any]) -> None:
    """
        Execute a task with a thread and save result in an on reference passed storage/variable.
    """

    result = task(*args)  # Store result
    result_storage_sem.acquire()
    result_storage[thread_number] = result
    result_storage_sem.release()
    thread_sem.release()  # Next thread can start


def parallelize_task_with_return_values(task: Callable, args: Iterable, max_number_of_runnings_threads: int = 8) -> Dict[int, Any]:
    """
        Executes a passed function "task" with passed arguments "args"
        and returns results of each thread as list.
    """

    # results = []  # Results of each thread
    # pool = ThreadPool(max_number_of_runnings_threads)

    # # Add all tasks and create threads
    # for args_per_function_call in args:
    #     args_per_function_call = args_per_function_call if isinstance(args_per_function_call, Iterable) else (args_per_function_call)
    #     results.append(pool.apply_async(task, args=args_per_function_call))

    # # Wait for results of the pool
    # pool.close()
    # pool.join()
    # results = [r.get() for r in results]
    # return results

    results = {}
    thread_sem = Semaphore(max_number_of_runnings_threads)
    result_storage_sem = Semaphore(1)

    for i, args_per_function_call in enumerate(args):
        if i % 10 == 0:
            print(f"Iteration {i}")

        thread_sem.acquire()
        args = [task, args_per_function_call, thread_sem, result_storage_sem, i, results]  # Args of function task
        t = Thread(target=parallelize_task_and_return_results, args=args)
        t.start()

    return results


if __name__ == "__main__":
    # Define variables
    netflix_movies = {}
    netflix_series = {}

    # Connect to database and read all movies
    all_movies_table = Movies()
    all_movies = dict([(movie_id, {
            "original_title": movie["original_title"], "title": movie["title"],
            "release_year": movie["release_date"].split("-")[0]
        }) for movie_id, movie in list(all_movies_table.get_all().items())])

    # Read movies listed by Netflix
    with open(vars.local_netflix_movies_file_path, "r") as netflix_movies_file:
        args = [(movie, all_movies) for movie in netflix_movies_file.readlines()]
        results = parallelize_task_with_return_values(find_netflixmovie_in_database, args, COUNT_OF_MAX_RUNNING_THREADS)

        for thread_id, movie in results.items():
            netflix_movie_id = movie["netflix_movie_id"]
            del movie["netflix_movie_id"]
            netflix_movies[netflix_movie_id] = movie

    # print(list(netflix_movies.keys())[0])
    # netflix_movies_str = ["{}_{}".format(movie["name"], movie["id"]) for movie in netflix_movies.values()]
    # print(len(netflix_movies_str), len(set(netflix_movies_str)))
    # print([item for item, count in collections.Counter(netflix_movies_str).items() if count > 1])

    # Save found movies in file
    found_movies = dict([(movie_id, movie) for movie_id, movie in netflix_movies.items() if movie["id"] != None])
    print(found_movies)
    with open(vars.map_for_netflix_movies_to_db_movies, "wb") as file:
        pickle.dump(found_movies, file)

    # Find netflix movies in database
    # for netflix_movie in 
