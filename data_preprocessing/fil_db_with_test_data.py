import codecs
import os
import pickle
import re
import time
import sys

from pathlib import Path
from threading import Thread, Semaphore
from typing import Any, Dict, List

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[1]
sys.path.append(str(project_dir))

import helper.variables as vars

from database.movie import Movies
from database.genre import Genres
from database.user import Users
from helper.api_requester import request_url, request_movie_reviews, request_movie
from helper.file_system_interaction import (
    load_json_objects_from_file,
    save_json_objects_in_file,
)


# Define constants
NUMBER_OF_MAX_RUNNING_THREADS = 16

# Define variables for parallel execution
threads = []
count_of_running_threads_sem = Semaphore(NUMBER_OF_MAX_RUNNING_THREADS)
missing_movies_semaphore = Semaphore(1)  # Mutexe for only one thread writing on this file
max_i_mutex = Semaphore(1)
all_reviews_sem = Semaphore(1)

all_actors, all_actors_ids = {}, []
all_producers, all_producers_ids = {}, []
all_production_companies, all_production_companies_ids = {}, []
all_genres = request_url(vars.genre_url, vars.headers)  # Request all genres defined in TMDB
all_genres = (
    {} if "genres" not in all_genres else dict([(genre["id"], {"name": genre["name"]}) for genre in all_genres["genres"]])
)
all_genres_indices = dict(
    [(genre_id, i) for i, genre_id in enumerate(all_genres)]
)  # Map indices of all genres to their IDs
all_reviews = {}


# -------- Define methods for requesting data from API and storing it --------
def merge_user_reviews_from_different_movies(user_reviews: Dict[str, Any], new_reviews: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge new reviews into dictionary of old/other reviews and return this
    dictionary. This function works with on reference objects for parameters
    and return values.

    Parameters
    ----------
    user_reviews: Dict[str, Any]
        The existing user reviews to which new one will be added
    new_reviews : Dict[str, Any]
        New user reviews which will be added to existing ones

    Returns
    -------
    Dict[str, Any]
        Extended user reviews, old object will be extended
    """

    if new_reviews != {}:
        for user_name, review in new_reviews.items():
            if user_name not in user_reviews:
                user_reviews[user_name] = {}

            time_watched = review["created_at"]
            del review["created_at"]
            user_reviews[user_name][time_watched] = review

    return user_reviews


def request_movie_by_id(index: int, line: str) -> None:
    """
    Request movies by ID and save important information about movies and their
    reviews in database. It also requests the reviews of a movie.\n
    Parallel execution will be used.\n
    Thhis function can be interrupted and started to another time. Then it
    will continue, where it has stopped. The temporary inbetween results will
    be stored in some temporary files.

    Parameters
    ----------
    index : int
        Index of parameter line in file of all TMDB movies. It will be used
        for remebering, when the program was stopped and where to continue
        next time
    line: str
        Contains information about a movie, espicially the movie ID of the
        database TMDB. Use this for requesting the movie with the help of the
        TMDB API
    """

    global count_of_running_threads_sem, max_i, max_i_mutex, all_reviews, all_actors_ids, all_producers_ids, all_production_companies_ids, all_genres_indices

    try:
        line = line.decode("utf-8")
        shortened_line = re.sub(r".*\"id\":", "", line)
        movie_tmdb_id = re.sub(r",.*", "", shortened_line).strip()

        # Request API for movie data
        url = re.sub("replace_id", f"{movie_tmdb_id}", vars.abstract_movie_url)
        movie = request_movie(url, movie_tmdb_id, vars.headers)

        # Convert genre IDs to indices for efficient updating values
        movie["genres"] = [all_genres_indices[genre["id"]] for genre in movie["genres"]]

        # Save all actors', producers' and production companies' IDs
        all_actors_ids.extend(movie["credits"]["cast"])
        all_producers_ids.extend(list(movie["credits"]["crew"].keys()))
        all_production_companies_ids.extend(movie["production_companies"])

        # Save movie in database
        all_movies_table.insert_one(movie)

        # Request API for movie reviews
        url = re.sub("replace_id", f"{movie_tmdb_id}", vars.abstract_movie_reviews_url)
        movie_reviews = request_movie_reviews(url, movie_tmdb_id, 1)

        # Merge all perviuos reviews with new ones
        try:
            all_reviews_sem.acquire()
            all_reviews = merge_user_reviews_from_different_movies(all_reviews, movie_reviews)
        finally:
            all_reviews_sem.release()

        # Synchronize results with other threads
        max_i_mutex.acquire()  # Save index, if index is maximum index => for saving next index while interrupting program
        if max_i < index:
            max_i = index
        max_i_mutex.release()
    except Exception as ex:
        # Write missing movies into a file
        missing_movies_semaphore.acquire()
        with codecs.open(vars.missing_movies_file, "a", "utf-8") as file:
            file.write(line)
        missing_movies_semaphore.release()

        print(ex)
    finally:
        count_of_running_threads_sem.release()  # Thread is done calculating, another one can be created


# Find all relevant actors and save them
def find_all_necessary_entities(
    all_entity_ids_from_first_src: List[str],
    all_entity_ids_from_second_src: List[str],
    all_entities_from_second_src: List[Dict[str, Any]],
    necessary_keys: List[str],
    url: str,
) -> Dict[str, Any]:
    """
    Searches in second passed source for IDs in first source. Similiar IDs
    mean similiar object, so that these object from
    "all_entities_from_second_src" will be returned. The IDs, which are only
    in the first passed list of IDs, will be requested per API call with
    passed url "url". Only passed keys "necessary_keys" will be saved from
    requested entities.

    Parameters
    ----------
    all_entity_ids_from_first_src : List[str]
        IDs of entities in first source
    all_entity_ids_from_second_src : List[str]
        IDs of entities in second source
    all_entities_from_second_src : List[Dict[str, Any]]
        Entities form second source
    necessary_keys : List[str]
        List of necessary keys to remain from requested entries (API requests)
    url : str
        URL to request new entries from API

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Returns all entities as dict with ID as first key from first source.
    """

    all_entities_ids_in_both_srcs = set(all_entity_ids_from_first_src) & set(all_entity_ids_from_second_src)
    missing_entity_ids_to_request = set(all_entity_ids_from_first_src) - all_entities_ids_in_both_srcs
    all_entities = {}

    # Collect all entities, who are listed in both sources
    for entity_id in all_entities_ids_in_both_srcs:
        all_entities[entity_id] = all_entities_from_second_src[entity_id]

    # Request missing entities from API by their ID
    for entity_id in missing_entity_ids_to_request:
        entity_url = re.sub("replace_id", f"{entity_id}", url)
        entity = request_url(entity_url, vars.headers)
        all_entities[entity_id] = dict([(key, entity[key]) for key in necessary_keys])

    return all_entities


if __name__ == "__main__":
    # Connect to database
    all_movies_table = Movies()
    all_genres_table = Genres()
    all_users_table = Users()

    # Read file for index of next movie, if program was interrupted last time
    if os.path.exists(vars.index_of_next_movie_file):
        with open(vars.index_of_next_movie_file, "r") as file:
            index_of_first_movie = int(file.read())
    else:
        index_of_first_movie = 0

    if os.path.exists(vars.all_reviews_tmp_data_file):
        with open(vars.all_reviews_tmp_data_file, "rb") as file:
            all_reviews = pickle.load(file)
    else:
        all_reviews = {}

    max_i = 0

    # Start requesting API
    start_time = time.time()

    with open(vars.local_movie_data_set_path, "rb") as file:
        for i, line in enumerate(file.readlines()):
            try:
                # Skip already seen movies
                if i < index_of_first_movie:
                    continue
                elif i % 1000 == 0:
                    print(f"line {i}, {time.time() - start_time}")
                    start_time = time.time()

                # Save each 5,000 iterations all user reviews into a file
                if i % 5000 == 0:
                    all_reviews_sem.acquire()
                    with open(vars.all_reviews_tmp_data_file, "wb") as file:
                        pickle.dump(all_reviews, file)
                    print(
                        f"Iteration {i}:",
                        sum([len(reviews) for reviews in all_reviews.values()]),
                        sum([len(reviews) for reviews in all_reviews.values()]) / len(all_reviews),
                        max([len(reviews) for reviews in all_reviews.values()]),
                    )
                    all_reviews_sem.release()

                # Request database parallely
                count_of_running_threads_sem.acquire()  # Acquire one thread for running
                t = Thread(target=request_movie_by_id, args=[i, line])
                threads.append(t)
                t.start()
            except KeyboardInterrupt:
                # Wait for temrinating all threads
                for t in threads[-NUMBER_OF_MAX_RUNNING_THREADS:]:
                    t.join()

                # Update next index
                with open(vars.index_of_next_movie_file, "w") as file:
                    file.write(str(max_i + 1))
                # print("Next index", max_i + 1)

                # Save all current reviews
                with open(f"{vars.all_reviews_tmp_data_file}", "wb") as file:
                    pickle.dump(all_reviews, file)

                # End program
                print("Terminate program with KeyboardInterrupt")
                exit(0)

        # Save all collected reviews
        with open(f"{vars.all_reviews_tmp_data_file}", "wb") as file:
            pickle.dump(all_reviews, file)

    # Wait for last threads to end
    for t in threads[-NUMBER_OF_MAX_RUNNING_THREADS:]:
        t.join()

    # Save all genres in database
    for genre_id, genre in all_genres.items():
        genre["id"] = genre_id
        all_genres_table.insert_one(genre)

    # Save all user reviews in database
    for user, user_reviews in all_reviews.items():
        user_reviews = dict(sorted(user_reviews.items()))
        user_reviews["user"] = user
        all_users_table.insert_one(user_reviews)


    # Find all relevant actors and producers and save their data
    important_person_keys = ["adult", "id", "name", "popularity"]
    all_persons = load_json_objects_from_file(vars.local_producers_and_actors_data_set_path, important_person_keys)
    all_persons_ids = list(all_persons.keys())

    # Find all relevant actors and producers
    all_actors = find_all_necessary_entities(
        all_actors_ids,
        all_persons_ids,
        all_persons,
        important_person_keys,
        vars.abstract_person_url,
    )
    all_producers = find_all_necessary_entities(
        all_producers_ids,
        all_persons_ids,
        all_persons,
        important_person_keys,
        vars.abstract_person_url,
    )

    # Find all relevant production companies and save their data
    important_production_company_keys = ["id", "name"]
    all_production_companies_from_file = load_json_objects_from_file(
        vars.local_producer_company_data_set_path, important_production_company_keys
    )
    all_production_companies_ids_from_file = list(all_production_companies_from_file.keys())
    all_production_companies = find_all_necessary_entities(
        all_production_companies_ids,
        all_production_companies_ids_from_file,
        all_production_companies_from_file,
        important_production_company_keys,
        vars.abstract_production_company_url,
    )

    # Save all actors, producers and production companies in a file
    save_json_objects_in_file(vars.local_actors_file_path, all_actors)
    save_json_objects_in_file(vars.local_producers_file_path, all_producers)
    save_json_objects_in_file(vars.local_production_companies_file_path, all_production_companies)
