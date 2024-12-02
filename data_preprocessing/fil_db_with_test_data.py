import codecs
import os
import pickle
import re
import time

from threading import Thread, Semaphore
from typing import Any, Dict

import helper.variables as vars

from database.movie import Movies
from database.genre import Genres
from database.user import Users
from helper.api_requester import request_url, request_movie_reviews, request_movie
from helper.file_system_interaction import save_object_in_file
from helper.parallelizer import parallelize_task_with_return_values




from database.actor import Actors
from database.producer import Producers
from database.production_company import ProductionCompanies



# Define constants
COUNT_OF_MAX_RUNNING_THREADS =  16  # Windows: look up in task manager; Linux: look up with command "nproc" (output number of virtual CPUs)

# Define variables for parallel execution
threads = []
count_of_running_threads_sem = Semaphore(COUNT_OF_MAX_RUNNING_THREADS)
missing_movies_semaphore = Semaphore(1)  # Mutexe for only one thread writing on this file
max_i_mutex = Semaphore(1)
all_reviews_sem = Semaphore(1)

all_actors, all_actors_ids = {}, []
all_producers, all_producers_ids = {}, []
all_production_companies, all_production_companies_ids = {}, []
all_genres = request_url(vars.genre_url, vars.headers)  # Request all genres defined in TMDB
all_genres = {} if "genres" not in all_genres else dict([(genre["id"], {"name": genre["name"]}) for genre in all_genres["genres"]])
all_genres_indices = dict([(genre_id, i) for i, genre_id in enumerate(all_genres)])  # Map indices of all genres to their IDs
all_reviews = {}


# -------- Define methods for requesting data from API and storing it --------
def merge_user_reviews_from_different_movies(user_reviews: Dict[str, Any], new_reviews: Dict[str, Any]):
    """
        Merge new reviews into dictionary of old/other reviews and return this new dictionary.
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
    """Request movies by id and save important information.
        Parallel execution is possible."""

    global count_of_running_threads_sem, max_i, max_i_mutex, all_reviews

    try:
        line = line.decode("utf-8")
        shortened_line = re.sub(r".*\"id\":", "", line)
        movie_tmdb_id = re.sub(r",.*", "", shortened_line).strip()

        # Request API for movie data
        url = re.sub("movie_id", f"{movie_tmdb_id}", vars.abstract_movie_url)
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
        url = re.sub("movie_id", f"{movie_tmdb_id}", vars.abstract_movie_reviews_url)
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




# # Find all missing actors and add them to database
# # calc_real_genres_error_file_actors
# # calc_real_genres_error_file_producers
# # calc_real_genres_error_file_production_companies
# with open(vars.calc_real_genres_error_file_production_companies, "r") as file:
#     lines = file.readlines()
# missing_ids = [re.sub(r"'.*", "", re.sub(r".*\('", "", line)).strip() for line in lines]

# print(len(missing_ids))
# missing_ids = list(set(missing_ids))
# print(len(missing_ids))
# # url = "https://api.themoviedb.org/3/person/replace_id?language=en-US"
# url = "https://api.themoviedb.org/3/company/replace_id"


# def foo(id, url):
#     entitiy_url = re.sub("replace_id", f"{id}", url)
#     entitiy = request_url(entitiy_url)
#     try:
#         # entitiy = {"adult": entitiy["adult"], "id": entitiy["id"], "name": entitiy["name"], "popularity": entitiy["popularity"]}
#         entitiy = {"id": entitiy["id"], "name": entitiy["name"]}
#     except Exception as ex:
#         print(ex, id, entitiy)
#         raise ex
#     return entitiy

# entities = parallelize_task_with_return_values(foo, [(id, url) for id in missing_ids], 16)

# # local_producers_and_actors_data_set_path
# # local_producer_company_data_set_path
# with open(vars.local_producer_company_data_set_path, "a", encoding="utf-8") as file:
#     for _, entity in entities.items():
#         # adult = "true" if entity["adult"] else "false"
#         id = entity["id"]
#         name = entity["name"]
#         # popularity = entity["popularity"]
#         # line = "{" + f"\"adult\":{adult},\"id\":{id},\"name\":\"{name}\",\"popularity\":{popularity}" + "}\n"
#         line = "{" + f"\"id\":{id},\"name\":\"{name}\"" + "}\n"
#         file.write(line)

# exit()





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

# Connect to database
all_movies_table = Movies()
all_genres_table = Genres()
all_users_table = Users()

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
                print(f"Iteration {i}:", sum([len(reviews) for reviews in all_reviews.values()]),
                      sum([len(reviews) for reviews in all_reviews.values()]) / len(all_reviews),
                      max([len(reviews) for reviews in all_reviews.values()]))
                all_reviews_sem.release()

            # Request database parallely
            count_of_running_threads_sem.acquire()  # Acquire one thread for running
            t = Thread(target=request_movie_by_id, args=[i, line])
            threads.append(t)
            t.start()
        except KeyboardInterrupt as ex:
            # Wait for temrinating all threads
            for t in threads[-COUNT_OF_MAX_RUNNING_THREADS:]:
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
for t in threads[-COUNT_OF_MAX_RUNNING_THREADS:]:
    t.join()

# Save all actors' IDs in a file
save_object_in_file(vars.local_actors_ids_file_path, all_actors_ids)

# Save all producers' IDs in a file
save_object_in_file(vars.local_producers_ids_file_path, all_producers_ids)

# Save all production companies' IDs in a file
save_object_in_file(vars.local_producction_companies_ids_file_path, all_production_companies_ids)

# Save all genres in database
for genre_id, genre in all_genres.items():
    genre["id"] = genre_id
    all_genres_table.insert_one(genre)

# Save all user reviews in database
for user, user_reviews in all_reviews.items():
    user_reviews = dict(sorted(user_reviews.items()))
    user_reviews["user"] = user
    all_users_table.insert_one(user_reviews)

# Read all movies and find all actors (comming soon)
# ...
# with open(vars.local_actors_ids_file_path, "wb") as file:
#     pickle.dump(all_actors_ids, file)
