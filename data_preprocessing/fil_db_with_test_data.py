import codecs
import os
import re
import requests
import time

from threading import Thread, Semaphore
from typing import Any, Dict

import variables as vars

from database.movie import Movies
from database.genre import Genres


def request_url(url: str, headers: Dict[str, str]) -> Dict[Any, Any]:
    """Requests the passed URL and returns the response as json = dict,
       if the request was successsful, else return en empty dict '{}'"""

    response = requests.get(url=url, headers=headers)
    return response.json() if response.status_code == 200 else {}


# Define variables for parallel execution
threads = []
COUNT_OF_MAX_RUNNING_THREADS = 16  # Windows: look up in task manager; Linux: look up with command "nproc" (output number of virtual CPUs)
count_of_running_threads_sem = Semaphore(COUNT_OF_MAX_RUNNING_THREADS)
missing_movies_semaphore = Semaphore(1)  # Mutexe for only one thread writing on this file
max_i_mutex = Semaphore(1)

all_actors = {}
all_producers = {}
all_production_companies = {}
all_genres = request_url(vars.genre_url, vars.headers)  # Request all genres defined in TMDB
all_genres = {} if "genres" not in all_genres else dict([(genre["id"], {"name": genre["name"]}) for genre in all_genres["genres"]])
all_genres_indices = dict([(genre_id, i) for i, genre_id in enumerate(all_genres)])  # Map indices of all genres to their IDs

# Read file for index of next movie, if program was interrupted last time
missing_movies_file = "updated_data/fil_db_with_test_data/missing_movies.txt"
index_of_next_movie_file = "updated_data/fil_db_with_test_data/index_of_next_movie.txt"

if os.path.exists(index_of_next_movie_file):
    with open(index_of_next_movie_file, "r") as file:
        index_of_first_movie = int(file.read())
else:
    index_of_first_movie = 0

max_i = 0

# Connect to database
all_movies_table = Movies()
all_genres_table = Genres()

# Start requesting API
start_time = time.time()

with open(vars.local_movie_data_set_path, "rb") as file:
    for i, line in enumerate(file.readlines()):
        def request_movie_by_id(index: int, line: str) -> None:
            """Request movies by id and save important information.
               Parallel execution is possible."""

            global count_of_running_threads_sem, max_i, max_i_mutex

            try:
                line = re.sub(r".*\"id\":", "", line.decode("utf-8"))
                movie_tmdb_id = re.sub(r",.*", "", line)
                url = re.sub("movie_id", f"{movie_tmdb_id}", vars.abstract_movie_url)
                data = request_url(url, vars.headers)

                # Only keep necessary keys
                if data != {}:  # Skip not existing movie
                    # continue  # For single threaded execution
                    raise Exception("Movie does not exist in database")  # Terminate thread (parallel execution with multiple threads)

                # data = {"id": data["id"], "adult": data["adult"], "imdb_id": data["imdb_id"], "original_title": data["original_title"], "overview": data["overview"],
                #         "popularity": data["popularity"], "budget": data["budget"], "genres": data["genres"],
                #         "production_companies": data["production_companies"], "release_date": data["release_date"], "revenue": data["revenue"],
                #         "runtime": data["runtime"], "status": data["status"], "vote_average": data["vote_average"], "vote_count": data["vote_count"],
                #         "credits": data["credits"]}

                # Remove duplicate infos from movie
                data["credits"]["cast"] = [person["id"] for person in data["credits"]["cast"]]
                data["credits"]["crew"] = dict([(person["id"], {"department": person["department"], "job": person["job"]})
                                           for person in data["credits"]["crew"]])
                data["production_companies"] = [company["id"] for company in data["production_companies"]]

                # Convert genre IDs to indices for efficient updating values
                data["genres"] = [all_genres_indices[genre["id"]] for genre in data["genres"]]

                # Add movies to list of all movies
                all_movies_table.insert_one(data)  # Insert into database (parallel execution possible)

                # Save index, if index is maximum index => for saving next index while interrupting program
                max_i_mutex.acquire()
                if max_i < index:
                    max_i = index
                max_i_mutex.release()
            except Exception as ex:
                # Write missing movies into a file
                missing_movies_semaphore.acquire()
                with codecs.open(missing_movies_file, "a", "utf-8") as file:
                    file.write(line)
                missing_movies_semaphore.release()

                raise ex
            finally:
                count_of_running_threads_sem.release()  # Thread is done calculating, another one can be created

        try:
            # Skip already seen movies
            if i < index_of_first_movie:
                continue
            elif i % 1000 == 0:
                print(f"line {i}, {time.time() - start_time}")
                start_time = time.time()

            if i % 10000 == 0:
                print("Little cooldown starts")
                time.sleep(5)
                print("Little cooldown is over")

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
            with open(index_of_next_movie_file, "w") as file:
                file.write(str(max_i + 1))
            # print("Next index", max_i + 1)

            # End program
            print("Terminate program with KeyboardInterrupt")
            exit(0)

# Write all genres
for genre_id, genre in all_genres.items():
    genre["id"] = genre_id
    all_genres_table.insert_one(genre)
