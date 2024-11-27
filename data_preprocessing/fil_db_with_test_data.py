import configparser
import json
import numpy as np
import os
import pickle
import pymongo
import re
import requests
import time

from threading import Thread, Semaphore


# Define variables
# url_data_all_movies = "http://files.tmdb.org/p/exports/movie_ids_09_23_2024.json.gz"
# url_data_all_producers_and_actors = "http://files.tmdb.org/p/exports/person_ids_09_23_2024.json.gz"
# url_data_all_companies = "http://files.tmdb.org/p/exports/production_company_ids_09_23_2024.json.gz"
api_key = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiNGY1YTQ0ZWM2NGM0YTY3NWY0NTJmMTFmMmVhY2QxYyIsIm5iZiI6MTczMTI0NzU1OC4wOTcxOTk0LCJzdWIiOiI2NmVkNjI0MzU3NmUyY2NhMWFmZTA2NWUiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.WPajvzzzoD8U1vqoJuwpkKFcczmFHZOFzb_1xhpDryM"
abstract_url = "https://api.themoviedb.org/3/movie/movie_id?append_to_response=credits&language=en-US"
headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Define variables for parallel execution
threads = []
COUNT_OF_MAX_RUNNING_THREADS = 8  # Windows: look up in task manager; Linux: look up with command "nproc" (output number of virtual CPUs)
count_of_running_threads_sem = Semaphore(COUNT_OF_MAX_RUNNING_THREADS)
missing_movies_file = "data/missing_movies.txt"
missing_movies_semaphore = Semaphore(1)  # Mutexe for only one thread writing on this file
max_i_mutex = Semaphore(1)

# Read data from ".ini"-file and write to database
# config = configparser.ConfigParser()
# config.read('movies.ini')

# all_movies = {}
all_actors = {}
all_producers = {}
all_production_companies = {}
all_genres = {  # All genres defined in TMDB
    28: {"name": "Action"},
    12: {"name": "Adventure"},
    16: {"name": "Animation"},
    35: {"name": "Comedy"},
    80: {"name": "Crime"},
    99: {"name": "Documentary"},
    18: {"name": "Drama"},
    10751: {"name": "Family"},
    14: {"name": "Fantasy"},
    36: {"name": "History"},
    27: {"name": "Horror"},
    10402: {"name": "Music"},
    9648: {"name": "Mystery"},
    10749: {"name": "Romance"},
    878: {"name": "Science Fiction"},
    10770: {"name": "TV Movie"},
    53: {"name": "Thriller"},
    10752: {"name": "War"},
    37: {"name": "Western"},
}
all_genres_indices = dict([(genre_id, i) for i, genre_id in enumerate(all_genres)])  # Map indices of all genres to their IDs
index_of_next_movie_file = "data/index_of_next_movie.txt"

if os.path.exists(index_of_next_movie_file):
    with open(index_of_next_movie_file, "r") as file:
        index_of_first_movie = int(file.read())
else:
    index_of_first_movie = 0

max_i = 0

# Connect to MongoDB
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongodb = mongo_client["watch_tip"]
all_movies = mongodb["all_movies"]

# for movie_section in config.sections():
#     for movie_name, movie_tmdb_id in config.items(movie_section):
start_time = time.time()

with open("data/movie_ids_09_23_2024.json", "rb") as file:
    for i, line in enumerate(file.readlines()):
        def request_movie_by_id(index: int, line: str) -> None:
            """Request movies by id and save important information.
               Parallel execution is possible."""

            global count_of_running_threads_sem, max_i, max_i_mutex

            try:
                line = re.sub(r".*\"id\":", "", line.decode("utf-8"))
                movie_tmdb_id = re.sub(r",.*", "", line)
                url = re.sub("movie_id", f"{movie_tmdb_id}", abstract_url)
                response = requests.get(url, headers=headers)
                data = response.json()

                # Only keep necessary keys
                if "status_code" in data and data["status_code"] != 0:  # Skip not existing movie
                    # continue  # For single threaded execution
                    exit()  # For parallel execution with multiple threads 

                # data = {"id": data["id"], "adult": data["adult"], "imdb_id": data["imdb_id"], "original_title": data["original_title"], "overview": data["overview"],
                #         "popularity": data["popularity"], "budget": data["budget"], "genres": data["genres"],
                #         "production_companies": data["production_companies"], "release_date": data["release_date"], "revenue": data["revenue"],
                #         "runtime": data["runtime"], "status": data["status"], "vote_average": data["vote_average"], "vote_count": data["vote_count"],
                #         "credits": data["credits"]}

                # Remove duplicate infos from movie
                data["credits"]["cast"] = [{"id": person["id"], "known_for_department": person["known_for_department"], "popularity": person["popularity"],
                                            "profile_path": person["profile_path"] } for person in data["credits"]["cast"]]
                data["credits"]["crew"] = [{"id": person["id"], "known_for_department": person["known_for_department"], "popularity": person["popularity"],
                                            "profile_path": person["profile_path"], "department": person["department"], "job": person["job"]
                                            } for person in data["credits"]["crew"]]
                data["production_companies"] = [company["id"] for company in data["production_companies"]]

                # Convert genre IDs to indices for efficient updating values
                data["genres"] = [all_genres_indices[genre["id"]] for genre in data["genres"]]

                # Add movies to list of all movies
                all_movies.insert_one(data)  # Insert into database (parallel execution possible)

                # Save index, if index is maximum index => for savin next index while interrupting program
                max_i_mutex.acquire()
                if max_i < index:
                    max_i = index
                max_i_mutex.release()
            except Exception as ex:
                # Write missing movies into a file
                missing_movies_semaphore.acquire()
                with open(missing_movies_file, "a") as file:
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

# Write all genres to file
with open("data/all_genres.txt", 'wb') as file:
    pickle.dump(all_genres, file)
