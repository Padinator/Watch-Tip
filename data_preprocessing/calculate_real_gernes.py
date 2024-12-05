import numpy as np
import sys

from threading import Thread, Semaphore
from typing import Any, Dict

# ---------- Import own python files ----------
sys.path.append("../")

import helper.variables as vars

from database.actor import Actors
from database.genre import Genres
from database.movie import Movies
from database.producer import Producers
from database.production_company import ProductionCompanies


# Predfined optins for execution
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


# Define variables for logging errors while computing real genres
error_file_actors_sem = Semaphore(1)  # Mutex for accessing "error_file_actors"
error_file_producers_sem = Semaphore(1)  # Mutex for accessing "error_file_producers"
error_file_production_companies_sem = Semaphore(1)  # Mutex for accessing "error_file_production_companies"

# Define variables for parallel execution
COUNT_OF_MAX_RUNNING_THREADS = 16  # Windows: look up in task manager; Linux: look up with command "nproc" (output number of virtual CPUs)
count_of_running_threads_sem = Semaphore(COUNT_OF_MAX_RUNNING_THREADS)


def calculate_real_genres(all_movies_table: 'Movies', movie_id: int, movie: Dict[str, Any], all_actors: Dict[str, Dict[str, Any]], all_genres: Dict[int, Dict[str, str]],
                          all_producers: Dict[str, Dict[str, Any]], all_production_companies: Dict[str, Dict[str, str]]) -> float:
    """Calculate real genres of a movie, based on the genres of its actors,
    producers and production comanies."""

    global error_file_actors, error_file_producers, error_file_production_companies,\
        error_file_actors_sem, error_file_producers_sem, error_file_production_companies_sem,\
        count_of_running_threads_sem

    # Define variables for computing and storing real genres
    real_genres_actors = np.zeros(len(all_genres), dtype=np.float64)
    real_genres_producers = np.zeros(len(all_genres), dtype=np.float64)
    real_genres_production_companies = np.zeros(len(all_genres), dtype=np.float64)
    movie_gernes = np.zeros(len(all_genres), dtype=np.float64)
    # sum_popularities = 0

    # Find genres of all actors
    for actor_id in movie["credits"]["cast"]:
        try:
            actor = all_actors[actor_id]
            real_genres_actors += np.array(actor["genres"], dtype=np.float64) / actor["played_movies"]
            # real_genres_actors += actor["genres"] / actor["played_movies"] * actor["popularity"]
            # sum_popularities += actor["popularity"]
        except Exception as ex:
            error_file_actors_sem.acquire()
            with open(vars.calc_real_genres_error_file_actors, "a", encoding="utf-8") as file:
                file.write(f"Error: {str(ex), {movie_id}}\n")
            error_file_actors_sem.release()

    # Find genres of all producers
    for producer_id in movie["credits"]["crew"]:
        try:
            producer_id = int(producer_id)
            producer = all_producers[producer_id]
            real_genres_producers += np.array(producer["genres"], dtype=np.float64) / producer["produced_movies"]
            # real_genres_producers += producer["genres"] / producer["produced_movies"] * producer["popularity"]
            # sum_popularities += producer["popularity"]
        except Exception as ex:
            error_file_producers_sem.acquire()
            with open(vars.calc_real_genres_error_file_producers, "a", encoding="utf-8") as file:
                file.write(f"Error: {str(ex), {movie_id}}\n")
            error_file_producers_sem.release()

    # Find genres of all producers
    for company_id in movie["production_companies"]:
        try:
            company = all_production_companies[company_id]
            real_genres_production_companies += np.array(company["genres"], dtype=np.float64) / company["financed_movies"]
        except Exception as ex:
            error_file_production_companies_sem.acquire()
            with open(vars.calc_real_genres_error_file_production_companies, "a", encoding="utf-8") as file:
                file.write(f"Error: {str(ex), {movie_id}}\n")
            error_file_production_companies_sem.release()

    # Find genres of the movie
    movie_gernes[movie["genres"]] += 1
    # movie_gernes *= len(movie["credits"]["cast"])
    # movie_gernes *= len(movie["credits"]["cast"]) * sum_popularities / len(movie["credits"]["cast"])

    # Normize genres
    # movie_gernes /= np.max(movie_gernes)  # Is already normized
    real_genres_actors /= np.max(real_genres_actors)
    real_genres_producers /= np.max(real_genres_producers)
    real_genres_production_companies /= np.max(real_genres_production_companies)
    real_genres = (movie_gernes + real_genres_actors + real_genres_producers + real_genres_production_companies) / 4 * 100

    # Output results
    # print(f"Movie {movie['original_title']} before:\n{movie['genres']}")
    # print(f"Actor genres: {real_genres_actors}")
    # print(f"Producer genres: {real_genres_producers}")
    # print(f"Production company genres: {real_genres_production_companies}")
    # print(f"Movie genres: {movie_gernes}")
    # print(f"Sum of genres and actor genres: {(movie_gernes + real_genres_actors + real_genres_producers + real_genres_production_companies) * 100}")
    # print(f"Normized sum of genres and actor genres: {(movie_gernes + real_genres_actors + real_genres_producers + real_genres_production_companies) / 4 * 100}")

    # Save real genres in database
    res = all_movies_table.update_one_by_attr("id", movie_id, "real_genres", real_genres.tolist())
    count_of_running_threads_sem.release()  # Thread is done calculating, another one can be created
    return res


if __name__ == "__main__":
    # Connect to database
    all_movies_table = Movies()
    all_genres_table = Genres()
    all_actors_table = Actors()
    all_producers_table = Producers()
    all_production_companies_table = ProductionCompanies()

    # Read all entries from database
    # Some movie IDs: 24 = Kill Bill 1; 1726 = Iron Man 1; 38757 = Rapunzel
    all_movies = all_movies_table.get_all()
    all_genres = all_genres_table.get_all()
    all_actors = all_actors_table.get_all()
    all_producers = all_producers_table.get_all()
    all_production_companies = all_production_companies_table.get_all()

    # Calculate real genres for all movies
    i = 0

    for movie_id, movie in all_movies.items():  # Cursor for iterating over all movies
        if i % 100000 == 0:
            print(f"Iteration {i}")

        count_of_running_threads_sem.acquire()
        t = Thread(target=calculate_real_genres, args=[all_movies_table, movie_id, movie, all_actors, all_genres, all_producers, all_production_companies])
        t.start()
        i += 1
