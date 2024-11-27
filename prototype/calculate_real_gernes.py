import copy as cp
import numpy as np
import pickle
import pymongo

from typing import Any, Dict


# Predfined optins for execution
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def calculate_real_genres(movie: Dict[str, Any], all_actors: Dict[str, Dict[str, Any]], all_producers: Dict[str, Dict[str, Any]],
                          all_production_companies: Dict[str, Dict[str, str]]) -> float:
    """Calculate real genres of a movie, based on the genres of its actors,
    producers and production comanies."""

    movie_name = movie["original_title"]
    movie_id = movie["_id"]
    # print(f"Calculate real genres of {movie_name}")

    # movie = all_movies[movie_to_get_genre]
    real_genres_actors = np.zeros(len(all_genres), dtype=np.float64)
    real_genres_producers = np.zeros(len(all_genres), dtype=np.float64)
    real_genres_production_companies = np.zeros(len(all_genres), dtype=np.float64)
    movie_gernes = np.zeros(len(all_genres), dtype=np.float64)
    # sum_popularities = 0

    # Find genres of all actors
    for actor_id in movie["credits"]["cast"]:
        try:
            actor = all_actors[actor_id]
            real_genres_actors += actor["genres"] / actor["played_movies"]
        except Exception as ex:
            with open("updated_data/error_calc_real_genres_actors.txt", "a") as file:
                file.write(f"Error: {str(ex), {movie_id}}\n")
        # real_genres_actors += actor["genres"] / actor["played_movies"] * actor["popularity"]
        # sum_popularities += actor["popularity"]

    # Find genres of all producers
    for producer_id in movie["credits"]["crew"]:
        try:
            producer = all_producers[producer_id]
            real_genres_producers += producer["genres"] / producer["produced_movies"]
        except Exception as ex:
            with open("updated_data/error_calc_real_genres_producers.txt", "a") as file:
                file.write(f"Error: {str(ex), {movie_id}}\n")
        # real_genres_producers += producer["genres"] / producer["produced_movies"] * producer["popularity"]
        # sum_popularities += producer["popularity"]

    # Find genres of all producers
    for company_id in movie["production_companies"]:
        try:
            company = all_production_companies[company_id]
            real_genres_production_companies += company["genres"] / company["financed_movies"]
        except Exception as ex:
            with open("updated_data/error_calc_real_genres_production_companies.txt", "a") as file:
                file.write(f"Error: {str(ex), {movie_id}}\n")

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
    # print(f"Normalized sum of genres and actor genres: {(movie_gernes + real_genres_actors + real_genres_producers + real_genres_production_companies) / 4 * 100}")

    return real_genres


def read_from_database_as_dict(db: pymongo.synchronous.database.Database, collection_name: str) -> Dict[int, Any]:
    """Reads a collection from database and passes user a deep
       copy for not manipulating the data by accident."""

    data = {}

    for entity in db[collection_name].find():  # Cursor for iterating over all genres
        entity_id = entity["id"]
        entity_copied = cp.copy(entity)
        del entity_copied["_id"]
        del entity_copied["id"]
        data[entity_id] = entity_copied

    return data


if __name__ == "__main__":
    # Define variables
    all_actors = {}
    all_producers = {}
    all_production_companies = {}
    all_genres = {}
    all_genres_indices = {}
    real_genres_of_a_movie = {}

    # Connect to MongoDB
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    mongodb = mongo_client["watch_tip"]
    all_movies = mongodb["all_movies"]

    # Load all data sets from file/database
    all_genres = read_from_database_as_dict(mongodb, "all_genres")  # Load all genres
    all_actors = read_from_database_as_dict(mongodb, "all_actors")  # Load all actors
    all_producers = read_from_database_as_dict(mongodb, "all_producers")  # Load all producers
    all_production_companies = read_from_database_as_dict(mongodb, "all_production_companies")  # Load all production companies

    # Calculate real genres for one movie
    # movie_to_get_genre = 1726  # 24: Kill Bill 1, 1726: Iron Man 1, 38757: Rapunzel
    # movie = all_movies.find_one({"id": movie_to_get_genre})  # Cursor for iterating over all movies

    # Calculate real genres for all movies
    i = 0

    for movie in all_movies.find():  # Cursor for iterating over all movies
        if i % 100000 == 0:
            print(f"Iteration {i}")

        real_genres_of_a_movie = calculate_real_genres(movie, all_actors, all_producers, all_production_companies)
        movie["real_genres"] = real_genres_of_a_movie
        res = all_movies.find_one_and_update({"_id": movie["_id"]}, {"$set" : {"real_genres": real_genres_of_a_movie.tolist() }})
        i += 1
