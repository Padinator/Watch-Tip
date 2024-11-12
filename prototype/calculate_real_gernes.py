import numpy as np
import pickle
import pymongo


# Define variables
all_actors = {}
all_producers = {}
all_production_companies = {}
all_genres = {}
all_genres_indices = {}


# Connect to MongoDB
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongodb = mongo_client["watch_tip"]
all_movies = mongodb["all_movies"]

# Load all data sets from file/database
# Load and all genres from file
with open("data/all_genres.txt", 'rb') as file:
    all_genres = pickle.loads(file.read())

for i, (genre_id, genre) in enumerate(all_genres.items()):
    print(f"{genre}: {i}")
print()

# Load all movies from database
# all_movies_cursor = all_movies.find({})  # Cursor for iterating over all movies

# Load all updated actors from file
with open("updated_data/all_actors.txt", 'rb') as file:
    all_actors = pickle.loads(file.read())

# Load all updated producer from file
with open("updated_data/all_producers.txt", 'rb') as file:
    all_producers = pickle.loads(file.read())

# Load all updated production companies from file
with open("updated_data/all_production_companies.txt", 'rb') as file:
    all_production_companies = pickle.loads(file.read())


# Calculate real genres
movie_to_get_genre = 1726  # 24: Kill Bill 1, 1726: Iron Man 1, 38757: Rapunzel
movie = all_movies.find_one({"id": movie_to_get_genre})  # Cursor for iterating over all movies

if movie is None:
    print(f"Could not find movie with ID: {movie_to_get_genre}")
    exit(1)
else:
    movie_name = movie["name"]
    print(f"Found movie Iron man 1")

    # movie = all_movies[movie_to_get_genre]
    real_genres_actors = np.zeros(len(all_genres), dtype=np.float64)
    real_genres_producers = np.zeros(len(all_genres), dtype=np.float64)
    real_genres_production_companies = np.zeros(len(all_genres), dtype=np.float64)
    movie_gernes = np.zeros(len(all_genres), dtype=np.float64)
    sum_popularities = 0

    # Find genres of all actors
    for actor_id in movie["credits"]["cast"]:
        actor = all_actors[actor_id]
        real_genres_actors += actor["genres"] / actor["played_movies"]
        real_genres_actors += actor["genres"] / actor["played_movies"] * actor["popularity"]
        # sum_popularities += actor["popularity"]

    # Find genres of all producers
    for producer_id in movie["credits"]["crew"]:
        producer = all_producers[producer_id]
        real_genres_producers += producer["genres"] / producer["produced_movies"]
        real_genres_producers += producer["genres"] / producer["produced_movies"] * producer["popularity"]
        # sum_popularities += producer["popularity"]

    # Find genres of all producers
    for company_id in movie["production_companies"]:
        company = all_production_companies[company_id]
        real_genres_production_companies += company["genres"] / company["financed_movies"]

    # Find genres of the movie
    movie_gernes[movie["genres"]] += 1
    # movie_gernes *= len(movie["credits"]["cast"])
    # movie_gernes *= len(movie["credits"]["cast"]) * sum_popularities / len(movie["credits"]["cast"])

    # Normalize genres
    # movie_gernes /= np.max(movie_gernes)
    real_genres_actors /= np.max(real_genres_actors)
    real_genres_producers /= np.max(real_genres_producers)
    real_genres_production_companies /= np.max(real_genres_production_companies)

    # Output results
    print(f"Movie {movie['original_title']} before:\n{movie['genres']}")
    print(f"Actor genres: {real_genres_actors}")
    print(f"Producer genres: {real_genres_producers}")
    print(f"Production company genres: {real_genres_production_companies}")
    print(f"Movie genres: {movie_gernes}")
    print(f"Sum of genres and actor genres: {(movie_gernes + real_genres_actors + real_genres_producers + real_genres_production_companies) * 100}")
    print(f"Normalized sum of genres and actor genres: {(movie_gernes + real_genres_actors + real_genres_producers + real_genres_production_companies) / 4 * 100}")
