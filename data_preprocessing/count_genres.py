import copy as cp
import json
import numpy as np
import pickle
import pymongo
import re


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


# Load and all genres from file
print("\nRead all genres")
with open("data/all_genres.txt", 'rb') as file:
    all_genres = pickle.loads(file.read())

genre_counters = np.zeros(len(all_genres))  # Add to each actor, producer and company all genres dict of counters starting with zero

# Load all movies from database
print("\nRead all movies")
all_movies_cursor = all_movies.find({})  # Cursor for iterating over all movies

# Load all actors from file
print("\nRead all actors")

with open("data/person_ids_09_23_2024.json", 'rb') as file:
    for i, line in enumerate(file.readlines()):
        if i % 500000 == 0:
            print(f"Iteration: {i}")
        actor = json.loads(line.decode("utf-8"))
        actor["id"] = int(actor["id"])
        actor["popularity"] = float(actor["popularity"])
        actor["genres"] = np.copy(genre_counters)
        actor["played_movies"] = 0

        # Save actor in dict of all actors
        actor_id = actor["id"]
        # del actor["id"]
        all_actors[actor_id] = actor

# Load all producer from file
print("\nRead all producers")
all_producers = cp.deepcopy(all_actors)

for i, (producer_id, producer) in enumerate(all_producers.items()):
    if i % 500000 == 0:
        print(f"Iteration: {i}")
    producer["produced_movies"] = 0  # Add counter for produced movies
    del producer["played_movies"]  # Remove counter for played movies

# Load all production companies from file
print("\nRead all production companies")

with open("data/production_company_ids_09_23_2024.json", 'rb') as file:  # {"id":1,"name":"Lucasfilm Ltd."}
    for line in file.readlines():
        if i % 50000 == 0:
            print(f"Iteration: {i}")
        company = json.loads(line.decode("utf-8"))
        company["genres"] = np.copy(genre_counters)
        company["financed_movies"] = 0  # Add counter for financed movies

        # Save producer in dict of all production companies
        company_id = company["id"]
        # del company["id"]
        all_production_companies[company_id] = company


# Count genres for actors from movies
print("\nCount real genres")
skipped_movies = 0
i = 0

for movie in all_movies_cursor:
    if i % 10000 == 0:
        print(f"Iteration: {i}")

    movie_genres = movie["genres"]

    if len(movie_genres) == 0:  # Skip empty array
        skipped_movies += 1
        movie_str = str(movie)
        with open("updated_data/skipped_movies.txt", "w", encoding="utf-8") as file:
            file.write(f"{movie_str}\n")
        continue

    # Count genres of all actors of a movie
    for actor_id in movie["credits"]["cast"]:
        try:
            all_actors[actor_id]["genres"][movie_genres] += 1  # Increment counters of each genre
            all_actors[actor_id]["played_movies"] += 1
        except Exception as e:
            if actor_id not in all_actors:
                pass
            else:
                print(e)
                print(actor_id)
                print(all_actors[actor_id])
                print(all_actors[actor_id]["genres"])
                raise e

    # Count genres of all producers of a movie
    for producer_id in movie["credits"]["crew"]:
        try:
            all_producers[producer_id]["genres"][movie_genres] += 1  # Increment counters of each genre
            all_producers[producer_id]["produced_movies"] += 1
        except Exception as e:
            if producer_id not in all_producers:
                pass
            else:
                print(e)
                print(producer_id)
                print(all_producers[producer_id])
                print(all_producers[producer_id]["genres"])
                raise e

    # Count genres of all production companies of a movie
    for company_id in movie["production_companies"]:
        try:
            all_production_companies[company_id]["genres"][movie_genres] += 1  # Increment counters of each genre
            all_production_companies[company_id]["financed_movies"] += 1
        except Exception as e:
            if company_id not in all_production_companies:
                pass
            else:
                print(e)
                print(company_id)
                print(all_production_companies[company_id])
                print(all_production_companies[company_id]["genres"])
                raise e

    i += 1

print(f"Skipped {skipped_movies} movies")


# Write all updated data sets
'''
with open("updated_data/all_actors.txt", 'wb') as file:
    pickle.dump(all_actors, file)

with open("updated_data/all_producers.txt", 'wb') as file:
    pickle.dump(all_producers, file)

with open("updated_data/all_production_companies.txt", 'wb') as file:
    pickle.dump(all_production_companies, file)
'''

# Write all genres
all_genres_db = mongodb["all_genres"]

for genre_id, genre in all_genres.items():
    genre["id"] = genre_id
    all_genres_db.insert_one(genre)

# Write all updated actors
all_actors_collection = mongodb["all_actors"]

for i, (actor_id, actor) in enumerate(all_actors.items()):
    if i % 500000 == 0:
        print(f"Iteration: {i}")
    actor["id"] = actor_id
    actor["genres"] = actor["genres"].tolist()
    all_actors_collection.insert_one(actor)

# Write all updated producers
all_producers_collection = mongodb["all_producers"]

for i, (producer_id, producer) in enumerate(all_producers.items()):
    if i % 500000 == 0:
        print(f"Iteration: {i}")
    producer["id"] = producer_id
    producer["genres"] = producer["genres"].tolist()
    all_producers_collection.insert_one(producer)

# Write all updated production companies
all_production_companies_collection = mongodb["all_production_companies"]

for i, (company_id, company) in enumerate(all_production_companies.items()):
    if i % 50000 == 0:
        print(f"Iteration: {i}")
    company["id"] = company_id
    company["genres"] = company["genres"].tolist()
    all_production_companies_collection.insert_one(company)
