import copy as cp
import json
import numpy as np

import variables as vars

from database.actor import Actors
from database.genre import Genres
from database.movie import Movies
from database.producer import Producers
from database.production_company import ProductionCompanies


# Define variables
all_actors = {}
all_producers = {}
all_production_companies = {}
all_genres_indices = {}
skipped_movies_file_path = "updated_data/count_genres/skipped_movies.txt"

# Connect to database
all_movies_table = Movies()
all_genres_table = Genres()
all_actors_table = Actors()
all_producers_table = Producers()
all_production_companies_table = ProductionCompanies()

# Read all genres from database
print("Read all genres from database")
all_genres = all_genres_table.get_all()
genre_counters = np.zeros(len(all_genres), dtype=np.float64)  # Add to each actor, producer and company all genres dict of counters starting with zero

# Read all movies from database
print("\nRead all movies from database")
all_movies = all_movies_table.get_all()  # Cursor for iterating over all movies

# Load all actors from file
print("\nRead all actors from file")

with open(vars.local_producers_and_actors_data_set_path, 'rb') as file:
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
        all_actors[actor_id] = actor


# Load all producer from file
print("\nRead all producers from file")
all_producers = cp.deepcopy(all_actors)

for i, (producer_id, producer) in enumerate(all_producers.items()):
    if i % 500000 == 0:
        print(f"Iteration: {i}", flush=True)
    producer["produced_movies"] = 0  # Add counter for produced movies
    del producer["played_movies"]  # Remove counter for played movies


# Load all production companies from file
print("\nRead all production companies")

with open(vars.local_producer_company_data_set_path, 'rb') as file:  # {"id":1,"name":"Lucasfilm Ltd."}
    for line in file.readlines():
        if i % 50000 == 0:
            print(f"Iteration: {i}")
        company = json.loads(line.decode("utf-8"))
        company["genres"] = np.copy(genre_counters)
        company["financed_movies"] = 0  # Add counter for financed movies

        # Save producer in dict of all production companies
        company_id = company["id"]
        all_production_companies[company_id] = company


# Count genres for actors from movies
print("\nCount real genres of all movies")
skipped_movies = []
i = 0

for movie_id, movie in all_movies.items():
    if i % 10000 == 0:
        print(f"Iteration: {i}")

    movie_genres = movie["genres"]

    if len(movie_genres) == 0:  # Skip empty array
        skipped_movies.append(movie)
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
            producer_id = int(producer_id)
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

# Write all updated actors into database
print("\nWrite all updated actors into database")

for i, (actor_id, actor) in enumerate(all_actors.items()):
    if i % 500000 == 0:
        print(f"Iteration: {i}")
    actor["id"] = actor_id
    actor["genres"] = actor["genres"].tolist()
    all_actors_table.insert_one(actor)

# Write all updated producers into database
print("\nWrite all updated producers into database")

for i, (producer_id, producer) in enumerate(all_producers.items()):
    if i % 500000 == 0:
        print(f"Iteration: {i}")
    producer["id"] = producer_id
    producer["genres"] = producer["genres"].tolist()
    all_producers_table.insert_one(producer)

# Write all updated production companies into database
print("\nWrite all updated production companies into database")

for i, (company_id, company) in enumerate(all_production_companies.items()):
    if i % 50000 == 0:
        print(f"Iteration: {i}")
    company["id"] = company_id
    company["genres"] = company["genres"].tolist()
    all_production_companies_table.insert_one(company)


# Save all skipped movies
print(f"\nSkipped {len(skipped_movies)} movies")

with open(skipped_movies_file_path, "w", encoding="utf-8") as file:
    file.writelines([str(movie) + "\n" for movie in skipped_movies])
