import copy as cp
import json
import numpy as np

from typing import Any, Dict

import helper.parallelizer as para
import helper.variables as vars

from database.actor import Actors
from database.genre import Genres
from database.model import DatabaseModel
from database.movie import Movies
from database.producer import Producers
from database.production_company import ProductionCompanies


# Define variables
all_actors = {}
all_producers = {}
all_production_companies = {}
all_genres_indices = {}

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

with open(vars.local_actors_file_path, 'rb') as file:
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

with open(vars.local_producers_file_path, 'rb') as file:
    for i, line in enumerate(file.readlines()):
        if i % 500000 == 0:
            print(f"Iteration: {i}")

        producer = json.loads(line.decode("utf-8"))
        producer["id"] = int(producer["id"])
        producer["popularity"] = float(producer["popularity"])
        producer["genres"] = np.copy(genre_counters)

        # Save producer in dict of all producers
        producer_id = producer["id"]
        all_producers[producer_id] = producer


# Load all production companies from file
print("\nRead all production companies")

with open(vars.local_production_companies_file_path, 'rb') as file:  # {"id":1,"name":"Lucasfilm Ltd."}
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

for i, (movie_id, movie) in enumerate(all_movies.items()):
    movie = all_movies[movie_id]
    movie_genres = movie["genres"]

    if i % 10000 == 0:
        print(f"Iteration: {i}")

    if movie["release_date"] == "":  # Skip movie without a release data
        skipped_movies.append(movie)
        continue

    # Count genres of all actors of a movie
    for actor_id in movie["credits"]["cast"]:
        try:
            all_actors[actor_id]["genres"][movie_genres] += 1  # Increment counters of each genre
            all_actors[actor_id]["played_movies"] += 1
        except Exception as e:
            if actor_id not in all_actors:  # Actor not in database
                pass
            else:
                print(e)
                print(actor_id)
                print(all_actors[actor_id])
                print(all_actors[actor_id]["genres"])
                # raise e

    # Count genres of all producers of a movie
    for producer_id in movie["credits"]["crew"]:
        try:
            producer_id = int(producer_id)
            all_producers[producer_id]["genres"][movie_genres] += 1  # Increment counters of each genre
            all_producers[producer_id]["produced_movies"] += 1
        except Exception as e:
            if producer_id not in all_producers:  # Producer not in database
                pass
            else:
                print(e)
                print(producer_id)
                print(all_producers[producer_id])
                print(all_producers[producer_id]["genres"])
                # raise e

    # Count genres of all production companies of a movie
    for company_id in movie["production_companies"]:
        try:
            all_production_companies[company_id]["genres"][movie_genres] += 1  # Increment counters of each genre
            all_production_companies[company_id]["financed_movies"] += 1
        except Exception as e:
            if company_id not in all_production_companies:  # Production company not in database
                pass
            else:
                print(e)
                print(company_id)
                print(all_production_companies[company_id])
                print(all_production_companies[company_id]["genres"])
                # raise e


# Save all skipped movies
print(f"\nSkipped {len(skipped_movies)} movies")

with open(vars.skipped_movies_file_path, "w", encoding="utf-8") as file:
    file.writelines([str(movie) + "\n" for movie in skipped_movies])



def insert_one(table: 'DatabaseModel', id: int, entity: Dict[str, Any]) -> None:
    entity["id"] = id
    entity["genres"] = entity["genres"].tolist()
    table.insert_one(entity)


# Write all updated actors into database
print("\nWrite all updated actors into database")
para.parallelize_task_without_return_values(insert_one, [(all_actors_table, actor_id, actor) for actor_id, actor in all_actors.items()], 16, 500000)

# for i, (actor_id, actor) in enumerate(all_actors.items()):
#     if i % 500000 == 0:
#         print(f"Iteration: {i}")
#     actor["id"] = actor_id
#     actor["genres"] = actor["genres"].tolist()
#     all_actors_table.insert_one(actor)

# # Write all updated producers into database
print("\nWrite all updated producers into database")
para.parallelize_task_without_return_values(insert_one, [(all_producers_table, producer_id, producer) for producer_id, producer in all_producers.items()], 16, 500000)

# for i, (producer_id, producer) in enumerate(all_producers.items()):
#     if i % 500000 == 0:
#         print(f"Iteration: {i}")
#     producer["id"] = producer_id
#     producer["genres"] = producer["genres"].tolist()
#     all_producers_table.insert_one(producer)

# # Write all updated production companies into database
print("\nWrite all updated production companies into database")
para.parallelize_task_without_return_values(insert_one, [(all_production_companies_table, company_id, company) for company_id, company in all_production_companies.items()], 16, 50000)

# for i, (company_id, company) in enumerate(all_production_companies.items()):
#     if i % 50000 == 0:
#         print(f"Iteration: {i}")
#     company["id"] = company_id
#     company["genres"] = company["genres"].tolist()
#     all_production_companies_table.insert_one(company)
