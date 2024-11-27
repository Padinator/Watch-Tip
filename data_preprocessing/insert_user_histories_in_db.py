import collections
import pymongo


netflix_movie_filename = "data/netflix_user_data/movie_titles.csv"
netflix_movies = {}

# Read movies listed by Netflix
with open(netflix_movie_filename, "r") as netflix_movies_file:
    for line in netflix_movies_file.readlines():
        id, year, name = line.split(",", 2)
        netflix_movies[id] = {"name": name.strip(), "year": year}

movies = [movie["name"] for movie in netflix_movies.values()]
print(len(netflix_movies.values()), len(movies), len(set(movies)))
print([item for item, count in collections.Counter(movies).items() if count > 1])
