import numpy as np
import pandas as pd
import sys

from typing import Any, Dict, List, Tuple

# ---------- Import own python files ----------
sys.path.append('../')

import helper.variables as vars

from database.movie import Movies
from database.user import Users
from database.genre import Genres
from helper.file_system_interaction import load_object_from_file, save_object_in_file


# nan_movies = []


def find_real_genres_to_all_user_movies(movies: Dict[int, Dict[str, Any]], users: Dict[int, Dict[str, Any]]) -> Dict[int, List[np.array]]:
    """
        Find real genres of users (= watched movies) with real genres of movies.
        Returns a dict of all movies as real genres = numpy arrays.
    """

    user_movie_histories = {}
    i = 0

    for user_id, reviews in users.items():
        reviews = dict(sorted(reviews.items()))  # Sort user reviews by creation date
        user_movie_histories[user_id] = []

        for _, review in reviews.items():
            if i % 1000 == 0:
                print(f"Iteration: {i}")
            movie_id = int(review["movie_id"])
            real_movie_genres = movies[movie_id]["real_genres"]
            user_movie_histories[user_id].append(np.array(real_movie_genres, dtype=np.float64))
            i += 1

    return user_movie_histories


def find_real_genres_to_all_user_movies_for_visualization(movies: Dict[int,
        Dict[str, Any]], users: Dict[int, Dict[str, Any]],
        genres: Dict[int, Dict[str, str]]) -> pd.DataFrame:
    """
        Find real genres of users (= watched movies) with real genres of movies.
        Returns a pandas DataFrame containing all movies with real genres
        watched by users. It's useful for visualizations and analyzations of
        read data.
    """

    # global nan_movies

    genre_names = [genre["name"] for genre in genres.values()]
    user_movie_histories = dict(zip(genre_names + ["username"], [[] for _ in range(len(genre_names) + 1)]))
    i = 0

    for user_id, reviews in users.items():
        reviews = dict(sorted(reviews.items()))  # Sort user reviews by creation date

        for _, review in reviews.items():
            if i % 1000 == 0:
                print(f"Iteration: {i}")
            movie_id = int(review["movie_id"])
            real_movie_genres = movies[movie_id]["real_genres"]

            for j, genre in enumerate(genre_names):
                user_movie_histories[genre].append(real_movie_genres[j])
            user_movie_histories["username"].append(user_id)
            i += 1

            # if np.isnan(np.min(real_movie_genres)):
            #     nan_movies.append(movie_id)

    return pd.DataFrame(user_movie_histories)


if __name__ == "__main__":
    # Read data from database
    print("Read all movies, all users reviews and all genres from database.")
    all_movies = Movies().get_all()
    all_users = Users().get_all()
    all_genres = Genres().get_all()
    genre_names = np.array([genre["name"] for genre in all_genres.values()])

    # Find real genres to movies, users have watched
    print("\nFind real genres to movies, users have watched and save them to file.")
    user_movie_histories = find_real_genres_to_all_user_movies(all_movies, all_users)
    save_object_in_file(vars.user_history_file_path_with_real_genres, user_movie_histories)

    # Read data again, ceate a pandas DataFrame with the read data and save it to file
    print("\nRead data again, ceate a pandas DataFrame with the read data and save it to file.")
    df_user_movie_histories = find_real_genres_to_all_user_movies_for_visualization(all_movies, all_users, all_genres)
    save_object_in_file(vars.user_history_file_path_with_real_genres_visualization, df_user_movie_histories)
    # print(nan_movies)
