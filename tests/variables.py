import numpy as np
import pandas as pd
import sys

from pathlib import Path

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[1]
sys.path.append(str(project_dir))

from helper.file_system_interaction import load_one_json_object_from_file


# --------------- Define variables for testing all modules ---------------
json_file_path = project_dir / "tests/jsons_files"  # Define path to test JSON files

# ------------ test_api_requester.py ------------
api_requester_json_basic_file_path = json_file_path / "test_api_requester_jsons"

# ------------ test_prepare_data.py ------------
test_prepare_data_jsons_path = json_file_path / "test_prepare_data_jsons"

prepare_data_all_movies_json_file_path = test_prepare_data_jsons_path / "all_movies.json"
prepare_data_movies_per_user_one_user = test_prepare_data_jsons_path / "movies_per_user_one_user.pickle"
prepare_data_movies_per_user_many_users_one_movie = (
    test_prepare_data_jsons_path / "movies_per_user_many_users_one_movie.pickle"
)
prepare_data_movies_per_user_many_users_equal_movies = (
    test_prepare_data_jsons_path / "movies_per_user_many_users_equal_movies.pickle"
)

# --------------- Define variables for testing movies ---------------
# Define all movies
all_movies = load_one_json_object_from_file(prepare_data_all_movies_json_file_path)  # Load movies from file
all_movies = dict([(int(movie_id), movie) for movie_id, movie in all_movies.items()])  # Replace str keys with int keys

# One user watched many movies
all_movies_real_genres_per_user_one_user = {
    0: [np.array(movie["real_genres"], dtype=np.float64) for movie in all_movies.values()]
}
max_history_len_one_user = len(list(all_movies_real_genres_per_user_one_user.values())[0])
all_movies_real_genres_per_user_one_user_unraveled = [(0, movie) for movie in all_movies.values()]
all_movies_real_genres_per_user_one_user_raveled = pd.DataFrame(
    dict([(f"m{i}", movie["real_genres"] + [0]) for i, movie in enumerate(all_movies.values())])
).T

# Many users watched exactly one movie
max_history_len_many_users_one_movie = 1
all_movies_real_genres_per_user_many_users_one_movie = dict(
    [(i, [np.array(movie["real_genres"], dtype=np.float64)]) for i, movie in enumerate(all_movies.values())]
)
all_movies_real_genres_per_user_many_users_one_movie_unraveled = [(i, movie) for i, movie in enumerate(all_movies.values())]
all_movies_real_genres_per_user_many_users_one_movie_raveled = pd.DataFrame(
    dict([(f"m{i}", movie["real_genres"] + [i]) for i, movie in enumerate(all_movies.values())])
).T

# Each user watched 10 movies
max_history_len_many_users_equal_movies = 10
all_movies_real_genres_per_user_many_users_equal_movies = dict(
    [
        (
            i // max_history_len_many_users_equal_movies,
            [
                np.array(movie["real_genres"], dtype=np.float64)
                for movie in list(all_movies.values())[i:i + max_history_len_many_users_equal_movies]
            ],
        )
        for i in range(0, len(all_movies), max_history_len_many_users_equal_movies)
    ]
)
all_movies_real_genres_per_user_many_users_equal_movies_unraveled = [
    (i // max_history_len_many_users_equal_movies, movie) for i, movie in enumerate(all_movies.values())
]
all_movies_real_genres_per_user_many_users_equal_movies_raveled = pd.DataFrame(
    dict(
        [
            (f"m{i}", movie["real_genres"] + [i // max_history_len_many_users_equal_movies])
            for i, movie in enumerate(all_movies.values())
        ]
    )
).T
