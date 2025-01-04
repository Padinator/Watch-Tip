import numpy as np
import pandas as pd
import sys

from collections import defaultdict
from MulticoreTSNE import MulticoreTSNE as TSNE
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[1]
sys.path.append(str(project_dir))

import helper.variables as vars

from database.movie import Movies
from database.user import Users
from database.genre import Genres
from helper.file_system_interaction import load_object_from_file, save_object_in_file

# from helper.parallelizer import parallelize_task_with_return_values
from helper.parallelizer import ThreadPool


# Define constants
CPU_KERNELS = 16
NUMBER_DIMENSIONS = 3  # Number of target dimensions to reduce dataset to (with t-SNE)
NUMBER_ITERATIONS = 100000
NUMBER_OF_VISUALIZED_NETFLIX_MOVIE_WATCHINGS = 30000


def find_real_genres_to_a_movie(
    user_id: int, movie: Dict[str, Any], all_movies: Dict[int, Dict[str, Any]]
) -> Tuple[int, int, np.ndarray]:
    """
    Searches to a movie the real genres of it. Uses for this the passed dict
    of all movies.\n
    Passing the user ID has no usage, it will be returned immediately as first
    parameter of return tuple. It's only for easy matching real genres of a
    movie, a user has watched, to the user's movie history in mode parallel
    execution.

    Parameters
    ----------
    movie : Dict[str, Any]
        A movie to find real gernes of.
    all_movies : Dict[int, Dict[str, Any]]
        Dict with all movies, movie ID as key and movie properies as values in a nother dict

    Returns
    -------
    Tuple[int, int, np.ndarray]
        First enrty contains ID of user, which have watched a movie, second
        entry contains real genres of a movie and the third one the ID of the
        movie.\n
        Returns for second entry None, if a movie has no genres (all genres
        are very close to zero).
    """

    # Find watched movie and it's real genres
    movie_id = int(movie["movie_id"])
    real_movie_genres = all_movies[movie_id]["real_genres"]

    # Skip movies without any real genre
    if all(-1e-3 < genre_value * 100 < 1e-3 for genre_value in real_movie_genres):
        return user_id, movie_id, None

    return user_id, movie_id, real_movie_genres


def find_real_genres_to_all_user_movies(
    all_movies: Dict[int, Dict[str, Any]],
    all_user_reviews_unraveled: List[Tuple[int, Dict[str, Any]]],
    genre_names: List[str],
    cpu_kernels: int = 8,
    output_iterations: int = 1000,
) -> Tuple[Dict[int, List[Tuple[int, np.ndarray]]], pd.DataFrame]:
    """
    Find real genres of users (= watched movies) with real genres of movies.
    Returns a dict of all movies as real genres = numpy arrays.

    Parameters
    ----------
    all_movies : Dict[int, Dict[str, Any]]
        Dict with all movies, movie ID as key and movie properies as values in
        another dict
    all_user_reviews_unraveled : Dict[int, List[Dict[str, Any]]]
        Dict with all movies a user have watched with username as key and
        lists of movies as values
    genre_names : List[str]
        Names of all genres
    cpu_kernels : int, default 8
        Define number of used CPU kernels for executing this function in
        parallel mode
    output_iterations : int, default 1000
        Define number of iterations after which a line will be outputted.
        Pass None or 0, if no print is requested.

    Returns
    -------
    Tuple[
                Dict[int, List[Tuple[int, np.ndarray]]],\n
                pd.DataFrame
    }
        First entry contains dict with all usernames as IDs (key) and lists of
        tuples with movie IDs and real genres of these movies movies users
        have watched (values).
        Second entry contains unraveled form of first entry for visualization
        (row: movie; column: genre anmes + username).
    """

    # Define variables
    user_movies_raveled = defaultdict(list)
    user_movie_histories = defaultdict(list)

    # Prepare arguments for execution and find real genres
    print("Prepare arguments for execution")
    args = [(user_id, review, all_movies) for user_id, review in all_user_reviews_unraveled]
    print("Find real genres")
    thread_pool = ThreadPool(max_number_of_runnings_threads=cpu_kernels)
    res = thread_pool.join(task=find_real_genres_to_a_movie, args=args, print_iteration=output_iterations)
    # res = parallelize_task_with_return_values(find_real_genres_to_a_movie, args, cpu_kernels, output_iterations)

    # Create raveled dict of all watchings per user
    print("Create raveled dict of all watchings per user")
    for watching in res:
        user_id, movie_id, real_movie_genres = watching

        if real_movie_genres is not None:  # Ignore movies withput any genre
            user_movies_raveled[user_id].append((movie_id, real_movie_genres))

    # Create unravled dict with all watchings as rows and genre names as columns
    print("Create unraveled dict with all watchings as rows and genre names as columns")
    for watching in res:
        user_id, movie_id, real_movie_genres = watching

        if real_movie_genres is not None:  # Ignore movies withput any genre
            for j, genre in enumerate(genre_names):
                user_movie_histories[genre].append(real_movie_genres[j])
            user_movie_histories["username"].append(user_id)
            user_movie_histories["movie"].append(movie_id)

    return (user_movies_raveled, pd.DataFrame(user_movie_histories))


def reduce_dimensions_on_user_histories_visualization(
    df_user_movie_histories: pd.DataFrame, n_dimensions: int = 3, cpu_kernels: int = 8
) -> pd.DataFrame:
    """
    Reduces dimension of passed DataFrame with the helpt of t-SNE and returns
    a DataFrame with only n_dimensions as passed.

    Parameters
    ----------
    df_user_movie_histories : pd.DataFrame
        Contains all movies users have watched (rows: movies; columns: genres and username)
    n_dimensions : int, default 3
        Target dimensions, DataFrame/genres will be reduced to n_components dimensions
    cpu_kernels : int, default 8
        Number of CPU kernels to calculate t-SNE with

    Returns
    -------
    pd.DataFrame
        Contains data with reduced dimensions, n_dimensions columns and same number of rows (movies)
    """

    # Ignore column username for dimension reduction
    df_user_movie_histories_without_username = df_user_movie_histories.loc[:, df_user_movie_histories.columns != "username"]

    # Reduce dimensions
    user_movie_histories_reduced_dim = TSNE(n_components=n_dimensions, n_jobs=cpu_kernels).fit_transform(
        df_user_movie_histories_without_username
    )

    # Save data with reduced dimensions in a DataFrame
    dimensions = {}  # Store data of each dimension with specific key

    for i in range(n_dimensions):
        dimensions[f"dim{i}"] = user_movie_histories_reduced_dim[:, i]

    # Add to dimension data username and create one DataFrame based on them
    df_user_movie_histories_reduced_dim = pd.DataFrame(
        {
            **dimensions,
            "username": df_user_movie_histories["username"].values,
            "movie": df_user_movie_histories["movie"].values
        }
    )

    return df_user_movie_histories_reduced_dim


def groupd_movie_histories_by_user(
    user_movie_histories: Dict[int, List[np.ndarray]], df_user_movie_histories_reduced_dim: pd.DataFrame
) -> Dict[int, np.ndarray]:
    """
    Extract from (visualized) DataFrame (with reduced dimension data)
    movie histories per user. Then, this data can be used as train
    data for an AI model.\n
    This will be done by grouping the data/real genres of movies by the key
    username.

    Parameters
    ----------
    user_movie_histories : Dict[int, List[np.ndarray]]
        User movie histories with real genres (computed with e.g. "find_real_genres_to_all_user_movies").
        It's necessary for sorting the result same as user_movie_histories.
    df_user_movie_histories_reduced_dim : pd.DataFrame
        Contains user movie histories with reduced dimensions as visualized DataFrame

    Returns
    -------
    Dict[int, np.ndarray]
        Returns a dict containing for each movie the real genres with
        reduced dimensions. The first dimension is the username and the second
        one consists of the real movie genres.
    """

    usernames = list(user_movie_histories.keys())
    columns_except_username = [col for col in df_user_movie_histories_reduced_dim if col != "username"]
    user_movie_histories_reduced_dim = {}  # Store all dimension reduced real movie genres per user

    # Group movies by users and use sorting of object "user_movie_histories"
    for username in usernames:
        rows = df_user_movie_histories_reduced_dim.loc[
            df_user_movie_histories_reduced_dim["username"] == username,
            columns_except_username,
        ]
        user_movie_histories_reduced_dim[username] = rows.values

    return user_movie_histories_reduced_dim


if __name__ == "__main__":
    # Read data from database
    print("Read all movies, all users reviews and all genres from database.")
    # all_movies = Movies().get_all()
    # all_users = Users().get_all()
    # all_genres = Genres().get_all()

    all_movies = load_object_from_file("tmp_all_movies.pickle")
    all_users = load_object_from_file("tmp_all_users.pickle")
    all_genres = load_object_from_file("tmp_all_genres.pickle")

    genre_names = np.array([genre["name"] for genre in all_genres.values()])

    # Find real genres to movies, users have watched and save them to file
    print("\nFind real genres to movies, users have watched and save them to file.")
    all_user_reviews_unraveled = [
        (user_id, review) for user_id, reviews in all_users.items() for review in reviews.values()
    ]
    user_movie_histories, df_user_movie_histories = find_real_genres_to_all_user_movies(
        all_movies, all_user_reviews_unraveled, genre_names=genre_names, cpu_kernels=CPU_KERNELS, output_iterations=1000
    )

    print("\nStatistics about found user reviews:")
    print(type(user_movie_histories))
    print(len(user_movie_histories))
    print(sum([len(watchings) for _, watchings in user_movie_histories.items()]))

    print("\nStatistics about found user reviews of DataFrame:")
    print(type(df_user_movie_histories))
    print(df_user_movie_histories.shape)

    # Save reviews per user and of all users (DataFrame)
    save_object_in_file(vars.user_history_file_path_with_real_genres, user_movie_histories)  # Reviews per user
    save_object_in_file(
        vars.user_history_file_path_with_real_genres_visualization,
        df_user_movie_histories,
    )  # Reviews of all users

    # Read data again, reduce dimensions to 3 and save it to file (visualization)
    print("\nRead data again, reduce dimensions to 3 and save it to file (visualization)")
    df_user_movie_histories_reduced_dim = reduce_dimensions_on_user_histories_visualization(
        df_user_movie_histories, n_dimensions=NUMBER_DIMENSIONS, cpu_kernels=CPU_KERNELS
    )
    save_object_in_file(
        vars.user_history_file_path_with_real_genres_and_reduced_dimensions_visualization,
        df_user_movie_histories_reduced_dim,
    )

    # Transform dimension reduced data back to user specific arrays (no DataFrame)
    print("\nTransform dimension reduced data back to user specific arrays (no DataFrame)")

    # Get all usernames in the order like above and save them
    user_movie_histories_reduced_dim = groupd_movie_histories_by_user(
        user_movie_histories, df_user_movie_histories_reduced_dim
    )
    save_object_in_file(
        vars.user_history_file_path_with_real_genres_and_reduced_dimensions,
        user_movie_histories_reduced_dim,
    )

    # Prepare Netflix data (full data)
    print("\nStart reading")
    user_watchings = load_object_from_file(vars.netflix_movies_watchings_path)
    print("Unravel watchings")
    all_user_watchings_unraveled = [
        (user_id, watching) for user_id, watchings in list(user_watchings.items()) for watching in watchings
    ]
    print("Start function")
    user_watchings_with_real_genres, df_user_watchings_with_real_genres = find_real_genres_to_all_user_movies(
        all_movies, all_user_watchings_unraveled, genre_names=genre_names, cpu_kernels=CPU_KERNELS, output_iterations=100000
    )

    print("\nStatistics about user watchings results:")
    print(type(user_watchings_with_real_genres))
    print(len(user_watchings_with_real_genres))
    print(sum([len(watchings) for _, watchings in user_watchings_with_real_genres.items()]))
    print(f"\nShape user watchings DataFrame: {df_user_watchings_with_real_genres.shape}\n")

    print("\nSave user watchings per user")
    save_object_in_file(vars.user_watchings_file_path_with_real_genres, user_watchings_with_real_genres)
    print("Save user watchings of all user (DataFrame)")
    save_object_in_file(vars.user_watchings_file_path_with_real_genres_visualization, df_user_watchings_with_real_genres)

    # Prepare Netflix data (small part of data, good for some visualizations)
    df_user_watchings_with_real_genres_small = df_user_watchings_with_real_genres.iloc[
        :NUMBER_OF_VISUALIZED_NETFLIX_MOVIE_WATCHINGS, :
    ]

    netflix_prize_small_dataset_usernames = set(df_user_watchings_with_real_genres_small["username"])
    netflix_prize_small_dataset_username_dict = dict([(username, []) for username in netflix_prize_small_dataset_usernames])
    user_watchings_with_real_genres_small = groupd_movie_histories_by_user(  # Nochmal machen weil gruppieren nicht geht!!!
        netflix_prize_small_dataset_username_dict, df_user_watchings_with_real_genres_small
    )

    print("\nSave user watchings per user of some little part of all users")
    save_object_in_file(vars.user_watchings_file_path_with_real_genres_small, user_watchings_with_real_genres_small)

    print("Save user watchings of some little part of all users (DataFrame)")
    save_object_in_file(
        vars.user_watchings_file_path_with_real_genres_visualization_small, df_user_watchings_with_real_genres_small
    )
