import numpy as np
import pandas as pd
import sys

from sklearn.linear_model import LinearRegression
from typing import Any, Dict, List, Tuple


# Define constants
MIN_MOVIE_HISTORY_LEN = 2
DISTANCE_TO_OTHER_MOVIES = 0.1
TRAIN_DATA_RELATIONSHIP = 0.8


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


def extract_features(user_movie_histories: Dict[int, List[np.array]], movie_history_len: int, min_movie_history_len: int=MIN_MOVIE_HISTORY_LEN) -> List[Tuple[np.array, np.array]]:
    """
        Extract features: partionate user histories into parts with length
        "min_movie_history_len" so that next movie is the predicted target.
        Returns tuples consisting of the last seen movies and the next one
        to predict (= target, label) out ot the previous ones.
    """

    all_extracted_features = []
    skipped_users, used_users = 0, 0

    for users_movie_history in user_movie_histories.values():  # Iterate over all users' histories
        if len(users_movie_history) < min_movie_history_len:  # User has not enough movies watched
            skipped_users += 1
            continue
        elif len(users_movie_history) <= movie_history_len:  # Use has watched enoguh movies, but not many
            # Find movies and target/label
            movies = users_movie_history[:-1]
            target_label = users_movie_history[-1]

            # Fill missing movies with zeros
            number_of_missing_movies = movie_history_len - len(movies)
            zero_movie = np.zeros(target_label.shape[0])  # Create movie containing only 0 for all real genres
            zero_movies = list(np.tile(zero_movie, (number_of_missing_movies, 1)))

            # Create one list with zero movies and watched movies of a user
            history_feature = (zero_movies + movies, target_label)
            all_extracted_features.append(history_feature)
        else:  # Use history only, if it is long enough
            all_extracted_features.extend(
                [(np.copy(users_movie_history[i:i+movie_history_len]), users_movie_history[movie_history_len])
                    for i in range(0, len(users_movie_history) - movie_history_len - 1, movie_history_len)]
            )

    used_users = len(user_movie_histories) - skipped_users
    print(f"Extracted histories of {used_users} users")
    print(f"Skipped {skipped_users} histroies, because they have less than "\
          + f"{min_movie_history_len} movies in their history of movies")

    return used_users, all_extracted_features


if __name__ == "__main__":
    # Read data from database
    # all_movies = Movies().get_all()
    # all_users = Users().get_all()
    # all_genres = Genres().get_all()

    # Find real genres to movies, users have watched
    # user_movie_histories = find_real_genres_to_all_user_movies(all_movies, all_users)
    # save_object_in_file(vars.user_history_file_path_with_real_genres, user_movie_histories)

    # Read data again and create a pandas DataFrame with the 
    # df_user_movie_histories = find_real_genres_to_all_user_movies_for_visualization(all_movies, all_users, all_genres)
    # save_object_in_file(vars.user_history_file_path_with_real_genres_visualization, df_user_movie_histories)
    # print(nan_movies)

    # Visualize data
    # df_user_movie_histories = load_object_from_file(vars.user_history_file_path_with_real_genres_visualization)
    # print(df_user_movie_histories.iloc[68])
    # print([i for i, row in df_user_movie_histories.iterrows() if row.isnull().any()])

    # Compute based on this extracted features
    # user_movie_histories = load_object_from_file(vars.user_history_file_path_with_real_genres)
    # used_users, extracted_features = extract_features(user_movie_histories, 10, 5)
    # save_object_in_file(vars.extracted_features_file_path, (used_users, extracted_features))

    # # Read extracted features
    used_users, extracted_features = load_object_from_file(vars.extracted_features_file_path)
    shapes = set([len(f) for f, l in extracted_features])
    print(shapes)

    # Split data into train, test and validation data
    # e.g. with "train_test_split"
    train_data_len = int(TRAIN_DATA_RELATIONSHIP * len(extracted_features))
    train_data = extracted_features[:train_data_len]
    test_data = extracted_features[train_data_len:]

    # Split data into X and y
    X_train, y_train = [x for x, _ in train_data], [y for _, y in train_data]
    X_test, y_test = [x for x, _ in test_data], [y for _, y in test_data]

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(model.score(X_train, y_train))
