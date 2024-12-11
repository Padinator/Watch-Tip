import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from typing import Any, Dict, List, Tuple

# ---------- Import own python files ----------
sys.path.append('../')

import helper.variables as vars

from database.movie import Movies
from database.user import Users
from database.genre import Genres
from helper.file_system_interaction import load_object_from_file, save_object_in_file


# Define constants
HISTORY_LEN = 10
MIN_MOVIE_HISTORY_LEN = 5
DISTANCE_TO_OTHER_MOVIES = 0.1
TRAIN_DATA_RELATIONSHIP = 0.85
SEED = 1234

# Constants for computing the difference between multiple values
EPSILON = 50
INDEPENDENT_MAX_DIFF_PER_GENRE = 5
NUMBER_OF_INTERVALS = 5
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


def extract_features(user_movie_histories: Dict[int, List[np.array]],
        movie_history_len: int, min_movie_history_len: int=MIN_MOVIE_HISTORY_LEN,
        fill_history_len_with_zero_movies=True) -> List[Tuple[np.array, np.array]]:
    """
        Extract features: partionate user histories into parts with length
        "min_movie_history_len" so that next movie is the predicted target.
        Returns tuples consisting of the last seen movies and the next one
        to predict (= target, label) out ot the previous ones.
    """

    all_extracted_features = []
    skipped_histories, used_histories = 0, 0

    for users_movie_history in user_movie_histories.values():  # Iterate over all users' histories
        if len(users_movie_history) < min_movie_history_len\
                or ((not fill_history_len_with_zero_movies)\
                    and len(users_movie_history) <= movie_history_len):  # User has not enough movies watched
            skipped_histories += 1
            continue
        elif fill_history_len_with_zero_movies\
                and len(users_movie_history) <= movie_history_len:  # Use has watched enoguh movies, but not many
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

    used_histories = len(user_movie_histories) - skipped_histories
    print(f"Extracted histories of {used_histories} users")
    print(f"Skipped {skipped_histories} histories, because they have less than "\
          + f"{min_movie_history_len} movies in their history of movies")

    return used_histories, all_extracted_features


def calc_distance(ys_true: np.float64, ys_pred: np.float64, allowed_diff_per_value: float=INDEPENDENT_MAX_DIFF_PER_GENRE,
                  number_of_intervals: float=NUMBER_OF_INTERVALS) -> np.float64:
    """
        Computes distance between the true and the predicted y values.
        For each combination of true an dpredicted y values:\n
        If the true y value is higher, then a higher difference is
        acceptable, else the difference must be lower, e.g.:\n
        y_true = 86; y_pred = 80\n
        -> difference should be a maximum of 8.5\n
        => y_pred is okay\n
        \n
        y_true = 4; y_pred = 10\n
        -> difference should be a maximum of 0.5\n
        => y_pred is not okay\n
        \n
        Differences increase by 0.5 in the following intervals:\n
        Intervals:  [0,5), [5,10), [10,15), [15,20), ...\n
        Differences: 0.5      1      1.5       2     ...\n
        \n
        Returns the sum of all differences being too high.
    """

    overall_diff = 0

    for y_true, y_pred in zip(ys_true, ys_pred):
        diff = abs(y_true - y_pred)
        allowed_diff = (y_pred // number_of_intervals + 1) * allowed_diff_per_value

        if allowed_diff < diff:  # Only add differenes, which are too high
            overall_diff += diff

    return overall_diff


def evaluate_model(y_test: np.array, predictions: np.array) -> float:
    """
        Evaluates a model by comparing true test values with predicted y
        values. Compare each y value will be compared with its corresponding
        prediction value.\n
        Returns the accuracy.
    """

    # Define variables
    distances = []

    # Compute distances of pair of predicted and true y values
    for y, y_pred in zip(y_test, predictions):
        # distance = np.linalg.norm(y - y_pred)  # Euclidean distances between points
        distance = calc_distance(y, y_pred)  # Own distane per genre/value
        distances.append(distance)

    # Output some metrics
    overall_mean_deviation = sum(distances) / len(distances)
    correct_classifications_distances = [dist for dist in distances if dist <= EPSILON]
    false_classifications_distances = [dist for dist in distances if EPSILON < dist]
    mean_deviation_from_correct_classifications = sum(correct_classifications_distances) / len(correct_classifications_distances)
    mean_deviation_from_false_classifications = sum(false_classifications_distances) / len(false_classifications_distances)
    print(f"\nCorrect classifications: {len(correct_classifications_distances)},"\
        + f"false classifications: {len(false_classifications_distances)}, "\
        + f"accuracy: {len(correct_classifications_distances) / len(distances)}")
    print(f"Correct classifications deviations: {mean_deviation_from_correct_classifications}")
    print(f"False classifications deviations: {mean_deviation_from_false_classifications}")
    print(f"Overall mean deviation: {overall_mean_deviation}")

    return len(correct_classifications_distances) / len(distances)



if __name__ == "__main__":
    # Set seed
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

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
    # TODO: Look for mean genres
    # TODO: Compare genres
    # TODO: Cluster genres
    # TODO: View history of movies, the user have watched => progress change of genres
    # TODO: Eigene Loss-Funktion definieren
    df_user_movie_histories = load_object_from_file(vars.user_history_file_path_with_real_genres_visualization)
    pca = PCA(n_components=19)
    res = pca.fit_transform(df_user_movie_histories.values[:,:-1])
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    df = pd.DataFrame(data=res, columns=[f"c{i}" for i in range(len(res[0]))])
    
    for col_i in df.columns:
        for col_j in df.columns:
            if col_i == col_j:  # Do a box plot
                plt.boxplot(df[col_i])
            else:
                plt.scatter(df[col_i].values, df[col_j].values)
                plt.title(f"Components: {col_i} and {col_j}")
                plt.ylabel(f"component {col_j}")
            plt.xlabel(f"component {col_i}")
            plt.show()

    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(projection='3d')
    # colors = ["orange", "yellow", "red", "blue", "green", "purple", "black", "gold", "cyan", "bisque", "forestgreen", "deepskyblue", "slategrey", "violet", "salmon", "tan", "crimson", "lime", "darkgreen"]
    # for i, column in enumerate(df.columns):
    #     ax.scatter(list(range(df.shape[0])), df[column].values, [i * df.shape[0] // df.shape[1] for _ in range(df.shape[0])], color=colors[i])
    plt.show()
    exit()

    # Compute based on this extracted features
    user_movie_histories = load_object_from_file(vars.user_history_file_path_with_real_genres)
    used_histories, extracted_features = extract_features(user_movie_histories, HISTORY_LEN, MIN_MOVIE_HISTORY_LEN, fill_history_len_with_zero_movies=False)
    save_object_in_file(vars.extracted_features_file_path, (used_histories, extracted_features))

    # Read extracted features
    used_histories, extracted_features = load_object_from_file(vars.extracted_features_file_path)
    shapes = set([len(f) for f, l in extracted_features])
    print(used_histories, shapes)

    # Split data into train, test and validation data
    # e.g. with "train_test_split"
    train_data_len = int(TRAIN_DATA_RELATIONSHIP * len(extracted_features))
    train_data = extracted_features[:train_data_len]
    test_data = extracted_features[train_data_len:]

    # Split data into X and y
    # X_train, y_train = np.array([np.array(x).ravel() for x, _ in train_data], dtype=np.float64), np.array([y for _, y in train_data], dtype=np.float64)  # RFR
    # X_test, y_test = np.array([np.array(x).ravel() for x, _ in test_data], dtype=np.float64), np.array([y for _, y in test_data], dtype=np.float64)  # RFR
    X_train, y_train = np.array([np.array(x) for x, _ in train_data], dtype=np.float64), np.array([y for _, y in train_data], dtype=np.float64)  # LSTM
    X_test, y_test = np.array([np.array(x) for x, _ in test_data], dtype=np.float64), np.array([y for _, y in test_data], dtype=np.float64)  # LSTM
    print(f"Train shapes: X: {X_train.shape}, y: {y_train.shape}")
    print(f"Test shapes: X: {X_test.shape}, y: {y_test.shape}")

    # Train model
    predictions = []

    # # model = LinearRegression(positive=True)  # Only positive coefficients = no negative results/genres
    # """
    #     Random Forest Regressor (RFR)
    #     criterion: friedman_mse < absolute_error == squared_error < poisson
    #     max_features: log2 < sqrt < 19 < 1.0
    # """
    # model = RandomForestRegressor(random_state=SEED, n_estimators=2 * HISTORY_LEN,
    #                               max_features=19, criterion="poisson", max_depth=30,
    #                               min_samples_split=2, min_samples_leaf=1)
    # model.fit(X_train, y_train)

    # # Summarize model
    # print("Max depths of decision trees in forest:", [estimator.get_depth() for estimator in model.estimators_])
    # print(model.estimators_)  # Trees in the forest
    # print(model.n_features_in_)  # dimension of input features x = 190
    # print(model.feature_importances_)  # Contribution value of each value in features (dimension 190)
    # print(model.n_outputs_)  # dimension of outputs y = 19
    # print(model.oob_score)
    # print(model.get_params())

    # # Test model
    # predictions = model.predict(X_test)

    # Define hyper parameters of the LSTM
    epochs = 30
    steps_per_epoch = 40
    batch_size = 3

    # Build the LSTM
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(10, seed=SEED),  # , go_backwards=True),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(19, activation="relu")
    ])

    # Compile/Set solver, loss and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        # optimizer='rmsprop',
        loss=tf.keras.losses.BinaryCrossentropy(),
        
        metrics=[
            'acc'
            # tf.keras.metrics.FalseNegatives(),
            ],
    )

    # Train und test model
    history = model.fit(X_train / 100, y_train / 100, validation_split=0.05, epochs=epochs, steps_per_epoch=steps_per_epoch, batch_size=batch_size)
    model.evaluate(X_test / 100, y_test / 100, batch_size=batch_size)

    # Plot accuracy
    plt.plot(history.history['acc'], label='accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'], label='loss')
    plt.legend()
    plt.show()

    # Test model with own evaluation function
    predictions = model.predict(X_test / 100, batch_size=batch_size) * 100
    evaluate_model(y_test, predictions)

    # Output some predictions and true values/target labels
    print("\nOutput some example predictions and true values:")

    for i in range(2):
        print(predictions[i])
        print(y_test[i], "\n")
