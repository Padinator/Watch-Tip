import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import sys
import tensorflow as tf

from collections import Counter, defaultdict
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Conv1D, SimpleRNN, AveragePooling1D, MaxPool1D, Flatten
from tensorflow.keras import Sequential, backend
from typing import Any, Dict, List, Tuple

# ---------- Import own python modules ----------
project_dir = Path(os.path.abspath(__file__)).parents[1]
sys.path.append(str(project_dir))

import helper.variables as vars

from helper.file_system_interaction import load_object_from_file, save_object_in_file


# Define constants
MAX_DATA = 50000
HISTORY_LEN = 50
MIN_MOVIE_HISTORY_LEN = 20
FILL_HISTORY_LEN_WITH_ZERO_MOVIES = False
FINE_GRAINED_EXTRACTING = False
DISTANCE_TO_OTHER_MOVIES = 0.1
TRAIN_DATA_RELATIONSHIP = 0.85
SEED = 1234

# Constants for computing the difference between multiple values
EPSILON = 1  # This mean thah prediction can lay one genre next to the correct one
INDEPENDENT_MAX_DIFF_PER_GENRE = 5
NUMBER_OF_INTERVALS = 5

# Set print options for numpy
np.set_printoptions(formatter={"all": lambda x: "{0:0.3f}".format(x)})

# Set seeds
np.random.seed(seed=SEED)
random.seed(SEED)
tf.random.set_seed(seed=SEED)

# Define variables
save_dir = Path(
    f"results/{MAX_DATA}_{TRAIN_DATA_RELATIONSHIP}_{HISTORY_LEN}_{MIN_MOVIE_HISTORY_LEN}_{FILL_HISTORY_LEN_WITH_ZERO_MOVIES}_{FINE_GRAINED_EXTRACTING}"
)
extracted_features_file_path = (
    vars.extracted_features_folder
    / f"extracted_features_{FILL_HISTORY_LEN_WITH_ZERO_MOVIES}_{FINE_GRAINED_EXTRACTING}.pickle"
)


def one_hot_encoding_1d_arr(arr: np.array, factor: int = 50) -> np.array:
    """
        Does a one-hot encoding: make features binary = containing only 0 and 1\
        Each value greater than factor will be mapped to a 1, else 0.
    """

    return np.array([1 if factor < x else 0 for x in arr], dtype=np.float64)


def one_hot_encoding_2d_arr(arr: np.array) -> np.array:
    """
        Does a one-hot encoding like "one_hot_encoding_1d_arr":\
        Make all features/values in 2D array binary = containing only 0 and 1\
        Each value/field greater than factor will be mapped to a 1, else 0.
    """

    return np.array([one_hot_encoding_1d_arr(x) for x in arr], dtype=np.float64)


def extract_features(
    user_movie_histories: Dict[int, List[Tuple[int, np.array]]],
    movie_history_len: int,
    min_movie_history_len: int = MIN_MOVIE_HISTORY_LEN,
    fill_history_len_with_zero_movies: bool = True,
    fine_grained_extracting: bool = False,
) -> List[Tuple[np.array, np.array]]:
    """
    Extract features: partionate user histories into parts with length
    "movie_history_len" so that next movie is the predicted target.\n
    Returns tuples consisting of the last seen movies (= input) and the next
    one to predict (= target, label) out ot the previous ones.

    Parameters
    ----------
    user_movie_histories : Dict[int, List[Tuple[int, np.array]]]
        Grouped/Unraveled movies histories per user, e.g.:
        {
            1038924: [
                (102, [10.2, 39.3, 59.5, 56.4, 12.8, 0.76, 96.4, 21.3, 69.0, 98.5,
                28.2, 49.1, 19.01, 39.4, 18.9, 38.2, 98.5, 25.6, 9.3]),  # Real genres of movie 102 of user 1038924\n
                ...
            ]
        }
    movie_history_len : int
        Resulting length of each movie history, which represents an input
        feature for an AI model.
    min_movie_history_len : int, default MIN_MOVIE_HISTORY_LEN
        Minimal history length of a movie history. Smaller history lengths
        will be ignored.
    fill_history_len_with_zero_movies : bool, default True
        If fill_history_len_with_zero_movies is True and if
        min_movie_history_len < movie_history_len, then missing movies will be
        added with zero movies (zero vectors). These zero movies will be added
        as the first elements of a movie history (so the ones, which are nearer
        to the past).
    fine_grained : bool, default False
        If False, then all movies will be split into packets of
        "movie_history_len", else it will happen like the following example:\n
        movies: [a, b, c, d]; movie_history_len = 3\n
        packates: [a, b, c], [b, c, d]

    Returns
    -------
    List[Tuple[np.array, np.array]]
        Returns tuples consisting of the last seen movies (= input) and the
        next one to predict (= target, label) out ot the previous ones.
    """

    all_extracted_features = []
    skipped_histories, used_histories = 0, 0
    steps_size = 1 if fine_grained_extracting else movie_history_len

    for users_movie_history in user_movie_histories.values():  # Iterate over all users' histories
        if len(users_movie_history) < min_movie_history_len or (
            (not fill_history_len_with_zero_movies) and len(users_movie_history) <= movie_history_len
        ):  # User has not enough movies watched
            skipped_histories += 1
            continue
        elif (
            fill_history_len_with_zero_movies and len(users_movie_history) <= movie_history_len
        ):  # Use has watched enoguh movies, but not many
            # Find movies and target/label
            movies = users_movie_history[:-1]
            movie_id, target_label = users_movie_history[-1]  # Save in label movie ID and target real genres

            # Fill missing movies with zeros
            number_of_missing_movies = movie_history_len - len(movies)
            zero_movie = (-1, np.zeros(len(target_label)))  # Create movie containing only 0 for all real genres with ID -1
            zero_movies = list(np.tile(zero_movie, (number_of_missing_movies, 1)))

            # Create one list with zero movies and watched movies of a user
            history_feature = (zero_movies + movies, (movie_id, target_label))
            all_extracted_features.append(history_feature)
        else:  # Use history only, if it is long enough
            all_extracted_features.extend(
                [
                    (
                        np.copy(users_movie_history[i:i + movie_history_len]),
                        users_movie_history[i + movie_history_len],  # Save in label movie ID and target real genres
                    )
                    for i in range(0, len(users_movie_history) - movie_history_len, steps_size)
                ]
            )

    used_histories = len(user_movie_histories) - skipped_histories
    print(f"Extracted histories of {used_histories} users")
    print(
        f"Skipped {skipped_histories} histories, because they have less than "
        + f"{min_movie_history_len} movies in their history of movies"
    )

    return used_histories, all_extracted_features


def calc_distance_relative(
    ys_true: np.float64,
    ys_pred: np.float64,
    allowed_diff_per_value: float = INDEPENDENT_MAX_DIFF_PER_GENRE,
    number_of_intervals: float = NUMBER_OF_INTERVALS,
) -> np.float64:
    """
    Computes distance between the true and the predicted y values.
    For each combination of true an dpredicted y values:\n
    If the true y value is higher, then a higher difference is
    acceptable, else the difference must be lower, e.g.:\n
    y_true = 86; y_pred = 80\n
    -> difference should be a maximum of 8\n
    => y_pred is okay\n
    \n
    y_true = 4; y_pred = 10\n
    -> difference should be a maximum of 1\n
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
        allowed_diff = (y_true // number_of_intervals + 1) * allowed_diff_per_value
        # allowed_diff = allowed_diff_per_value

        if allowed_diff < diff:  # Only add differenes, which are too high
            overall_diff += diff

    return overall_diff


def calc_distance_euclidean(ys_true: np.float64, ys_pred: np.float64) -> np.float64:
    """
    Calculates and returns euclidean distance of two numpy arrays.
    """

    return np.linalg.norm(ys_true - ys_pred)


def evaluate_model(
    y_test: np.array,
    predictions: np.array,
    save_dir: Path,
    distance_method="euclidean",
    epsilon: float = EPSILON,
) -> float:
    """
    Evaluates a model by comparing true test values with predicted y
    values. Compare each y value will be compared with its corresponding
    prediction value.\n
    Returns the accuracy.
    """

    # Define variables
    distances = []

    # Find method for computing distance between a predicted and a true vector
    if distance_method == "euclidean":
        calc_distance = calc_distance_euclidean
    else:
        calc_distance = calc_distance_relative

    # Compute distances of pair of predicted and true y values
    for y, y_pred in zip(y_test, predictions):
        # distance = np.linalg.norm(y - y_pred)  # Euclidean distances between points
        distance = calc_distance(y, y_pred)  # Own distane per genre/value
        distances.append(distance)

    # Output some metrics
    overall_mean_deviation = sum(distances) / len(distances)
    correct_classifications_distances = [dist for dist in distances if dist <= epsilon]
    false_classifications_distances = [dist for dist in distances if epsilon < dist]

    if correct_classifications_distances != []:
        mean_deviation_from_correct_classifications = sum(correct_classifications_distances) / len(
            correct_classifications_distances
        )
    else:
        mean_deviation_from_correct_classifications = -1

    if false_classifications_distances != []:
        mean_deviation_from_false_classifications = sum(false_classifications_distances) / len(
            false_classifications_distances
        )
    else:
        mean_deviation_from_false_classifications = -1

    with open(save_dir / "own_evaluation.txt", "w") as file:
        print(
            f"\nCorrect classifications: {len(correct_classifications_distances)},"
            + f"false classifications: {len(false_classifications_distances)}, "
            + f"accuracy: {len(correct_classifications_distances) / len(distances)}",
            file=file,
        )
        print(f"Correct classifications deviations: {mean_deviation_from_correct_classifications}", file=file)
        print(f"False classifications deviations: {mean_deviation_from_false_classifications}", file=file)
        print(f"Overall mean deviation: {overall_mean_deviation}", file=file)

    return len(correct_classifications_distances) / len(distances)


def build_random_forest() -> RandomForestRegressor:
    """
    Build Random Forest Regressor (RFR):
    criterion: friedman_mse < absolute_error == squared_error < poisson
    max_features: log2 < sqrt < 19 < 1.0
    """

    return RandomForestRegressor(
        random_state=SEED,
        n_estimators=2 * HISTORY_LEN,
        max_features=100,
        criterion="poisson",
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
    )


def train_random_forest_and_predict(
    rf: RandomForestRegressor, X_train: np.array, X_test: np.array, y_train: np.array, save_dir: Path
) -> np.ndarray:
    """
    Trains a random forest and returns the predictions for the test data.
    """

    global HISTORY_LEN

    # Train random forest
    rf.fit(X_train, y_train)

    # Save random forest to file
    if not os.path.exists(save_dir):  # Create save dir, if it doesn't exist
        os.makedirs(save_dir)
    save_object_in_file(save_dir / "random_forest.pickle", rf)

    # Summarize random forest
    print(
        "Depths of decision trees in forest:",
        [estimator.get_depth() for estimator in rf.estimators_],
    )
    print(rf.estimators_)  # Trees in the forest
    print(rf.n_features_in_)  # dimension of input features x = 190
    print(rf.feature_importances_)  # Contribution value of each value in features (dimension 190)
    print(rf.n_outputs_)  # dimension of outputs y = 19
    print(rf.oob_score)
    print(rf.get_params())

    # Return predictions of random forest
    return rf.predict(X_test)


def build_LSTM() -> LSTM:
    """
    Builds and returns LSTM.
    """

    lstm = Sequential(
        [
            Conv1D(32, 3, padding="same", activation="relu"),
            LSTM(64, input_shape=(HISTORY_LEN, 19), kernel_initializer="HeUniform"),
            Dense(19, activation="sigmoid"),
        ]
    )

    # Compile/Set solver, loss and metrics
    lstm.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="huber",  # mse (more sensible to outliers and changes) is better than mae (less sensible to outliers -> less different values) and it has less loss => "get both worlds": huber loss
        metrics=[
            "acc"
            # tf.keras.metrics.FalseNegatives(),
        ],
    )

    return lstm


def train_and_test_LSTM(
    lstm: LSTM,
    X_train: np.array,
    X_test: np.array,
    y_train: np.array,
    y_test: np.array,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    callbacks: List[Any],
    save_dir: Path,
) -> np.ndarray:
    """
    Trains and test a LSTM and returns the predictions for the test data.
    """

    # Train und test LSTM
    history = lstm.fit(
        X_train,
        y_train,
        validation_split=0.0588,
        epochs=epochs,
        # steps_per_epoch=steps_per_epoch,  # Automatically calculated with: len(training_data) // batch_size
        batch_size=batch_size,
        callbacks=callbacks,
    )
    lstm.evaluate(X_test, y_test, batch_size=batch_size)

    # Save results in files
    if not os.path.exists(save_dir):  # Create save dir, if it doesn't exist
        os.makedirs(save_dir)
    lstm.save(save_dir / "lstm.keras")  # Save model

    # Plot accuracy and save it to file
    plt.plot(history.history["acc"], label="accuracy")
    plt.legend()
    plt.savefig(save_dir / "accuracy.png", bbox_inches="tight")
    plt.show()

    # Plot loss and save it to file
    plt.plot(history.history["loss"], label="loss")
    plt.legend()
    plt.savefig(save_dir / "loss.png", bbox_inches="tight")
    plt.show()

    # Return predictions of LSTM
    return lstm.predict(X_test, batch_size=batch_size)


def label_data(genre_names: List[str], all_movies: Dict[int, np.array], movie_ids: List[int], label: str) -> pd.DataFrame:
    """
    Creates a DataFrame with genre names as column names and rows for each
    movie from list of passed movie IDs "movie_ids".

    Parameters
    ----------
    genre_names : List[str]
        Names of genres like "Action", "Adventure", ..., which will be used as
        column names for DataFrame
    all_movies : Dict[int, np.array]
        Dictionary of all movies, where movie IDs are keys and real genres
        values
    movie_ids : List[int]
        List of all relevant movie IDs to insert into the resulting DataFrame
    label : str
        Label which will be added to the inserted data

    Returns
    -------
    pd.DataFrame
        Returns a DataFrame with genre names as column names and rows for each
        movie from list of passed movie IDs "movie_ids".
    """

    return pd.DataFrame(
        dict(
            [(genre_name, [all_movies[movie_id][i] for movie_id in movie_ids]) for i, genre_name in enumerate(genre_names)]
            + [("movie", movie_ids), ("label", [label] * len(movie_ids))]
        )
    )


def merge_data_for_visualization(
    label: str, X_movie_ids: List[int], y_movie_ids: List[int], predictions: np.ndarray, genre_names: List[str]
) -> pd.DataFrame:
    """
    Merges data (e.g. train data or test data) into one DataFrame for
    visualizing it. For this label inputs, outputs (targets/labels) and
    predictions with different labels.

    Parameters
    ----------
    label : str
        A label for naming the current dataset, e.g. "train" or "test"
    X_movie_ids : List[int]
        IDs of input movies (movie histories)
    y_movie_ids : List[int]
        IDs of output movies (targets/labels)
    predictions : np.ndarray
        Predictions of input data
    genre_names: List[str]
        Names of genres like "Action", "Adventure", ..., which will be used as
        column names for DataFrame

    Returns
    -------
    pd.DataFrame
        Returns merged data with different labels (input, target, predictions
        and unused). Unused is for movies, which weren't used, but still
        exists.
    """

    # Merge movies, whiche were used as input or output with predictions into one DataFrame
    df_X_performance = label_data(genre_names, all_movies, X_movie_ids, f"{label}_data_input")
    df_y_performance = label_data(genre_names, all_movies, y_movie_ids, f"{label}_data_target")
    df_preds_preformance = pd.DataFrame(
        dict(
            [(genre_name, [movie[i] for movie in predictions]) for i, genre_name in enumerate(genre_names)]
            + [("movie", [-1] * len(predictions)), ("label", [f"{label}_data_preds"] * len(predictions))]
        )
    )
    df_performance = pd.concat((df_X_performance, df_y_performance, df_preds_preformance))

    # Find other movies, which don't occur in (train/test) data
    ids_of_unused_movies = [movie_id for movie_id in all_movies.keys() if movie_id not in df_performance["movie"].values]
    df_other_movies = label_data(genre_names, all_movies, ids_of_unused_movies, "unused_movies")
    df_performance = pd.concat((df_performance, df_other_movies))

    # Sort DataFrame by IDs of movies
    df_performance = df_performance.sort_values(by=["movie"])

    # Output results
    print(f"df_X_{label}_performance: {df_X_performance.shape}")
    print(f"df_y_{label}_performance: {df_y_performance.shape}")
    print(f"df_preds_{label}_preformance: {df_preds_preformance.shape}")
    print(f"df_other_movies_{label}: {df_other_movies.shape}")
    print(f"df_{label}_performance: {df_performance.shape}")
    print(df_performance)

    return df_performance


class Model:
    """
    Wrapper class for loading and predicting different models

    Attributes
    ----------
    _model_type : str
        Type of model, e.g. "LSTM" or "RandomForestRegressor"
    _model_args : Dict[str, Any]
        Extra arguments for predicting with a specific model
    _model : Any
        Model which will be loaded and used for predicting
    """

    def __init__(self, model_type: str, path_to_model: Path = None, model_args: Dict[str, Any] = {}):
        """
        Creates a model object be loading a model from passed path.

        Parameters
        ----------
        model_type : str
            Type of model, necessary because different models will be
            loaded/called differently
        path_to_model : Path
            Load model from this path
        """

        self._model_type = model_type
        self._path_to_model = path_to_model
        self._model_args = model_args

        if path_to_model:
            print("Try to load model from file ...")

            if model_type == "LSTM":
                self._model = tf.keras.models.load_model(path_to_model)
            elif model_type == "RandomForestRegressor":
                self._model = load_object_from_file(path_to_model)
            else:
                raise ValueError("Unknown model name passed, cannot load any model!")

            print(f"Loaded successfully: {self._model_type}")

    def predict_one(self, x: Any) -> Any:
        """
        Pass an x value and get the prediction of the model for this.

        Parameters
        ----------
        x : Any
            x value for predicting a y value, which can be from any type

        Returns
        -------
        Any
            Returns the result, which can be of any type, depending on the model.
        """

        return self.predict_many(X=np.array([x]))

    def predict_many(self, X: np.ndarray) -> np.ndarray:
        """
        Pass x values and get the predictions of the model for these.

        Parameters
        ----------
        x : np.ndarray
            x values for predicting y values, which can be from any type

        Returns
        -------
        np.ndarray
            Returns the results as numpy array, which can be of any type, depending on the model.
        """

        predictions = np.empty(X.shape[0])

        if self._model_type == "LSTM":
            predictions = self._model.predict(x=X, batch_size=self._model_args["batch_size"])
        elif self._model_type == "RandomForestRegressor":
            predictions = self._model.predict(X=X)

        return np.array(predictions)

    def build_train_and_test_model(
        self,
        X_train: np.array,
        X_test: np.array,
        y_train: np.array,
        y_test: np.array,
        save_dir: Path,
    ) -> Tuple[np.ndarray, Path]:
        """
        Builds, trains, eventually tests model and returns predictions for passed
        test data.\n
        Returns save_dir, because some models will change the directory for saving
        files, because they input more details in the path to save.

        Parameters
        ----------
        X_train : np.array
            x values of train data
        X_test : np.array
            x values of test data
        y_train : np.array
            y values of train data
        y_test : np.array
            y values of test data
        save_dir : Path
            Directory to store/save model into

        Returns
        -------
        Tuple[np.ndarray, Path]
            Returns the predictions as first entry of the tuple and the path
            where the model is stored in the second entry.
        """

        # Define variables
        predictions = []

        # Output shapes of train and test data
        print(f"Train shapes: X: {X_train.shape}, y: {y_train.shape}")
        print(f"Test shapes: X: {X_test.shape}, y: {y_test.shape}\n")

        # Build, train and eventually test model
        if self._model_type == "RandomForestRegressor":
            # Unravel data = reduce dimension to 2 dimensions
            X_train = np.array([np.array(x).ravel() for x in X_train], dtype=np.float64)
            X_test = np.array([np.array(x).ravel() for x in X_test], dtype=np.float64)

            # Build and train model
            rf = build_random_forest()
            predictions = train_random_forest_and_predict(rf, X_train, X_test, y_train, save_dir)
            self._model = rf  # Save trained model
        elif self._model_type == "LSTM":
            # Define variables
            epochs = 50
            steps_per_epoch = X_train.shape[0]  # Currently irrelevant
            batch_size = 32
            self._model_args["batch_size"] = batch_size  # Save bacth size for predicting values
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=10, min_delta=0.0005, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
            ]
            save_dir = Path(f"{save_dir}_{epochs}_{steps_per_epoch}_{batch_size}")

            # Build LSTM
            lstm = build_LSTM()
            predictions = train_and_test_LSTM(
                lstm, X_train, X_test, y_train, y_test, epochs, steps_per_epoch, batch_size, callbacks, save_dir
            )
            self._model = lstm  # Save trained model
        else:
            print("No model chosen: Do nothing!")

        self._path_to_model = save_dir  # Store path to mdoel

        return predictions, self._path_to_model

    def find_similiar_movies(
        self, predicted_movie: np.ndarray, all_movies: Dict[int, Dict[str, Any]], n_closest_movies: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Searches in data of all movies for the n closest/most similar movies.
        The movies are sorted ascending, so closest/most similar movie comes
        first.\n
        Compute distance between real genres of movies with the squared
        euclidean distance (faster than the normal euclidean distance).\n
        Searches for real genres in "all_movies" by key "real_genres". This
        key must exist!

        Parameters
        ----------
        predicted_movie : np.ndarray
            The predicted movie to search for similiar movies
        all_movies : Dict[int, Dict[str, Any]]
            Dict of all movies containing similiar movies
        n_closest_movies : int, default 10
            Number of movies which, will be returned.

        Returns
        -------
        List[Dict[str, Any]]
            Returns the "n_closest_movies" movies. If
            len(all_movies) < n_closest_movies, then all movies will be
            returned, but not more.
        """

        sorted_movies = []

        for movie in all_movies.values():
            dist = cdist([predicted_movie], [movie["real_genres"]], "sqeuclidean")[0][0]  # Calc squared euclidean distance
            sorted_movies.append((dist, movie))

        return sorted(sorted_movies, key=lambda x: x[0])[:n_closest_movies]


if __name__ == "__main__":
    # Set seeds
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Define variables
    genre_names = [
        "Action",
        "Adventure",
        "Animation",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Family",
        "Fantasy",
        "History",
        "Horror",
        "Music",
        "Mystery",
        "Romance",
        "Science Fiction",
        "TV Movie",
        "Thriller",
        "War",
        "Western",
    ]
    model_type = "LSTM"  # "RandomForestRegressor", "LSTM"
    normalize_fator = 100
    output_factor = 1
    predictions = []

    # Define variables for data visualization
    all_movies = {}  # All movies as dict of {movie_id: real_genres_of_a_movie}

    # Öoad data
    if os.path.exists(extracted_features_file_path):  # Load extracted features from file
        used_histories, extracted_features = load_object_from_file(extracted_features_file_path)
    else:  # Extract features from all data
        # Load all movies from file
        user_movie_histories = load_object_from_file(
            # vars.user_history_file_path_with_real_genres  # TMDB data
            # vars.user_history_file_path_with_real_genres_and_reduced_dimensions  # TMDB data dimension reduced
            vars.user_watchings_file_path_with_real_genres  # Netflix prize data
            # vars.user_watchings_file_path_with_real_genres_small  # Netflix prize data (small part)
        )

        # Compute extracted features
        used_histories, extracted_features = extract_features(
            user_movie_histories,
            HISTORY_LEN,
            MIN_MOVIE_HISTORY_LEN,
            fill_history_len_with_zero_movies=FILL_HISTORY_LEN_WITH_ZERO_MOVIES,
            fine_grained_extracting=FINE_GRAINED_EXTRACTING,
        )
        save_object_in_file(extracted_features_file_path, (used_histories, extracted_features))

    # Save all movies to visualize it later
    all_movies_target = dict([(movie_id, target) for _, (movie_id, target) in extracted_features])
    all_movies_feature = dict(
        [(movie_id, real_genres) for movie_history, _ in extracted_features for movie_id, real_genres in movie_history]
    )
    all_movies = {**all_movies_target, **all_movies_feature}
    print(f"All movies: {len(all_movies)}")

    # used_histories, extracted_features = load_object_from_file(extracted_features_file_path)
    print(f"Max amount of available data (before): {len(extracted_features)}")
    label_occurences = Counter([movie_id for _, (movie_id, _) in extracted_features])
    relevant_histories = []
    history_counter = defaultdict(int)

    # Filter movies, which occur too often as target -> model gets of subset of data more variance
    for movie_history, (movie_id, target) in extracted_features:
        if history_counter[movie_id] < 100:  # 100 < label_occurences[movie_id] and
            relevant_histories.append((movie_history, (movie_id, target)))
            history_counter[movie_id] += 1
        # if 100 < label_occurences[movie_id]:
        #     relevant_histories.append((movie_history, target))

    print(f"Max amount of available data (after): {len(relevant_histories)}")
    extracted_features = relevant_histories

    # Read extracted features
    # used_histories, extracted_features = load_object_from_file(extracted_features_file_path)
    random.shuffle(extracted_features)  # Shuffle data
    max_histories_to_use = min(MAX_DATA, len(extracted_features))  # Use MAX_DATA, if MAX_DATA is available
    print(f"Max amount of available data: {len(extracted_features)}, take only: {max_histories_to_use}")
    history_step_size = len(extracted_features) // max_histories_to_use
    print(f"history_step_size: {history_step_size}")
    extracted_features = extracted_features[::history_step_size]  # Choose as much as possible different histories
    shapes = set([len(feature) for feature, _ in extracted_features])  # "_" = label
    print(used_histories, shapes)

    # Adjust variables because of new training data
    save_dir = Path(
        f"results/{max_histories_to_use}_{TRAIN_DATA_RELATIONSHIP}_{HISTORY_LEN}_{MIN_MOVIE_HISTORY_LEN}_{FILL_HISTORY_LEN_WITH_ZERO_MOVIES}_{FINE_GRAINED_EXTRACTING}"
    )

    # Split data into train and test data + normalize data
    X = [
        [(movie_id, np.array(real_genres, dtype=np.float64) / normalize_fator) for movie_id, real_genres in movie_history]
        for movie_history, _ in extracted_features
    ]  # LSTM
    y = [
        (movie_id, np.array(target, dtype=np.float64) / normalize_fator) for _, (movie_id, target) in extracted_features
    ]  # LSTM
    # X, y = np.array([one_hot_encoding_2d_arr(x) for x, _ in extracted_features], dtype=np.float64), np.array([one_hot_encoding_1d_arr(y) for _, y in extracted_features], dtype=np.float64)  # LSTM
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, train_size=TRAIN_DATA_RELATIONSHIP, random_state=SEED, shuffle=True
    )

    # Remove IDs from data
    X_train = np.array(
        [[real_genres for _, real_genres in movie_history] for movie_history in X_train_raw], dtype=np.float64
    )
    X_test = np.array([[real_genres for _, real_genres in movie_history] for movie_history in X_test_raw], dtype=np.float64)
    y_train = np.array([target for _, target in y_train_raw], dtype=np.float64)
    y_test = np.array([target for _, target in y_test_raw], dtype=np.float64)

    # Define hyper parameters of the LSTM
    model = Model(model_type=model_type)
    predictions, real_save_dir = model.build_train_and_test_model(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, save_dir=save_dir
    )

    # Save all predictions in file
    predictions_file_path = real_save_dir / "predictions.pickle"
    save_object_in_file(f"{predictions_file_path}", (X_test, y_test, predictions))

    # Find zero predictions = model predicting only null vectors, because it fits the most
    binary_predictions = [one_hot_encoding_1d_arr(prediction, factor=0.5 * output_factor) for prediction in predictions]
    zero_predictions = [True for prediction in binary_predictions if all(-1e-3 < pred_x < 1e-3 for pred_x in prediction)]

    # Output some predictions and true values/target labels
    with open(real_save_dir / "sample_prediction_outputs.txt", "w") as file:
        print("\nOutput some example predictions and true values:", file=file)
        print(f"{len(zero_predictions)} are zero predictions\n", file=file)

    # Test model with own evaluation function
    evaluate_model(y_test * output_factor, predictions * output_factor, real_save_dir, epsilon=EPSILON * output_factor)

    # for i in range(len(y_test)):
    with open(real_save_dir / "sample_prediction_outputs.txt", "a") as file:
        for i in range(5):
            distance = calc_distance_euclidean(y_test[i] * output_factor, predictions[i] * output_factor)

            if distance <= output_factor:
                print("Correct predicted:", file=file)
            else:
                print("False predicted:", file=file)

            print(X_test[i] * output_factor, file=file)
            print(y_test[i] * output_factor, file=file)
            print(predictions[i] * output_factor, file=file)
            print(
                one_hot_encoding_1d_arr(predictions[i] * output_factor, factor=0.5 * output_factor) * output_factor,
                file=file,
            )
            print(f"Distance: {distance}\n", file=file)

    print(len([1 for pred in zip(predictions) if np.allclose(predictions[0], pred, atol=0.1)]), len(predictions))
    print(len([1 for pred in zip(predictions) if np.allclose(predictions[0], pred, atol=0.2)]), len(predictions))
    print(len([1 for pred in zip(predictions) if np.allclose(predictions[0], pred, atol=0.3)]), len(predictions))

    # Save data for visualization
    # Find predictions, which will be done during training
    # model = Model(model_type=model_type, path_to_model=Path("results/50000_0.9_50_20_False_False_50_47930_32/lstm.keras"), model_args={"batch_size": 32})

    # Save performance of model on train data in file to visualize it later
    train_predictions = model.predict_many(X_train)
    movie_ids_X_train = sorted(set([movie_id for movie_history in X_train_raw for movie_id, _ in movie_history]))
    movie_ids_y_train = sorted(set([movie_id for movie_id, _ in y_train_raw]))

    df_train_performance = merge_data_for_visualization(
        label="train",
        X_movie_ids=movie_ids_X_train,
        y_movie_ids=movie_ids_y_train,
        predictions=train_predictions,
        genre_names=genre_names,
    )
    df_train_performance.to_pickle(real_save_dir / "train_data.dataframe")  # Save train data performance to file

    # Save performance of model on test data in file to visualize it later
    # predictions = model.predict_many(X_test)
    movie_ids_X_test = sorted(set([movie_id for movie_history in X_test_raw for movie_id, _ in movie_history]))
    movie_ids_y_test = sorted(set([movie_id for movie_id, _ in y_test_raw]))

    df_test_performance = merge_data_for_visualization(
        label="test",
        X_movie_ids=movie_ids_X_test,
        y_movie_ids=movie_ids_y_test,
        predictions=predictions,
        genre_names=genre_names,
    )
    df_test_performance.to_pickle(real_save_dir / "test_data.dataframe")  # Save test data performance to file
