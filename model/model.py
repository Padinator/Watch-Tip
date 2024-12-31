import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import tensorflow as tf

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
HISTORY_LEN = 30
MIN_MOVIE_HISTORY_LEN = 10
DISTANCE_TO_OTHER_MOVIES = 0.1
TRAIN_DATA_RELATIONSHIP = 0.9
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
save_dir = Path(f"results/{MAX_DATA}_{TRAIN_DATA_RELATIONSHIP}_{HISTORY_LEN}_{MIN_MOVIE_HISTORY_LEN}")


class MinPooling2D(MaxPool1D):

    def __init__(self, pool_size, strides=None, padding="valid", data_format=None, **kwargs):
        super(MaxPool1D, self).__init__(pool_size, strides, padding, data_format, **kwargs)

    def pooling_function(inputs, pool_size, strides, padding, data_format):
        return -backend.pool1d(-inputs, pool_size, strides, padding, data_format, pool_mode="max")


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
    user_movie_histories: Dict[int, List[np.array]],
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
    user_movie_histories : Dict[int, List[np.array]]
        Grouped/Unraveled movies histories per user, e.g.:
        {
            1038924: [
                [10.2, 39.3, 59.5, 56.4, 12.8, 0.76, 96.4, 21.3, 69.0, 98.5,
                28.2, 49.1, 19.01, 39.4, 18.9, 38.2, 98.5, 25.6, 9.3],  # Real genres of first movie of user 1038924\n
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
            target_label = users_movie_history[-1]

            # Fill missing movies with zeros
            number_of_missing_movies = movie_history_len - len(movies)
            zero_movie = np.zeros(len(target_label))  # Create movie containing only 0 for all real genres
            zero_movies = list(np.tile(zero_movie, (number_of_missing_movies, 1)))

            # Create one list with zero movies and watched movies of a user
            history_feature = (zero_movies + movies, target_label)
            all_extracted_features.append(history_feature)
        else:  # Use history only, if it is long enough
            all_extracted_features.extend(
                [
                    (
                        np.copy(users_movie_history[i:i + movie_history_len]),
                        users_movie_history[i + movie_history_len],
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
            # LSTM
            # LSTM(128, return_sequences=True, input_shape=(HISTORY_LEN, 19), stateful=True),
            # LSTM(
            #     256 * 2,
            #     return_sequences=True,
            #     input_shape=(HISTORY_LEN, 19),
            #     kernel_initializer="HeUniform",
            #     kernel_regularizer="l1_l2"
            # ),
            # Dropout(0.3, seed=SEED),
            # LSTM(128 * 2, return_sequences=True, kernel_initializer="HeUniform"),
            # Dropout(0.3, seed=SEED),  # No dropout, else time relevant data/context can be lost
            # LSTM(64 * 2, return_sequences=False, kernel_initializer="GlorotUniform"),
            # # Decision layers at the end, using processed data from LSTM
            # Dense(64 * 4, activation="relu"),
            # # Dropout(0.4, seed=SEED),  # Dropout for don't learning by heart
            # Dense(64, activation="relu"),
            # # Dropout(0.4, seed=SEED),  # Dropout for don't learning by heart
            # Dense(19, activation="sigmoid"),

            # Convolutional modell
            # Conv1D(128, 2, activation="relu", padding="same"),
            # MaxPool1D(2, padding="same", strides=1),
            # Conv1D(128, 2, activation="relu", padding="same"),
            # MaxPool1D(2, padding="same", strides=1),
            # Conv1D(64, 2, activation="relu", padding="same"),
            # MaxPool1D(2, padding="same", strides=1),
            # Conv1D(64, 2, activation="relu", padding="same"),
            # MaxPool1D(2, padding="same", strides=1),
            # Conv1D(32, 2, activation="relu", padding="same"),
            # MaxPool1D(2, padding="same", strides=1),
            # Flatten(),
            # Dense(64, activation="relu"),
            # LSTM(8, kernel_initializer="HeUniform"),
            # Dense(19, activation="sigmoid"),
            # Dense(19, activation="sigmoid"),

            # LSTM
            Conv1D(32, 2, padding="same"),
            # SimpleRNN(64, input_shape=(HISTORY_LEN, 19), kernel_initializer="HeUniform"),
            # GRU(64, input_shape=(HISTORY_LEN, 19), kernel_initializer="HeUniform"),
            LSTM(64, input_shape=(HISTORY_LEN, 19), kernel_initializer="HeUniform"),
            # LSTM(256, input_shape=(HISTORY_LEN, 19), kernel_initializer="HeUniform"),
            # Dense(64 * 4, activation="relu"),
            # Dropout(0.4, seed=SEED),  # Dropout for don't learning by heart
            # Dense(64, activation="relu"),
            # Dropout(0.5, seed=SEED),  # Dropout for don't learning by heart
            Dense(19, activation="sigmoid"),

            # RNN
            # SimpleRNN(128, activation="tanh", input_shape=(HISTORY_LEN, 19), return_sequences=True),
            # Dropout(0.2),  # Avoid learning data by heart
            # SimpleRNN(64, activation="tanh", input_shape=(HISTORY_LEN, 19), return_sequences=True),
            # Dropout(0.2),  # Avoid learning data by heart
            # SimpleRNN(32, activation="tanh", input_shape=(HISTORY_LEN, 19)),
            # Dense(64, activation="softmax"),
            # Dropout(0.2),  # Avoid learning data by heart
            # Dense(32, activation="softmax"),
            # Dropout(0.2),  # Avoid learning data by heart
            # Dense(19, activation="sigmoid"),

            # # RNN (dimension reduced data)
            # SimpleRNN(128, activation="tanh", input_shape=(HISTORY_LEN, 3), return_sequences=True),
            # Dropout(0.2),  # Avoid learning data by heart
            # SimpleRNN(64, activation="tanh", input_shape=(HISTORY_LEN, 3), return_sequences=True),
            # Dropout(0.2),  # Avoid learning data by heart
            # SimpleRNN(32, activation="tanh", input_shape=(HISTORY_LEN, 3)),
            # Dense(64, activation="softmax"),
            # Dropout(0.2),  # Avoid learning data by heart
            # Dense(32, activation="softmax"),
            # Dropout(0.2),  # Avoid learning data by heart
            # Dense(3, activation="linear"),
            # SimpleRNN < (little bit) LSTM
        ]
    )

    # Compile/Set solver, loss and metrics
    lstm.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        # optimizer='rmsprop',
        # loss=tf.keras.losses.BinaryCrossentropy(),
        # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.2),
        # loss=tf.keras.losses.CosineSimilarity(),
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
        validation_split=0.1,
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

        predictions = np.empty()

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
    model_type = "LSTM"  # "RandomForestRegressor", "LSTM"
    normalize_fator = 100
    output_factor = 1
    predictions = []

    # Visualize data
    # TODO: Look for mean genres
    # TODO: Compare genres
    # TODO: Which inputs will be used? -> use a movie not too often as target/label (else biased)

    # # Compute extracted features
    # user_movie_histories = load_object_from_file(
    #     # vars.user_history_file_path_with_real_genres  # TMDB data
    #     # vars.user_watchings_file_path_with_real_genres  # Netflix prize data
    #     vars.user_history_file_path_with_real_genres_and_reduced_dimensions  # Netflix prize data (small part)
    # )
    # used_histories, extracted_features = extract_features(
    #     user_movie_histories,
    #     HISTORY_LEN,
    #     MIN_MOVIE_HISTORY_LEN,
    #     fill_history_len_with_zero_movies=False,
    #     fine_grained_extracting=False,
    # )
    # save_object_in_file(
    #     vars.extracted_features_file_path, (used_histories, extracted_features)
    # )

    """
    # df_user_movie_histories_reduced_dim = load_object_from_file(vars.user_history_file_path_with_real_genres_and_reduced_dimensions_visualization)

    # # Scale data
    # transformed_values = StandardScaler().fit_transform(df_user_movie_histories_reduced_dim.loc[:, df_user_movie_histories_reduced_dim.columns != "username"].values)
    # df_user_movie_histories_reduced_dim = pd.DataFrame({"dim1": transformed_values[:, 0],
    #                                                 "dim2": transformed_values[:, 1],
    #                                                 "dim3": transformed_values[:, 2],
    #                                                 "username": df_user_movie_histories_reduced_dim["username"].values})

    # usernames = list(load_object_from_file(vars.user_history_file_path_with_real_genres).keys())
    # columns_except_username = [col for col in df_user_movie_histories_reduced_dim if col != "username"]
    # user_movie_histories_reduced_dim = {}  # Store all dimension reduced real movie genres per user

    # # Group movies by users and use sorting of object "user_movie_histories"
    # for username in usernames:
    #     rows = df_user_movie_histories_reduced_dim.loc[df_user_movie_histories_reduced_dim["username"] == username, columns_except_username]
    #     user_movie_histories_reduced_dim[username] = rows.values
    # user_movie_histories = user_movie_histories_reduced_dim
    """

    # Read extracted features
    used_histories, extracted_features = load_object_from_file(vars.extracted_features_file_path)
    random.shuffle(extracted_features)  # Shuffle data
    max_histories_to_use = min(MAX_DATA, len(extracted_features))  # Use MAX_DATA, if MAX_DATA is available
    print(f"Max amount of available data: {len(extracted_features)}, take only: {max_histories_to_use}")
    history_step_size = len(extracted_features) // max_histories_to_use
    extracted_features = extracted_features[::history_step_size]  # Choose as much as possible different histories
    shapes = set([len(feature) for feature, label in extracted_features])
    print(used_histories, shapes)

    # Adjust variables because of new training data
    save_dir = Path(f"results/{max_histories_to_use}_{TRAIN_DATA_RELATIONSHIP}_{HISTORY_LEN}_{MIN_MOVIE_HISTORY_LEN}")

    # Split data into train and test data
    X, y = np.array([np.array(x) for x, _ in extracted_features], dtype=np.float64), np.array(
        [np.array(y, dtype=np.float64) for _, y in extracted_features], dtype=np.float64
    )  # LSTM
    # X, y = np.array([one_hot_encoding_2d_arr(x) for x, _ in extracted_features], dtype=np.float64), np.array([one_hot_encoding_1d_arr(y) for _, y in extracted_features], dtype=np.float64)  # LSTM
    X, y = X / normalize_fator, y / normalize_fator  # Normize data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_DATA_RELATIONSHIP, random_state=SEED, shuffle=True
    )

    # Define hyper parameters of the LSTM
    model = Model(model_type=model_type)
    predictions, real_save_dir = model.build_train_and_test_model(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, save_dir=save_dir
    )

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
        for i in range(3):
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
