import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, LSTM
from typing import Dict, List, Tuple

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[1]
sys.path.append(str(project_dir))

import helper.variables as vars

from helper.file_system_interaction import load_object_from_file, save_object_in_file


# Define constants
HISTORY_LEN = 10
MIN_MOVIE_HISTORY_LEN = 5
DISTANCE_TO_OTHER_MOVIES = 0.1
TRAIN_DATA_RELATIONSHIP = 0.85
SEED = 1234

# Constants for computing the difference between multiple values
EPSILON = 100  # This mean thah prediction can lay one genre next to the correct one
INDEPENDENT_MAX_DIFF_PER_GENRE = 5
NUMBER_OF_INTERVALS = 5

# Set print options for numpy
np.set_printoptions(formatter={"all": lambda x: "{0:0.3f}".format(x)})


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
    fill_history_len_with_zero_movies=True,
) -> List[Tuple[np.array, np.array]]:
    """
    Extract features: partionate user histories into parts with length
    "min_movie_history_len" so that next movie is the predicted target.
    Returns tuples consisting of the last seen movies and the next one
    to predict (= target, label) out ot the previous ones.
    """

    all_extracted_features = []
    skipped_histories, used_histories = 0, 0

    for (
        users_movie_history
    ) in user_movie_histories.values():  # Iterate over all users' histories
        if len(users_movie_history) < min_movie_history_len or (
            (not fill_history_len_with_zero_movies)
            and len(users_movie_history) <= movie_history_len
        ):  # User has not enough movies watched
            skipped_histories += 1
            continue
        elif (
            fill_history_len_with_zero_movies
            and len(users_movie_history) <= movie_history_len
        ):  # Use has watched enoguh movies, but not many
            # Find movies and target/label
            movies = users_movie_history[:-1]
            target_label = users_movie_history[-1]

            # Fill missing movies with zeros
            number_of_missing_movies = movie_history_len - len(movies)
            zero_movie = np.zeros(
                target_label.shape[0]
            )  # Create movie containing only 0 for all real genres
            zero_movies = list(np.tile(zero_movie, (number_of_missing_movies, 1)))

            # Create one list with zero movies and watched movies of a user
            history_feature = (zero_movies + movies, target_label)
            all_extracted_features.append(history_feature)
        else:  # Use history only, if it is long enough
            all_extracted_features.extend(
                [
                    (
                        np.copy(users_movie_history[i: i + movie_history_len]),
                        users_movie_history[i + movie_history_len],
                    )
                    for i in range(
                        0,
                        len(users_movie_history) - movie_history_len - 1,
                        movie_history_len,
                    )
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
        mean_deviation_from_correct_classifications = sum(
            correct_classifications_distances
        ) / len(correct_classifications_distances)
    else:
        mean_deviation_from_correct_classifications = -1

    if false_classifications_distances != []:
        mean_deviation_from_false_classifications = sum(
            false_classifications_distances
        ) / len(false_classifications_distances)
    else:
        mean_deviation_from_false_classifications = -1

    print(
        f"\nCorrect classifications: {len(correct_classifications_distances)},"
        + f"false classifications: {len(false_classifications_distances)}, "
        + f"accuracy: {len(correct_classifications_distances) / len(distances)}"
    )
    print(
        f"Correct classifications deviations: {mean_deviation_from_correct_classifications}"
    )
    print(
        f"False classifications deviations: {mean_deviation_from_false_classifications}"
    )
    print(f"Overall mean deviation: {overall_mean_deviation}")

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
    rf: RandomForestRegressor, X_train: np.array, X_test: np.array, y_train: np.array
) -> np.ndarray:
    """
    Trains a random forest and returns the predictions for the test data.
    """

    global HISTORY_LEN

    # Train random forest
    rf.fit(X_train, y_train)

    # Summarize random forest
    print(
        "Depths of decision trees in forest:",
        [estimator.get_depth() for estimator in rf.estimators_],
    )
    print(rf.estimators_)  # Trees in the forest
    print(rf.n_features_in_)  # dimension of input features x = 190
    print(
        rf.feature_importances_
    )  # Contribution value of each value in features (dimension 190)
    print(rf.n_outputs_)  # dimension of outputs y = 19
    print(rf.oob_score)
    print(rf.get_params())

    # Return predictions of random forest
    return rf.predict(X_test)


def build_LSTM() -> LSTM:
    """
    Builds and returns LSTM.
    """

    lstm = tf.keras.models.Sequential(
        [
            # Embedding
            # Embedding(input_dim=100, output_dim=19, input_length=10),
            # Dense(8, activation="relu", input_shape=(10, 19)),
            # LSTM
            # LSTM(128, return_sequences=True, input_shape=(HISTORY_LEN, 19), stateful=True),
            LSTM(128, return_sequences=True, input_shape=(HISTORY_LEN, 19)),
            Dropout(0.2),  # Avoid learning data by heart
            LSTM(64),
            Dropout(0.2),  # Avoid learning data by heart
            # Decision layers at the end, using processed data from LSTM
            Dense(32, activation="softmax"),
            Dense(19, activation="sigmoid"),
        ]
    )

    # Compile/Set solver, loss and metrics
    lstm.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        # optimizer='rmsprop',
        loss=tf.keras.losses.BinaryCrossentropy(),
        # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.2),
        # loss=tf.keras.losses.CosineSimilarity(),
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
) -> np.ndarray:
    """
    Trains and test a LSTM and returns the predictions for the test data.
    """

    # Train und test LSTM
    history = lstm.fit(
        X_train,
        y_train,
        validation_split=0.05,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
    )
    lstm.evaluate(X_test, y_test, batch_size=batch_size)

    # Plot accuracy
    plt.plot(history.history["acc"], label="accuracy")
    plt.legend()
    plt.show()

    # Plot loss
    plt.plot(history.history["loss"], label="loss")
    plt.legend()
    plt.show()

    # Return predictions of LSTM
    return lstm.predict(X_test, batch_size=batch_size)


def build_train_and_test_model(
    model_variant,
    X_train: np.array,
    X_test: np.array,
    y_train: np.array,
    y_test: np.array,
) -> np.ndarray:
    """
    Builds, trains, eventually tests model and returns
    predictions for passed test data.
    """

    # Define variables
    predictions = []

    # Output shapes of train and test data
    print(f"Train shapes: X: {X_train.shape}, y: {y_train.shape}")
    print(f"Test shapes: X: {X_test.shape}, y: {y_test.shape}\n")

    # Build, train and eventually test model
    if model_variant == 0:
        # Unravel data = reduce dimension to 2 dimensions
        X_train = np.array([np.array(x).ravel() for x in X_train], dtype=np.float64)
        X_test = np.array([np.array(x).ravel() for x in X_test], dtype=np.float64)

        # Build and train model
        rf = build_random_forest()
        predictions = train_random_forest_and_predict(rf, X_train, X_test, y_train)
    elif model_variant == 1:
        # Define variables
        epochs = 40
        steps_per_epoch = 40
        batch_size = 1

        # Build LSTM
        lstm = build_LSTM()
        predictions = train_and_test_LSTM(
            lstm, X_train, X_test, y_train, y_test, epochs, steps_per_epoch, batch_size
        )
    else:
        print("No model chosen: Do nothing!")

    return predictions


if __name__ == "__main__":
    # Set seeds
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Define variables
    model_number = 0  # 0: Random Forest, 1: LSTM
    normalize_fator = 100
    output_factor = 1
    predictions = []

    # Visualize data
    # TODO: Look for mean genres
    # TODO: Compare genres
    # TODO: Eigene Loss-Funktion definieren???
    # TODO: Daten genau anschauen und herausfinden bei welchen Inputs Modell versagt -> immer die selben oder andere??? (auch mit shuffle der Daten ausprobieren)
    # TODO: Embedding für LSTM machen -> für Random Forest auch???

    # Compute based on this extracted features
    user_movie_histories = load_object_from_file(
        vars.user_history_file_path_with_real_genres
    )
    used_histories, extracted_features = extract_features(
        user_movie_histories,
        HISTORY_LEN,
        MIN_MOVIE_HISTORY_LEN,
        fill_history_len_with_zero_movies=False,
    )
    save_object_in_file(
        vars.extracted_features_file_path, (used_histories, extracted_features)
    )

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
    used_histories, extracted_features = load_object_from_file(
        vars.extracted_features_file_path
    )
    shapes = set([len(f) for f, l in extracted_features])
    print(used_histories, shapes)

    # Split data into train and test data
    X, y = np.array(
        [np.array(x) for x, _ in extracted_features], dtype=np.float64
    ), np.array(
        [np.array(y, dtype=np.float64) for _, y in extracted_features], dtype=np.float64
    )  # LSTM
    # X, y = np.array([one_hot_encoding_2d_arr(x) for x, _ in extracted_features], dtype=np.float64), np.array([one_hot_encoding_1d_arr(y) for _, y in extracted_features], dtype=np.float64)  # LSTM
    X, y = X / normalize_fator, y / normalize_fator  # Normize data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_DATA_RELATIONSHIP, random_state=SEED, shuffle=False
    )

    # Define hyper parameters of the LSTM
    predictions = build_train_and_test_model(
        model_number, X_train, X_test, y_train, y_test
    )

    # Find zero predictions = model predicting only null vectors, because it fits the most
    binary_predictions = [
        one_hot_encoding_1d_arr(prediction) for prediction in predictions
    ]
    zero_predictions = [
        True
        for prediction in binary_predictions
        if all(-1e-3 < pred_x < 1e-3 for pred_x in prediction)
    ]

    # Output some predictions and true values/target labels
    print("\nOutput some example predictions and true values:")
    print(f"{len(zero_predictions)} are zero predictions")

    # Test model with own evaluation function
    evaluate_model(
        y_test * output_factor, predictions * output_factor, epsilon=output_factor
    )

    # for i in range(len(y_test)):
    for i in range(3):
        distance = calc_distance_euclidean(
            y_test[i] * output_factor, predictions[i] * output_factor
        )

        if distance <= output_factor:
            print("Correct:")
        else:
            print("False:")

        print(X_test[i] * output_factor)
        print(y_test[i] * output_factor)
        print(predictions[i] * output_factor)
        print(
            one_hot_encoding_1d_arr(predictions[i], factor=0.5 * output_factor)
            * output_factor
        )
        print(f"Distance: {distance}")
        print()
