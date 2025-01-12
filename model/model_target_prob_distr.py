import os
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import tensorflow as tf

from collections import Counter
from gensim.models import Word2Vec
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    Flatten,
)
from tensorflow.keras import Sequential
from typing import Any, Dict, List, Tuple, Set

# ---------- Import own python modules ----------
project_dir = Path(os.path.abspath(__file__)).parents[1]
sys.path.append(str(project_dir))

import helper.variables as vars

from database.movie import Movies
from helper.file_system_interaction import load_object_from_file, save_object_in_file


# Define some constants for feature extraction
WINDOW_SIZE = 2
NO_EMBEDDING = {}
REAL_GENRES_EMBEDDING = {"embedding": "real_genres"}
RANDOM_EMBEDDING = {"embedding": "random", "embedding_dim": 100}
SKIP_GRAMS_EMBEDDING = {"embedding": "skip_grams"}
WORD2VEC_EMBEDDING = {"embedding": "Word2Vec", "embedding_dim": 100, "window_size": WINDOW_SIZE}
ALL_EMBEDDINGS = [NO_EMBEDDING, REAL_GENRES_EMBEDDING, RANDOM_EMBEDDING, SKIP_GRAMS_EMBEDDING, WORD2VEC_EMBEDDING]
MIN_MOVIE_OCCURENCE = 10000

# Define constants for model training
MAX_DATA = 40000
TRAIN_DATA_RELATIONSHIP = 0.85
VALIDATION_DATA_RELATIONSHIP = 0.05
EPOCHS = 30
BATCH_SIZE = 32

# Define evaluatio nconstants
EPSILON_REAL_GENRE = 10  # Maximal difference between two sinlge real genres of two movies

# Define other constants
SEED = 1234
EXTRACTED_FEATURES_BASE_PATH = Path("extracted_features/target_prob_distr")
MODEL_BASE_PATH = Path("results/target_prob_distr")

# Set seeds
np.random.seed(seed=SEED)
random.seed(SEED)
tf.random.set_seed(seed=SEED)

# Set print options for numpy
np.set_printoptions(formatter={"all": lambda x: "{0:0.3f}".format(x)}, threshold=sys.maxsize)


def extract_features(
    user_movie_histories: Dict[int, List[Tuple[int, np.array]]],
    window_size: int = 2,
    input_features_one_hot_encoded: bool = True,
    embedding: Dict[str, Any] = {},
    min_movie_occurence: int = 1,
) -> Tuple[int, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Extract features: create skip grams with passed window size.

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
    window_size : int, default 2
        Size for creating skip grams/maximal distance between two movies for
        declaring them as neighbours for input feature and prediction
    input_features_one_hot_encoded : bool, default True
        If True input features for a model will be one hot encoded, else they
        will remain as real genres of a movie.
    embedding : Dict[tr, Any], default False
        Pass {"embedding": "random", "embedding_dim": <int>} for creating a
        random embedding.\n
        Pass {"embedding": "real_genres"} for creating an embedding wih real
        genres. The "embedding_dim" will be automatically set to 19.\n
        Pass {"embedding": "skip_grams"} for creating an
        embedding based on skip grams.\n
        Pass {"embedding": "Word2Vec", "embedding_dim": <int>, "window_size": <int>}
        for creating an embedding based on Word2Vec (use concatenated movie
        IDs), uses its weights.
    min_movie_occurence : int, default 1
        Minimal occurence of a movie in histories of all users, else a movie
        won't be taken into account

    Returns
    -------
    Tuple[\n
        int,\n
        List[Tuple[np.ndarray, np.ndarray]],
        np.nradday\n
    ]
        Reurns a tuple, first the number of movies and second the list of all
        tuples with input and output data for a model:\n
        (\n
            <input feature as 19-dim array>,\n
            <target/label as binary feature/one hot encoded array>\n
        )\n
        The last entry, which will be returned, is the embedding matrix.
    """

    # Define variables
    use_first_n_users = 100
    movie_number_mapping = {}  # Saves movies and maps them to a unique index
    bigrams = []  # Store bigrams with tuples of input and output features
    embedding_matrix = None

    # Find all movies for building skip grams
    all_movie_ids = [movie_id for user_movies in user_movie_histories.values() for movie_id, _ in user_movies]

    # Find all relevant movie IDs (more than e.g. 10,000 occurenes)
    movie_counter = Counter(all_movie_ids)
    relevant_movie_ids = sorted(
        set([movie_id for movie_id, occurences in movie_counter.items() if min_movie_occurence <= occurences])
    )
    print(f"Number of relevant movies: {len(relevant_movie_ids)}")

    # Create mapping between movies (sorted list) and index for one hot encoding of movies
    movie_number_mapping = dict([(movie_id, i) for i, movie_id in enumerate(relevant_movie_ids)])
    total_number_of_movies = len(movie_number_mapping)

    # Build skip grams
    iteration = 0
    user_iteration = 0
    """
    Stores as keys a movie ID, as value another dict with a movie ID as
    context of the first movie ID and as value a counter for counting the
    ouccrences
    """
    bigrams_raw = dict(
        [
            (movie_id, dict([(other_movie_id, 0) for other_movie_id in relevant_movie_ids]))
            for movie_id in relevant_movie_ids
        ]
    )

    for user_movies in user_movie_histories.values():  # Iterate over all movies of a user
        if user_iteration % 10 == 0:
            print(f"User iteration: {user_iteration}")
        if user_iteration == use_first_n_users:  # Only use first 100 users
            break
        user_iteration += 1

        for i, (movie_id, real_genres) in enumerate(user_movies):  # Iterate over all movies of a user
            if movie_id in relevant_movie_ids:  # Only use relevant movie IDs
                first_movie_index = max(0, i - window_size)  # First movie included in bigram
                last_movie_index = min(
                    len(user_movie_histories), i + window_size
                )  # Index of last movie + 1 included in bigram

                for other_movie_id, _ in user_movies[
                    first_movie_index:last_movie_index
                ]:  # Iterate over all movies in the window
                    if other_movie_id in relevant_movie_ids and movie_id != other_movie_id:  # Bigram not found
                        if iteration % 10000 == 0:
                            print(f"Movie iteration: {iteration}")
                        iteration += 1

                        # Save real genres with one hot encoded targets
                        if input_features_one_hot_encoded:  # One hot encode movie
                            input_feature = one_hot_encode_with_mapping(element=movie_id, mapping=movie_number_mapping)
                        else:  # Use real genres of a movie
                            input_feature = np.array(real_genres, dtype=np.float64)

                        # One hot encode target and save input and output
                        one_hot_encoded_target = one_hot_encode_with_mapping(
                            element=other_movie_id, mapping=movie_number_mapping
                        )
                        bigrams.append((input_feature, one_hot_encoded_target))
                        bigrams_raw[movie_id][other_movie_id] += 1  # Count occurences of bigrams

    # Create matrix for embedding
    first_embedding_matrix_dim = bigrams[0][0].shape[0]

    if not embedding:  # No embedding was passed
        pass
    elif embedding["embedding"] == "random" or (
        embedding["embedding"] in ["real_genres", "skip_grams", "Word2Vec"]
        and first_embedding_matrix_dim != total_number_of_movies
    ):  # Random ebedding
        if embedding["embedding"] in ["real_genres", "skip_grams"]:  # Set missing dimension of embedding
            embedding["embedding_dim"] = 100

        embedding_matrix = np.random.rand(first_embedding_matrix_dim, embedding["embedding_dim"])
    elif embedding["embedding"] == "real_genres":  # Embedding based on counted real genres
        embedding_matrix = np.zeros((total_number_of_movies, 19))
        all_movies_real_genres_mapping = dict(
            [
                (movie_id, real_genres)
                for _, user_movies in user_movie_histories.items()
                for movie_id, real_genres in user_movies
            ]
        )  # Find real genres of all passed movies and save them in a dict

        # Save real genres of users in the embedding matrix
        for i, movie_id in enumerate(relevant_movie_ids):
            real_genres = all_movies_real_genres_mapping[movie_id]
            embedding_matrix[i] = real_genres / np.max(real_genres)  # Normalize real genres
    elif embedding["embedding"] == "skip_grams":  # Embedding based on counted skip grams
        embedding_matrix = np.zeros((total_number_of_movies, total_number_of_movies))  # Quadratic matrix over all movies

        # Copy all occurences
        for i, movie_id in enumerate(relevant_movie_ids):
            for j, other_movie_id in enumerate(relevant_movie_ids):
                embedding_matrix[i, j] = bigrams_raw[movie_id][other_movie_id]

            # Normalize all vectors
            if 1e-3 < np.max(embedding_matrix[i]):
                embedding_matrix[i] /= np.max(embedding_matrix[i])
    elif embedding["embedding"] == "Word2Vec":  # Embedding based on Word2Vec
        embedding_matrix = np.zeros((total_number_of_movies, embedding["embedding_dim"]))

        # Create model Word2Vec with movie IDs and their context
        word2vec_sentences = [
            [str(movie_id) for movie_id, _ in user_movies]
            for user_movies in list(user_movie_histories.values())[:use_first_n_users]
        ]
        model = Word2Vec(
            sentences=word2vec_sentences,
            vector_size=embedding["embedding_dim"],
            window=embedding["window_size"],
            min_count=1,
            sg=1,
        )

        # Extract from Word2Vec model the embeddings
        for i, movie_id in enumerate(relevant_movie_ids):
            if str(movie_id) in model.wv:
                embedding_matrix[i] = model.wv[str(movie_id)]
            else:
                embedding_matrix[i] = np.zeros(embedding["embedding_dim"])
    else:  # Found unknown embedding
        raise Exception(f"Unknown embedding: {embedding}")

    return total_number_of_movies, bigrams, embedding_matrix


def one_hot_encode_with_mapping(element: Any, mapping: Dict[Any, int]) -> np.ndarray:
    """
    Creates and returns a one hot encoding of an element. This element must
    exist/be contained in the passed mapping "mapping".

    Parameters
    ----------
    element : Any
        Element to one hot encode, e.g. "The lord of the rings"
    mapping : Dict[Any, int]
        Map for finding elements and their index in the one hot encoded array.
        Passed map "mapping" must contain element, e.g.:\n
        {\n
            "The lord of the rings": 0,\n
            "Spider Man 1": 1,\n
            "Spider Man 2": 2\n
            ...\n
        }

    Returns
    -------
    np.ndarray
        Returns the one hot encoding of the passed element.
    """

    one_hot_encoding = np.zeros(len(mapping))
    index_of_one_entry = mapping[element]
    one_hot_encoding[index_of_one_entry] = 1

    return one_hot_encoding


def prepare_data(
    extracted_features: List[Tuple[np.ndarray, np.ndarray]], train_data_relationship: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data with following steps:\n
    1. Do genative sampling\n
    2. Split data into train and test + shuffling
    3. Output resulting shapes and one example row

    Parameters
    ----------
    extracted_features : List[Tuple[np.ndarray, np.ndarray]]
        Extracted features with utples consisting of input
        features and targets/labels
    train_data_relationship : float
        Relationship of train data for splitting train and test
        data

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Returns a tuple consisting of train data (x),
        test data (x), train data (y) and test data (y).
    """

    # Do negative sampling/Add negative samples
    # res = []

    # # Each tenth value add a negative sample/random value
    # for i, (x_i, y_i) in enumerate(extracted_features):
    #     if i % BATCH_SIZE == 0:
    #         fake_y_i = np.random.rand(*y_i.shape)
    #         res.append((x_i, fake_y_i))
    #     res.append((x_i, y_i))

    # extracted_features = res

    # Split data into train and test data
    X, y = np.array([x for x, _ in extracted_features], dtype=np.float64), np.array(
        [y for _, y in extracted_features], dtype=np.float64
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_data_relationship, random_state=SEED, shuffle=True
    )

    print("Train shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print("Test shapes:")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(X_train[0].shape, y_train[0].shape)
    print(X_train[0], y_train[0])

    return X_train, X_test, y_train, y_test


def build_model(
    total_number_of_movies: int, embedding_matrix: np.ndarray = None, input_dim: int = None, input_length: int = None
) -> "Sequential":
    """
    Build and returns model that can be trained.

    Parameters
    ----------
    total_number_of_movies : int
        Total number of different/unique movies
    embedding_matrix : np.ndarray, default None
        Matrix that will be used as embedding (optional)
    input_dim: int, default None
        Input dimension (size of vocabulary), only necessary, if an
        embedding is used
    input_length : int, default None
        Length of input that will be multiplied with the embedding matrix

    Returns
    -------
    Sequential
        Returns the tensorflow model consisting of several layers.
    """

    # Create model
    model = Sequential()

    # Add embedding layer
    if embedding_matrix is not None:
        print(embedding_matrix.shape)
        print(input_dim, embedding_matrix.shape[1], input_length)
        model.add(
            Embedding(
                input_dim=input_dim,  # Number of different possible movies
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                input_length=input_length,  # Same as embedding_matrix.shape[0]
                trainable=True,
            )
        )
        model.add(Flatten())

    # Add other layers
    # model.add(Dense(100, activation="relu"))
    # model.add(Dropout(0.3))
    model.add(Dense(total_number_of_movies, activation="softmax"))

    # Compile/Set solver, loss and metrics
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
        ],
    )

    return model


def find_movies_of_one_hot(
    one_hot_encoded_vector: np.ndarray, all_movie_ids_sorted_set: Set[str]
) -> List[Tuple[str, float]]:
    """
    Finds based on a movies one hot encoding the movie ID. Sorts
    the result, so that the movie with the highest probability
    occurs first.

    Parameter
    ---------
    one_hot_encoded_vector : np.ndarray
        One hot encoded numpy array, to which th emovie ID will be
        searched
    all_movie_ids_sorted_set : Set[str]
        List of movie IDs, which is sorted and directly matching
        to a one hot encoding of all movies

    Returns
    -------
    List[Tuple[str, float]]
        Returns a list of tuples, containing as first element the
        probabilities of each movie to be in the context of
        another movie and the second entry contains the
        corresponding movie ID.
    """

    movie_probs_with_id = []

    for x_i, movie_id in zip(one_hot_encoded_vector, all_movie_ids_sorted_set):
        if 0 < x_i:
            movie_probs_with_id.append((x_i, movie_id))

    return list(sorted(movie_probs_with_id, key=lambda x: -x[0]))  # Sort by prediction value


if __name__ == "__main__":
    all_movies = Movies().get_all()
    user_movie_histories = load_object_from_file(
        # vars.user_history_file_path_with_real_genres  # TMDB data
        # vars.user_history_file_path_with_real_genres_and_reduced_dimensions  # TMDB data dimension reduced
        vars.user_watchings_file_path_with_real_genres  # Netflix prize data
        # vars.user_watchings_file_path_with_real_genres_small  # Netflix prize data (small part)
    )
    all_movie_ids_sorted_set = sorted(
        set([movie_id for user_movies in user_movie_histories.values() for movie_id, _ in user_movies])
    )

    # Test all input/output and embedding possibilities
    for embedding in ALL_EMBEDDINGS:  # Test all different embeddings
        for encoding_type in [False, True]:  # Test one hot encoded input features and not encoded ones
            emedding_type = embedding["embedding"] if "embedding" in embedding else "no-embedding"
            print("============ {}-{} ============".format(emedding_type, str(encoding_type)))
            print("------------ Feature extraction ------------")

            # Define paths to store files
            extracted_features_file_path = EXTRACTED_FEATURES_BASE_PATH / "extracted_features_{}_{}.pickle".format(
                emedding_type, str(encoding_type)
            )
            results_path = MODEL_BASE_PATH / "{}_{}".format(emedding_type, str(encoding_type))

            # Check, if paths are existing, if not create them
            if not os.path.exists(extracted_features_file_path.parents[0]):
                os.makedirs(extracted_features_file_path.parents[0])

            if not os.path.exists(results_path):
                os.makedirs(results_path)

            # Load or extract data from file and save extarcted data to file
            if os.path.exists(extracted_features_file_path):  # Load already extracted features from file
                total_number_of_movies, extracted_features, embedding_matrix = load_object_from_file(
                    extracted_features_file_path
                )
            else:  # Load preprocessed data from file and extract it
                # Extarct features
                total_number_of_movies, extracted_features, embedding_matrix = extract_features(
                    user_movie_histories=user_movie_histories,
                    window_size=WINDOW_SIZE,
                    input_features_one_hot_encoded=encoding_type,
                    embedding=embedding,
                    min_movie_occurence=MIN_MOVIE_OCCURENCE,
                )

                # Save extracted features
                save_object_in_file(
                    extracted_features_file_path, (total_number_of_movies, extracted_features, embedding_matrix)
                )

            # Use only an handable amount of data
            random.shuffle(extracted_features)  # Shuffle data before using a handable part
            extracted_features = extracted_features[:MAX_DATA]

            # Output results of extracting
            print(f"Extracted {len(extracted_features)} features")

            if embedding_matrix is not None:
                print(f"Embedding matrix has shape {embedding_matrix.shape} with following format:")
                print(embedding_matrix[0])
            print()

            # Prepare data: negative sampling + split train and test data
            X_train, X_test, y_train, y_test = prepare_data(
                extracted_features=extracted_features, train_data_relationship=TRAIN_DATA_RELATIONSHIP
            )

            print("\n------------ Build, train and test model ------------")
            # Build model
            model = build_model(
                total_number_of_movies=total_number_of_movies,
                embedding_matrix=embedding_matrix,
                input_dim=X_train[0].shape[0],
                input_length=X_train[0].shape[0],
            )

            # Train model
            model_path = results_path / "model.keras"

            if os.path.exists(model_path):  # Load model from file
                model = tf.keras.models.load_model(model_path)
            else:  # Train model
                history = model.fit(
                    X_train,
                    y_train,
                    validation_split=VALIDATION_DATA_RELATIONSHIP / TRAIN_DATA_RELATIONSHIP,
                    epochs=EPOCHS,
                    # steps_per_epoch=steps_per_epoch,  # Automatically calculated with: len(training_data) // batch_size
                    batch_size=BATCH_SIZE,
                    callbacks=[
                        EarlyStopping(monitor="val_loss", patience=10, min_delta=0.01, restore_best_weights=True),
                        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
                    ],
                )
                model.save(model_path)

                # Output structure of model
                print(model.summary())

                # Plot accuracy and save it to file
                plt.plot(history.history["accuracy"], label="accuracy")
                plt.legend()
                # plt.savefig(save_dir / "accuracy.png", bbox_inches="tight")
                plt.show()

                # Plot loss and save it to file
                plt.plot(history.history["loss"], label="loss")
                plt.legend()
                # plt.savefig(save_dir / "loss.png", bbox_inches="tight")
                plt.show()

            # Test and evaluate model
            model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
            predictions = model.predict(X_test, batch_size=BATCH_SIZE)

            # Check quality of data by comparing genres of targets
            predictions_with_matching_predefined_genres = 0
            predictions_with_matching_real_genres = 0
            number_of_possible_matchings = (
                0  # Count number of predicted movies with probability > 0 (used for normalizing = computing accuracy)
            )
            epsilon_real_genres = np.sqrt(
                EPSILON_REAL_GENRE**2 * 19
            )  # Difference between the real genres of two movies (euclidean distance)
            average_real_genres_distance = 0

            for prediction, target_movie_one_hot_encoded in zip(predictions, y_test):
                # Find target movie information
                _, target_movie_id = find_movies_of_one_hot(
                    one_hot_encoded_vector=target_movie_one_hot_encoded, all_movie_ids_sorted_set=all_movie_ids_sorted_set
                )[0]
                target_movie_real_genres = np.array(all_movies[target_movie_id]["real_genres"], dtype=np.float64)
                target_movie_predefined_genres = all_movies[target_movie_id]["genres"]  # Predefined genres

                # Find predicted movies information
                predicted_movies = find_movies_of_one_hot(
                    one_hot_encoded_vector=prediction, all_movie_ids_sorted_set=all_movie_ids_sorted_set
                )

                for _, predicted_movie_id in predicted_movies:
                    # Find movie information
                    movie_real_genres = np.array(all_movies[predicted_movie_id]["real_genres"], dtype=np.float64)
                    movie_predefined_genres = all_movies[predicted_movie_id]["genres"]  # Predefined genres

                    # Compute euclidean distance of real genres
                    real_genres_distance = np.linalg.norm(movie_real_genres - target_movie_real_genres)
                    average_real_genres_distance += real_genres_distance

                    # Compare real genres
                    if real_genres_distance <= epsilon_real_genres:
                        predictions_with_matching_real_genres += 1

                    # Compare predefined genres
                    if target_movie_predefined_genres == movie_predefined_genres:
                        predictions_with_matching_predefined_genres += 1

                    number_of_possible_matchings += 1  #

            # Compute accuracy for predefined genres and for real genres
            accuracy_predefined_genres = predictions_with_matching_predefined_genres / number_of_possible_matchings
            accuracy_real_genres = predictions_with_matching_real_genres / number_of_possible_matchings
            average_real_genres_distance /= number_of_possible_matchings

            with open(results_path / "accuracy.txt", "w") as file:
                print(f"Accuracy predefined genres: {accuracy_predefined_genres}", file=file)
                print(f"Accuracy real genres (allowed deviation: {epsilon_real_genres}): {accuracy_real_genres}", file=file)
                print(f"Average real genre distance per prediction: {average_real_genres_distance}", file=file)

            # Output some sample predictions
            output_factor = 1

            with open(results_path / "sample_prediction_outputs.txt", "w") as file:
                for i in range(5):
                    distance = np.linalg.norm(y_test[i] * output_factor - predictions[i] * output_factor)
                    input_movie = find_movies_of_one_hot(
                        one_hot_encoded_vector=X_test[i], all_movie_ids_sorted_set=all_movie_ids_sorted_set
                    )[0][1]
                    output_movie = find_movies_of_one_hot(
                        one_hot_encoded_vector=y_test[i], all_movie_ids_sorted_set=all_movie_ids_sorted_set
                    )[0][1]

                    if distance <= output_factor:
                        print("Correct predicted:", file=file)
                    else:
                        print("False predicted:", file=file)

                    print(f"{input_movie} -> {output_movie}", file=file)
                    print(f"Overall probability: {distance}", file=file)
                    print("Exact movie distances:", file=file)
                    movie_distances = find_movies_of_one_hot(
                        predictions[i], all_movie_ids_sorted_set=all_movie_ids_sorted_set
                    )
                    print(movie_distances, "\n", file=file)

                print(
                    f"\nSame values from all {len(predictions)} entries (only one value of prediction array must differ):",
                    file=file,
                )
                print("0.1\t0.2\t0.3", file=file)
                print(
                    len([1 for pred in zip(predictions) if np.allclose(predictions[0], pred, atol=0.1)]),
                    len(predictions),
                    end="\t",
                    file=file,
                )
                print(
                    len([1 for pred in zip(predictions) if np.allclose(predictions[0], pred, atol=0.2)]),
                    len(predictions),
                    end="\t",
                    file=file,
                )
                print(
                    len([1 for pred in zip(predictions) if np.allclose(predictions[0], pred, atol=0.3)]),
                    len(predictions),
                    end="\t",
                    file=file,
                )

            # Separate output of current iteration from next iteration
            print("\n\n")
