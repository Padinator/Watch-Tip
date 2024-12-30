import numpy as np
import sys
import unittest

from pathlib import Path
from parameterized import parameterized
from typing import Dict, List, Tuple

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[2]
sys.path.append(str(project_dir))

import tests.variables as tvars

from model.model import extract_features


# Define constants
SEED = 1234

# Set seeds
np.random.seed(seed=SEED)


def generate_full_movies_histories(
    all_movies_real_genres: Dict[int, List[np.ndarray]], history_len: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates to a passed list of movies the movie histories. It's basically a
    part of the function to test.

    Parameters:
    user_movie_histories : Dict[int, List[np.array]]
        Grouped/Unraveled movies histories per user, e.g.:
        {
            1038924: [
                [10.2, 39.3, 59.5, 56.4, 12.8, 0.76, 96.4, 21.3, 69.0, 98.5,
                28.2, 49.1, 19.01, 39.4, 18.9, 38.2, 98.5, 25.6, 9.3],  # Real genres of first movie of user 1038924\n
                ...
            ]
        }
    history_len : int
        Resulting length of each movie history, which represents an input
        feature for an AI model.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        Returns a list with the input features in the first values and the
        targets/labels in the second values.
    """

    return [
        (users_movie_history[i:i + history_len], users_movie_history[i + history_len])
        for users_movie_history in all_movies_real_genres.values()
        for i in range(len(users_movie_history) - history_len)
        if 0 < (len(users_movie_history) - history_len)
    ]


class TestModel(unittest.TestCase):

    def setUp(self):
        pass

    # ------------ Test function "extract_features" ------------
    @parameterized.expand(
        [
            # -------- Test different an amount of users --------
            # Only one user with all movies (fine-grained movie packets)
            [
                tvars.all_movies_real_genres_per_user_one_user,
                10,
                5,
                False,
                True,  # Split movies into fine-grained packets
                generate_full_movies_histories(tvars.all_movies_real_genres_per_user_one_user, 10),
                len(tvars.all_movies_real_genres_per_user_one_user),  # All users
            ],
            # Only one user with all movies (not fine-grained movie packets)
            [
                tvars.all_movies_real_genres_per_user_one_user,
                10,
                5,
                False,
                False,  # Don't split movies into fine-grained packets
                generate_full_movies_histories(tvars.all_movies_real_genres_per_user_one_user, 10)[
                    ::10
                ],  # Same as splitting all movies into packets of 10
                len(tvars.all_movies_real_genres_per_user_one_user),  # All users
            ],
            # All users have too many histories, ignore all users movie histories
            [
                tvars.all_movies_real_genres_per_user_many_users_one_movie,
                10,
                5,
                False,
                True,
                [],
                0,
            ],
            # Each user watched 10 movies
            [
                tvars.all_movies_real_genres_per_user_many_users_equal_movies,
                9,  # 9 movies for the input feature and tenth for the target/label
                5,
                False,
                True,
                generate_full_movies_histories(
                    tvars.all_movies_real_genres_per_user_many_users_equal_movies,
                    9,
                ),
                len(tvars.all_movies_real_genres_per_user_many_users_equal_movies)
                - 1,  # Because one user watched only 5 movies
            ],
            # -------- Test different (not) filling with zero movies (= zero vector) --------
            # Don't fill with zero movies, because it's deactivated although
            # the min history length is lower then the movie history length of
            # the last user (5)
            [
                tvars.all_movies_real_genres_per_user_many_users_equal_movies,
                9,  # 9 movies for the input feature and one for the target/label
                4,  # User with only 5 movies is okay, but would be filled with 6 zero movies
                False,  # No filling with zero movies is possible (same result like in 3. parameterization)
                True,
                generate_full_movies_histories(tvars.all_movies_real_genres_per_user_many_users_equal_movies, 9),
                len(tvars.all_movies_real_genres_per_user_many_users_equal_movies)
                - 1,  # Because one user watched only 5 movies
            ],
            # Fill movie history of last user with zero movies, because he has
            # only 5 movies. So add 5 zero movies (=zero vectors).
            [
                tvars.all_movies_real_genres_per_user_many_users_equal_movies,
                9,  # 9 movies for the input feature and one for the target/label
                4,  # User with only 5 movies is okay, but would be filled with 6 zero movies
                True,  # Adding zero movies is no permitted
                True,
                generate_full_movies_histories(tvars.all_movies_real_genres_per_user_many_users_equal_movies, 9)
                + [
                    (
                        np.concatenate(
                            (
                                np.tile(np.zeros(19), (5, 1)),  # Add 5 zero movies
                                list(tvars.all_movies_real_genres_per_user_many_users_equal_movies.values())[-1][
                                    :4
                                ],  # Use 4 movies from user's movie history
                            ),
                            axis=0,
                        ),
                        list(tvars.all_movies_real_genres_per_user_many_users_equal_movies.values())[-1][
                            4
                        ],  # Use last movie form users movie history as target/label
                    )
                ],
                len(tvars.all_movies_real_genres_per_user_many_users_equal_movies),  # All users
            ],
        ]
    )
    def test_extract_features(
        self,
        user_movie_histories: Dict[int, List],
        movie_history_len: int,
        min_movie_history_len: int,
        fill_history_len_with_zero_movies: bool,
        fine_grained_extracting: bool,
        expected_extracted_features: List[Tuple[np.array, np.array]],
        expected_used_histories: int,
    ) -> None:
        """
        Test correct cases.

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
        expected_features : List[Tuple[np.array, np.array]]
            Tuples consisting of the last seen movies (= input) and the next
            one to predict (= target, label) out ot the previous ones.
        expected_used_histories : int
            Expected number of users histories, which were read and extracted
        """

        # Execute function to assert/test
        used_histories, extracted_features = extract_features(
            user_movie_histories=user_movie_histories,
            movie_history_len=movie_history_len,
            min_movie_history_len=min_movie_history_len,
            fill_history_len_with_zero_movies=fill_history_len_with_zero_movies,
            fine_grained_extracting=fine_grained_extracting,
        )

        # Check results
        self.assertEqual(used_histories, expected_used_histories)  # Compare number of extracted users
        self.assertEqual(len(extracted_features), len(expected_extracted_features))

        for (extracted_feature, target), (expected_extracted_feature, expected_target) in zip(
            extracted_features, expected_extracted_features
        ):
            np.testing.assert_array_almost_equal(target, expected_target)  # Compare targets/labels

            # Compare movie histories
            for real_genres, expected_real_genres in zip(extracted_feature, expected_extracted_feature):
                np.testing.assert_array_almost_equal(real_genres, expected_real_genres)
