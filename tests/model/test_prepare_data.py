import copy as cp
import numpy as np
import pandas as pd
import sys
import unittest

from pathlib import Path
from parameterized import parameterized
from typing import Any, Dict, List, Tuple

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[2]
sys.path.append(str(project_dir))

import tests.variables as tvars

from model.prepare_data import (
    find_real_genres_to_a_movie,
    find_real_genres_to_all_user_movies,
    reduce_dimensions_on_user_histories_visualization,
    groupd_movie_histories_by_user,
)


# Define constants
SEED = 1234

# Set seeds
np.random.seed(seed=SEED)

# Define names of all genres
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


class TestPrepareData(unittest.TestCase):

    def setUp(self):
        global genre_names

        self.__all_movies = tvars.all_movies
        self.__genre_names = genre_names

    # ------------ Test function "find_real_genres_to_a_movie" ------------
    @parameterized.expand(
        [[i, movie, tvars.all_movies, i, movie["real_genres"]] for i, movie in enumerate(tvars.all_movies.values())]
    )
    def test_find_real_genres_to_a_movie_1(
        self,
        user_id: int,
        movie: Dict[str, Any],
        all_movies: Dict[int, Dict[str, Any]],
        expected_user_id: int,
        expected_real_genres: List[float],
    ) -> None:
        """
        Tests correct cases.

        Parameters
        ----------
        user_id : int
            ID of user, who has watched the movie.
        movie : Dict[str, Any]
            Movie to which a movie will be added
        all_movies : Dict[int, Dict[str, Any]]
            List of all movies in which the real genres of the movie will be
            searched
        expected_user_id : int
            ID of user, who has watched the movie
        expected_real_genres : List[float]
            Expected value for testing function
        """

        # Execute function to test
        user_id, real_genres = find_real_genres_to_a_movie(user_id=user_id, movie=movie, all_movies=all_movies)
        expected_real_genres = np.array(expected_real_genres, dtype=np.float64)

        # Assert/Check results
        self.assertEqual(user_id, expected_user_id)
        np.testing.assert_almost_equal(real_genres, expected_real_genres, decimal=3)

    @parameterized.expand([[i, movie, tvars.all_movies, i, None] for i, movie in enumerate(tvars.all_movies.values())])
    def test_find_real_genres_to_a_movie_2(
        self,
        user_id: int,
        movie: Dict[str, Any],
        all_movies: Dict[int, Dict[str, Any]],
        expected_user_id: int,
        expected_real_genres: List[float],
    ) -> None:
        """
        Tests, if real genres are all zero, so the user ID and None will be
        returned.

        Parameters
        ----------
        user_id : int
            ID of user, who has watched the movie.
        movie : Dict[str, Any]
            Movie to which a movie will be added
        all_movies : Dict[int, Dict[str, Any]]
            List of all movies in which the real genres of the movie will be
            searched
        expected_user_id : int
            ID of user, who has watched the movie
        expected_real_genres : List[float]
            Expected value for testing function
        """

        # Reset genres of all movies, so that no genres will be used and None returned
        all_movies_with_zero_genres = cp.deepcopy(all_movies)

        for movie_id in all_movies_with_zero_genres:
            all_movies_with_zero_genres[movie_id]["real_genres"] = np.zeros(len(movie["real_genres"]))

        # Execute function to test
        user_id, real_genres = find_real_genres_to_a_movie(
            user_id=user_id, movie=movie, all_movies=all_movies_with_zero_genres
        )

        # Assert/Check results
        self.assertEqual(user_id, expected_user_id)
        self.assertEqual(real_genres, expected_real_genres)

    def test_find_real_genres_to_a_movie_3(self) -> None:
        """
        Tests, if a movie was not found in all movies.
        """

        with self.assertRaises(KeyError):
            find_real_genres_to_a_movie(user_id=0, movie={}, all_movies=self.__all_movies)

    # ------------ Test function "find_real_genres_to_all_user_movies" ------------
    @parameterized.expand(
        [
            # One user watched many movies
            [
                tvars.all_movies,
                tvars.all_movies_real_genres_per_user_one_user_unraveled,
                tvars.all_movies_real_genres_per_user_one_user,
                tvars.all_movies_real_genres_per_user_one_user_raveled,
            ],
            # Many users watched exactly one movie
            [
                tvars.all_movies,
                tvars.all_movies_real_genres_per_user_many_users_one_movie_unraveled,
                tvars.all_movies_real_genres_per_user_many_users_one_movie,
                tvars.all_movies_real_genres_per_user_many_users_one_movie_raveled,
            ],
            # Each user watched 10 movies
            [
                tvars.all_movies,
                tvars.all_movies_real_genres_per_user_many_users_equal_movies_unraveled,
                tvars.all_movies_real_genres_per_user_many_users_equal_movies,
                tvars.all_movies_real_genres_per_user_many_users_equal_movies_raveled,
            ],
        ]
    )
    def test_find_real_genres_to_all_user_movies_1(
        self,
        all_movies: Dict[int, Dict[str, Any]],
        all_user_reviews_unraveled: List[Tuple[int, Dict[str, Any]]],
        expected_real_genres_per_user: Dict[int, List[np.ndarray]],
        expected_real_genres_per_user_df: pd.DataFrame,
    ) -> None:
        """
        Tests correct cases.

        all_movies : Dict[int, Dict[str, Any]]
            Dict with all movies, movie ID as key and movie properies as values in another dict
        all_user_reviews_unraveled : Dict[int, List[Dict[str, Any]]]
            Dict with all movies a user have watched with username as key and lists of movies as values
        expected_real_genres_per_user : Dict[int, List[np.ndarray]]
            Expected real genres of each movie per user/users movie history
        expected_real_genres_per_user_df : pd.DataFrame
            Expected real genres of each movie per user/users movie history as DataFrame
        """

        # Execute function to assert/test
        real_genres_per_user, df_real_genres_per_user = find_real_genres_to_all_user_movies(
            all_movies=all_movies,
            all_user_reviews_unraveled=all_user_reviews_unraveled,
            genre_names=self.__genre_names,
            cpu_kernels=1,
            output_iterations=None,
        )

        # Check results of real genres (no DataFrame)
        self.assertEqual(len(real_genres_per_user), len(expected_real_genres_per_user))

        for (user, movies), (expected_user, expected_movies) in zip(
            real_genres_per_user.items(), expected_real_genres_per_user.items()
        ):  # Iterate over all users
            self.assertEqual(user, expected_user)
            self.assertEqual(len(movies), len(expected_movies))

            for real_genres, expected_real_genres in zip(movies, expected_movies):  # Iterate over all movies of a user
                np.testing.assert_almost_equal(real_genres, expected_real_genres, decimal=3)

        # Check results of real genres (DataFrame)
        self.assertEqual(df_real_genres_per_user.shape, expected_real_genres_per_user_df.shape)

        # Iterate over all real genres of all movies
        for (_, row), (_, expected_row) in zip(
            df_real_genres_per_user.iterrows(), expected_real_genres_per_user_df.iterrows()
        ):  # Iterate over all movies
            self.assertEqual(row.values[-1], expected_row.values[-1])  # Compare usernames
            np.testing.assert_almost_equal(row.values[:-1], expected_row.values[:-1], decimal=3)  # Compare all genres

    # ------------ Test function "reduce_dimensions_on_user_histories_visualization" ------------
    @parameterized.expand(
        [
            # Test same data but different target dimensions
            [2, 19, 1, pd.DataFrame({"dim0": [-6974.54099403, 6974.54099403], "username": ["test user"] * 2})],
            [
                2,
                19,
                2,
                pd.DataFrame(
                    {
                        "dim0": [4233.793, -4233.793],
                        "dim1": [5542.487, -5542.487],
                        "username": ["test user"] * 2,
                    }
                ),
            ],
            [
                2,
                19,
                3,
                pd.DataFrame(
                    {
                        "dim0": [-2485.503, 2485.503],
                        "dim1": [5149.832, -5149.832],
                        "dim2": [-3993.076, 3993.076],
                        "username": ["test user"] * 2,
                    }
                ),
            ],
            # Test different data with target dimensions
            [
                5,
                19,
                1,
                pd.DataFrame(
                    {
                        "dim0": [
                            1498.946113677773,
                            -349.532511454175,
                            -3947.018506729204,
                            675.1199916411869,
                            2122.484912864419,
                        ],
                        "username": ["test user"] * 5,
                    }
                ),
            ],
            [4, 2, 1, pd.DataFrame({"dim0": [2704.678, -2703.777, -3994.953, 3994.052], "username": ["test user"] * 4})],
        ]
    )
    def test_reduce_dimensions_on_user_histories_visualization_1(
        self,
        number_of_rows_in_df: pd.DataFrame,
        original_dimension: int,
        target_dimensions: int,
        expected_dimension_reduced_data: pd.DataFrame,
    ) -> None:
        """
        Test reducing dimensions with n dimension. Tests correct cases.\n
        This tets funtion generates input data for function
        "reduce_dimensions_on_user_histories_visualization" with numpy.random.
        For this the global seed "SEED" will be used.

        Parameters
        ----------
        number_of_rows_in_df : int
            Number of rows in DataFrame that will be created for testing
            reducing dimensions on this DataFrame
        original_dimension : int
            Original dimensions of passed DataFrame
        target_dimensions : int
            Target dimensions, DataFrame/genres will be reduced to n_components dimensions
        """

        # Create test data as input for functio to test
        df_user_movie_histories = pd.DataFrame(
            dict(
                [(f"dim{i}", list(np.random.rand(number_of_rows_in_df))) for i in range(original_dimension)]
                + [("username", ["test user"] * number_of_rows_in_df)]
            )
        )

        # Execute function to assert/test
        result = reduce_dimensions_on_user_histories_visualization(
            df_user_movie_histories=df_user_movie_histories, n_dimensions=target_dimensions, cpu_kernels=1
        )

        # Check results
        # self.assertEqual(result, expected_dimension_reduced_data)
        # Compare "username" columns
        np.testing.assert_equal(result.values[:, -1], expected_dimension_reduced_data.values[:, -1])

        # Compare all columns being not the "username" column
        np.testing.assert_almost_equal(result.values[2:, :-1], expected_dimension_reduced_data.values[2:, :-1], decimal=3)

    # ------------ Test function "groupd_movie_histories_by_user" ------------
    @parameterized.expand(
        [
            # Test different dimension reduced data (different output dimensions)
            [
                {0: []},  # Not important contents, only the username/-ID is important
                pd.DataFrame({"dim0": [1, 2, 3], "username": [0, 0, 0]}),
                {
                    0: [
                        [1],
                        [2],
                        [3],
                    ]
                },
            ],
            [
                {0: []},  # Not important contents, only the username/-ID is important
                pd.DataFrame({"dim0": [1, 2, 3], "dim1": [4, 5, 6], "username": [0, 0, 0]}),
                {0: [[1, 4], [2, 5], [3, 6]]},
            ],
            [
                {0: []},  # Not important contents, only the username/-ID is important
                pd.DataFrame({"dim0": [1, 2, 3], "dim1": [4, 5, 6], "dim2": [7, 8, 9], "username": [0, 0, 0]}),
                {0: [[1, 4, 7], [2, 5, 8], [3, 6, 9]]},
            ],
            # Test different users, who watched movies
            [
                {0: [], 1: [], 2: []},  # Not important contents, only the username/-ID is important
                pd.DataFrame({"dim0": [1, 2, 3], "username": [0, 1, 2]}),
                {
                    0: [[1]],
                    1: [[2]],
                    2: [[3]],
                },
            ],
            [
                {0: [], 1: [], 2: []},  # Not important contents, only the username/-ID is important
                pd.DataFrame({"dim0": [1, 2, 3], "dim1": [4, 5, 6], "dim2": [7, 8, 9], "username": [0, 1, 2]}),
                {0: [[1, 4, 7]], 1: [[2, 5, 8]], 2: [[3, 6, 9]]},
            ],
            # Test some users are missing in first dict ("user_movie_histories") with users and what they watched
            [
                {0: []},  # Not important contents, only the username/-ID is important
                pd.DataFrame({"dim0": [1, 2, 3], "username": [0, 1, 2]}),
                {
                    0: [[1]],  # Less users with their movies/real genres in the result, because they are unknown
                },
            ],
            # Test inserting to each user movies with real genres into first dict ("user_movie_histories")
            # -> This extra input won't have some effects to the result
            [
                {
                    0: [
                        [100, 25, 73],
                        [64, 55, 26],
                        [37, 28, 89],
                    ]  # Insert some movies, with real genres -> no change in result
                },
                pd.DataFrame({"dim0": [1, 2, 3], "username": [0, 0, 0]}),
                {
                    0: [
                        [1],
                        [2],
                        [3],
                    ]
                },
            ],
        ]
    )
    def test_groupd_movie_histories_by_user_1(
        self,
        user_movie_histories: Dict[int, List[np.ndarray]],
        df_user_movie_histories_reduced_dim: pd.DataFrame,
        expected_user_movie_histories_reduced_dim: Dict[int, np.ndarray],
    ) -> None:
        """
        Tests correct cases.

        Parameters
        ----------
        user_movie_histories : Dict[int, List[np.ndarray]]
            User movie histories with real genres (computed with e.g. "find_real_genres_to_all_user_movies").
            It's necessary for sorting the result same as user_movie_histories.
        df_user_movie_histories_reduced_dim : pd.DataFrame
            Contains user movie histories with reduced dimensions as visualized DataFrame
        expected_user_movie_histories_reduced_dim : Dict[int, np.ndarray]
            Expected grouped user movie histories per user with the corresponding movies
        """

        # Execute function to assert/test
        user_movie_histories_reduced_dim = groupd_movie_histories_by_user(
            user_movie_histories=user_movie_histories,
            df_user_movie_histories_reduced_dim=df_user_movie_histories_reduced_dim,
        )

        # Check results
        self.assertEqual(len(user_movie_histories_reduced_dim), len(expected_user_movie_histories_reduced_dim))

        for (user_id, movies), (expected_user_id, expected_movies) in zip(
            user_movie_histories_reduced_dim.items(), expected_user_movie_histories_reduced_dim.items()
        ):
            self.assertEqual(user_id, expected_user_id)

            for real_genres, expected_real_genres in zip(movies, expected_movies):
                np.testing.assert_almost_equal(real_genres, expected_real_genres)
