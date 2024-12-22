import unittest
import sys

from pathlib import Path
from parameterized import parameterized
from typing import Any, Dict, List, Tuple

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[2]
sys.path.append(str(project_dir))

from data_preprocessing.insert_user_histories_in_db import *


# Define constants
# Define raw lists of movies
movies_from_database = [
    {"id": 0, "original_title": "Lord of the rings 1", "title": "The lord of the rings 1", "release_year": 2001},
    {"id": 1, "original_title": "Lord of the rings 2", "title": "The lord of the rings 1", "release_year": 2005},
    {"id": 2, "original_title": "Lord of the rings 3", "title": "The lord of the rings 1", "release_year": 2010},
    {"id": 3, "original_title": "Forrest Gump", "title": "Forrest Gump", "release_year": 2000}
]
netflix_movies = [
    {"netflix_id": 1000, "title": "The lord of the rings 1", "year": 2001},
    {"netflix_id": 1001, "title": "The lord of the rings II", "year": 2006},
    {"netflix_id": 1002, "title": "The lord of the rings 3", "year": 1979},
    {"netflix_id": 1003, "title": "Edge of tomorrow", "year": 2015}
]

# Define grouped dicts of movies
true_grouped_movies_from_database = {
    "The": movies_from_database[:-1],
    "Forrest": [movies_from_database[-1]]
}  # Grouped with [True]
true_true_grouped_movies_from_database = {
    "Thelord": movies_from_database[:-1],
    "ForrestGump": [movies_from_database[-1]]
}  # Grouped with [True, True]
true_false_grouped_movies_from_database = {
    "Thel": movies_from_database[:-1],
    "ForrestG": [movies_from_database[-1]]
}  # Grouped with [True, False]
true_grouped_netflix_movies = {
    "The": netflix_movies[:-1],
    "Edge": [netflix_movies[-1]]
}  # Grouped with [True]
true_true_grouped_netflix_movies = {
    "Thelord": netflix_movies[:-1],
    "Edgeof": [netflix_movies[-1]]
}  # Grouped with [True, True]
true_false_grouped_netflix_movies = {
    "Thel": netflix_movies[:-1],
    "Edgeo": [netflix_movies[-1]]
}  # Grouped with [True, False]


class TestInsertUserHistories(unittest.TestCase):

    def setUp(self):
        pass

    # ------------ Test function "format_int" ------------
    @parameterized.expand([
        ["4", -1, 4],
        ["9", -1, 9],
        ["539428", -1, 539428],
        ["34234428", -1, 34234428],
        ["-2", -1, -2],
        ["-1", -1, -1],
        ["-2397342", -1, -2397342],
        ["-120382349", -1, -120382349]
    ])
    def test_format_int_1(self, value_to_format: str, default_value: int, expected_value: int) -> None:
        """
        Tests small/large, positive/negative int values.

        Parameters
        ----------
        value_to_format : str
            Args for function to test: format this value to an int
        default_value : int
            Args for function to test: default value, if formatting failed
        expected_value : int
            Expected value for testing function
        """

        result = format_int(value=value_to_format, default=default_value)
        self.assertEqual(result, expected_value)

    @parameterized.expand([
        # Test some texts as str converting into an int
        ["", -1, -1],
        ["()", -1, -1],
        ["1abc2", -1, -1],
        ["Hello", -1, -1],
        ["[1, 2, 3]", -1, -1],
        ["-1 * 3", -1, -1],
        ["xyz", -1, -1],
        ["-!?]23", -1, -1],

        # Test double values in a str converting into an int
        ["4.0", -1, -1],
        ["9.3", -1, -1],
        ["539428.1", -1, -1],
        ["34234428.3", -1, -1],
        ["-2.4", -1, -1],
        ["-1.7", -1, -1],
        ["-2397342.9", -1, -1],
        ["-120382349.3", -1, -1]
    ])
    def test_format_int_2(self, value_to_format: str, default_value: int, expected_value: int) -> None:
        """
        Tests illegal cases/usages.

        Parameters
        ----------
        value_to_format : str
            Args for function to test: format this value to an int
        default_value : int
            Args for function to test: default value, if formatting failed
        expected_value : int
            Expected value for testing function
        """

        result = format_int(value=value_to_format, default=default_value)
        self.assertEqual(result, expected_value)

    @parameterized.expand([
        [5, -1, 5],
        [7, -1, 7],
        [345793, -1, 345793],
        [20357430, -1, 20357430],
        [-3, -1, -3],
        [-6, -1, -6],
        [-9873450, -1, -9873450],
        [-845694578, -1, -845694578],
    ])
    def test_format_int_3(self, value_to_format: int, default_value, expected_value) -> None:
        """
        Tests other cases like passing an int value, which also work and are okay (misusage is allowed).

        Parameters
        ----------
        value_to_format : int
            Args for function to test: format this value to an int
        default_value : int
            Args for function to test: default value, if formatting failed
        expected_value : int
            Expected value for testing function
        """

        result = format_int(value=value_to_format, default=default_value)
        self.assertEqual(result, expected_value)

    @parameterized.expand([
        [4.0, -1, 4],
        [9.3, -1, 9],
        [539428.1, -1, 539428],
        [34234428.3, -1, 34234428],
        [-2.4, -1, -2],
        [-1.7, -1, -1],
        [-2397342.9, -1, -2397342],
        [-120382349.3, -1, -120382349]
    ])
    def test_format_int_4(self, value_to_format: float, default_value: int, expected_value: int) -> None:
        """
        Tests other cases like passing a float value, which also work and are okay (misusage is allowed).

        Parameters
        ----------
        value_to_format : float
            Args for function to test: format this value to an int
        default_value : int
            Args for function to test: default value, if formatting failed
        expected_value : int
            Expected value for testing function
        """

        result = format_int(value=value_to_format, default=default_value)
        self.assertEqual(result, expected_value)

    # ------------ Test function "build_grouping_expression" ------------
    @parameterized.expand([
        # Test valid and senseful arguments
        ["The lord of the rings 1", [True], "The"],
        ["The lord of the rings 1", [False], "T"],
        ["The lord of the rings 1", [True, True], "Thelord"],
        ["The lord of the rings 1", [True, False], "Thel"],
        ["The lord of the rings 1", [False, True], "The"],
        ["The lord of the rings 1", [False, False], "Th"],

        # Test possible separators (not only space)
        ["The!lord,of;the.rings:1", [True, True, True, True, True, True], "Thelordoftherings1"],
        ["The+lord_of*the/rings%1", [True, True, True, True, True, True], "Thelordoftherings1"],
        ["The\"lord_of'the#rings~1", [True, True, True, True, True, True], "Thelordoftherings1"],

        # Test feeble-minded arguments, which are still possible/valid
        ["A", [False, False, False, False, False], "A"],
        ["", [False], ""]
    ])
    def test_build_grouping_expression_1(self, word: str, group_n_critria: List[bool], expected_value: str) -> None:
        """
            Tests valid arguments/tests correct splits.

            Parameters
            ----------
            word : str
                Args for function to test: split this word into several "tokens" = create a grouping
            group_n_critria : List[bool]
                Args for function to test: contains split arguments, True means split by word/number
                and False by letter/number
            expected_value: str
        """

        result = build_grouping_expression(word=word, group_n_critria=group_n_critria)
        self.assertEqual(result, expected_value)

    @parameterized.expand([
        ["A", [], ""]
    ])
    def test_build_grouping_expression_2(self, word: str, group_n_critria: List[bool], expected_value: str) -> None:
        """
            Tests invalid arguments (no grouping/tokenization criteria passed).

            Parameters
            ----------
            word : str
                Args for function to test: split this word into several "tokens" = create a grouping
            group_n_critria : List[bool]
                Args for function to test: contains split arguments, True means split by word/number
                and False by letter/number
            expected_value: str
        """

        with self.assertRaises(AssertionError):
            build_grouping_expression(word=word, group_n_critria=group_n_critria)

    # ------------ Test function "group_movies_by_attr" ------------
    @parameterized.expand([
        # ---- Test movies from database ----
        # Use for attr "title"
        [movies_from_database, "title", [True], true_grouped_movies_from_database],
        [movies_from_database, "title", [False], { "T": movies_from_database[:-1], "F": [movies_from_database[-1]] }],
        [movies_from_database, "title", [True, True], true_true_grouped_movies_from_database],
        [movies_from_database, "title", [True, False], true_false_grouped_movies_from_database],
        [movies_from_database, "title", [False, True], { "The": movies_from_database[:-1], "Forrest": [movies_from_database[-1]] }],
        [movies_from_database, "title", [False, False], { "Th": movies_from_database[:-1], "Fo": [movies_from_database[-1]] }],

        # Use for attr "original title"
        [movies_from_database, "original_title", [True], { "Lord": movies_from_database[:-1], "Forrest": [movies_from_database[-1]] }],
        [movies_from_database, "original_title", [False], { "L": movies_from_database[:-1], "F": [movies_from_database[-1]] }],
        [movies_from_database, "original_title", [True, True], { "Lordof": movies_from_database[:-1], "ForrestGump": [movies_from_database[-1]] }],
        [movies_from_database, "original_title", [True, False], { "Lordo": movies_from_database[:-1], "ForrestG": [movies_from_database[-1]] }],
        [movies_from_database, "original_title", [False, True], { "Lord": movies_from_database[:-1], "Forrest": [movies_from_database[-1]] }],
        [movies_from_database, "original_title", [False, False], { "Lo": movies_from_database[:-1], "Fo": [movies_from_database[-1]] }],

        # ---- Test Netflix movies ----
        [netflix_movies, "title", [True], true_grouped_netflix_movies],
        [netflix_movies, "title", [False], { "T": netflix_movies[:-1], "E": [netflix_movies[-1]] }],
        [netflix_movies, "title", [True, True], true_true_grouped_netflix_movies],
        [netflix_movies, "title", [True, False], true_false_grouped_netflix_movies],
        [netflix_movies, "title", [False, True], { "The": netflix_movies[:-1], "Edge": [netflix_movies[-1]] }],
        [netflix_movies, "title", [False, False], { "Th": netflix_movies[:-1], "Ed": [netflix_movies[-1]] }],

        # Test feeble-minded arguments, which are still possible/valid
        [[], "title", [False, False], {}],
        [[], "original_title", [False, False], {}],
        [[], "abc!#as98", [False, False, True, False, True], {}]
    ])
    def test_group_movies_by_attr_1(self, movies: List[Dict[str, Any]], attr: str, grouping_criteria: List[bool],
                                    expected_tokenization: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Tests grouping/tokenization of movies from database with varying "attr" and
        "grouping_n_criteria".

        Parameters
        ----------
        movies : List[Dict[str, Any]]
            List of movie dicts to group with grouping criteria group_n_critria
        attr : str
            Attribute of a movie dictionary to use for grouping
        grouping_criteria : List[bool]
            Contains criteria for splitting the passed word. True means split by word/number,
            False means split by letter/number and length of this List means number of splits
        expected_tokenization : Dict[str, List[Dict[str, Any]]]
            The grouped dictionary with all str expressions as keys and the corresponding
            list of movies as values
        """

        result = group_movies_by_attr(movies=movies, attr=attr,
                                      group_n_critria=grouping_criteria)
        self.assertEqual(result, expected_tokenization)

    @parameterized.expand([
        [movies_from_database, "title", []],
        [movies_from_database, "original_title", []],
        [netflix_movies, "title", []]
    ])
    def test_group_movies_by_attr_2(self, movies: List[Dict[str, Any]], attr: str, grouping_criteria: List[bool]) -> None:
        """
        Tests invalid arguments (no grouping/tokenization criteria passed).

        Parameters
        ----------
        movies : List[Dict[str, Any]]
            List of movie dicts to group with grouping criteria group_n_critria
        attr : str
            Attribute of a movie dictionary to use for grouping
        grouping_criteria : List[bool]
            Contains criteria for splitting the passed word. True means split by word/number,
            False means split by letter/number and length of this List means number of splits
        expected_tokenization : Dict[str, List[Dict[str, Any]]]
            The grouped dictionary with all str expressions as keys and the corresponding
            list of movies as values
        """

        with self.assertRaises(AssertionError):
            group_movies_by_attr(movies=movies, attr=attr, group_n_critria=grouping_criteria)

    @parameterized.expand([
        ["Abcdef", "Abcdef", 0.85, (True, 1.0)],
        ["Abcdef", "aBcdef", 0.85, (False, 0.667)],
        ["Abcdef", "aBcdef", 0.65, (True, 0.667)],
        ["The lord of the rings 1", "Lord of the rings 1", 0.85, (True, 0.857)],
        ["The lord of the rings 1", "Lord of the rings I", 0.85, (False, 0.81)]
    ])
    # ------------ Test function "compare_strings" ------------
    def test_compare_strings(self, s1: str, s2: str, min_ratio: float, expected_value) -> None:
        """
        Tests valid arguments/comparisons.
        """

        similiarity, ratio = compare_strings(s1=s1, s2=s2, min_ratio=min_ratio)
        self.assertEqual(similiarity, expected_value[0])  # Check similarity
        self.assertAlmostEqual(ratio, expected_value[1], delta=0.1)  # Check ratio

    # ------------ Test function "find_netflix_movie_in_database" ------------
    @parameterized.expand([
        # ---- Test finding movies returns a movie ----
        # Finds a movie regularly (minimal ratio is good + filter by ratio + filter by year difference)
        [netflix_movies[0], true_grouped_movies_from_database["The"], True, 2, 0.9, True, 2, movies_from_database[0]],
        [netflix_movies[1], true_grouped_movies_from_database["The"], True, 2, 0.9, True, 2, movies_from_database[1]],

        # Test, if changing the year or removing "check-year-criterium" leads in passing last test case (but absurd results)
        [netflix_movies[2], true_grouped_movies_from_database["The"], True, 100, 0.9, True, 2, movies_from_database[0]],
        [netflix_movies[2], true_grouped_movies_from_database["The"], False, -1, 0.9, True, 2, movies_from_database[0]],

        # Test, if deactivating the "max-ratio-filter" results in still finding a correct movie based on fulfilling
        # the minmal ratio and using minimal year difference
        [netflix_movies[0], true_grouped_movies_from_database["The"], True, 2, 0.9, False, 2, movies_from_database[0]],

        # ---- Finds similiar movies with too different years returns original Netflix movie ----
        # Test finding movies finds at least one movie with similiar name, but to different release year
        [netflix_movies[2], true_grouped_movies_from_database["The"], True, 2, 0.9, True, 1, netflix_movies[2]],

        # Test finding movies finds a movie based on very low ratio
        [netflix_movies[3], true_grouped_movies_from_database["Forrest"], True, 2, 1e-10, True, 1, netflix_movies[3]],

        # ---- Test finding movies finds no matching movie ----
        # Ratio is too high
        [netflix_movies[1], true_grouped_movies_from_database["The"], True, 2, 0.95, True, 0, netflix_movies[1]],
        [netflix_movies[3], true_grouped_movies_from_database["The"], True, 2, 0.9, True, 0, netflix_movies[3]],
        [netflix_movies[0], true_grouped_movies_from_database["Forrest"], True, 2, 0.9, True, 0, netflix_movies[0]],
        [netflix_movies[1], true_grouped_movies_from_database["Forrest"], True, 2, 0.9, True, 0, netflix_movies[1]],
        [netflix_movies[2], true_grouped_movies_from_database["Forrest"], True, 2, 0.9, True, 0, netflix_movies[2]],
    ])
    def test_find_netflix_movie_in_database(self, netflix_movie: Dict[str, Any],
                                            movies_from_database: Dict[str, List[Dict[str, Any]]],
                                            evaluate_year_diff: bool, max_year_diff: int,
                                            min_ratio: int, use_max_ratio_filter: bool,
                                            expected_status_code: int, expected_movie: Dict[str, Any]) -> None:
        """
        Tests finding a Netflix movie in database movies.
        """

        # Change keys of expected Netflix movie
        if expected_status_code == 2:  # Found Netflix movie in database
            # Add it to dict of expected movie
            expected_movie["netflix_movie_id"] = netflix_movie["netflix_id"]
            expected_movie["netflix_title"] = netflix_movie["title"]
            expected_movie["netflix_year"] = netflix_movie["year"]
            print(expected_movie)

        # Execute function and test results
        status_code, found_movie = find_netflix_movie_in_database(netflix_movie=netflix_movie, all_movies=movies_from_database,
                                                evaluate_year_difference=evaluate_year_diff, max_year_difference=max_year_diff,
                                                min_ratio=min_ratio, use_max_ratio_filter=use_max_ratio_filter)
        self.assertEqual(status_code, expected_status_code)
        self.assertEqual(found_movie, expected_movie)
