import unittest
import sys

from pathlib import Path
from parameterized import parameterized

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[2]
sys.path.append(str(project_dir))

from data_preprocessing.insert_user_histories_in_db import *


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
    def test_build_grouping_expression_1(self, word: str, group_n_critria: List[bool], expected_value: str):
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
    def test_build_grouping_expression_2(self, word: str, group_n_critria: List[bool], expected_value: str):
        """
            Tests invalid arguments.

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
