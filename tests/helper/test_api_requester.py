import json
import sys
import unittest

from pathlib import Path
from unittest.mock import patch, MagicMock

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[2]
sys.path.append(str(project_dir))

from helper.api_requester import (
    request_url,
    request_movie,
    request_movie_reviews,
)


# Define constants
BASIC_PATH = project_dir / "tests/jsons_files/test_api_requester_jsons"


class TestApiRequester(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment by initializing file paths for various test cases.

        Attributes
        ----------
        __requested_url_with_status_200_path : Path
            Path to the JSON file containing the test data for a URL request with a status code of 200.
        __successful_requested_movie_path : Path
            Path to the JSON file containing the test data for a successfully requested movie.
        __requested_movie_from_single_page : Path
            Path to the JSON file containing the test data for a movie requested from a single page.
        __requested_movie_from_multiple_pages : Path
            Path to the JSON file containing the test data for a movie requested from multiple pages.
        """
        self.__requested_url_with_status_200_path = BASIC_PATH / "test_request_url_status_200.json"
        self.__successful_requested_movie_path = BASIC_PATH / "test_request_movie_successfully.json"
        self.__requested_movie_from_single_page = BASIC_PATH / "test_request_movie_single_page.json"
        self.__requested_movie_from_multiple_pages = BASIC_PATH / "test_request_movie_multiple_page.json"

    @patch("helper.api_requester.requests.get")
    def test_request_url_status_200(self, request) -> None:
        """
        Test the `request_url` method with a status code of 200.

        This test mocks an HTTP response with a status code of 200 and verifies
        that the `request_url` method correctly processes the response and returns
        the expected JSON data.

        Parameters
        ----------
        request : unittest.mock.MagicMock
            A mock object to simulate the HTTP request.

        Raises
        ------
        AssertionError
            If the result does not match the expected JSON data.

        Returns
        ------
        None
        """

        response = MagicMock()
        response.status_code = 200

        with open(self.__requested_url_with_status_200_path) as json_file:
            test_json_file = json.load(json_file)

        response.json.return_value = test_json_file

        request.return_value = response

        result = request_url("https://www.test.com")

        self.assertEqual(result, test_json_file)

    @patch("helper.api_requester.requests.get")
    def test_request_url_status_404(self, request) -> None:
        """
        Test the `request_url` method when the response status code is 404.

        Parameters
        ----------
        request : unittest.mock.MagicMock
            A mock object to simulate the HTTP request.

        Returns
        -------
        None
        """

        response = MagicMock()
        response.status_code = 404

        request.return_value = response

        result = request_url("https://www.test.com")

        self.assertEqual(result, {})

    @patch("helper.api_requester.requests.get")
    def test_request_url_status_429(self, request) -> None:
        """
        Test the method `request_url` with status code 429.

        Parameters
        ----------
        request : unittest.mock.MagicMock
            A mock object to simulate the HTTP request.

        Returns
        -------
        None
        """

        response = MagicMock()
        response.status_code = 429

        request.return_value = response

        result = request_url("https://www.test.com", max_retries=1, connection_error_timeout=1)

        self.assertEqual(result, {})

    @patch("helper.api_requester.request_url")
    def test_request_movie_successfully(self, request) -> None:
        """
        Test the `request_movie` method successfully.

        This test verifies that the 'request_movie' method returns the expected
        movie data when provided with a valid movie ID and a successful API response.

        Parameters
        ----------
        request : unittest.mock.MagicMock
            A mock object to simulate the HTTP request.

        Returns
        -------
        None
        """

        movie_id = 1

        with open(self.__successful_requested_movie_path) as json_file:
            test_json_file = json.load(json_file)

        request.return_value = test_json_file

        result = request_movie("https://www.test.com", movie_id)

        self.assertEqual(result["credits"]["cast"], [1, 2])
        self.assertEqual(
            result["credits"]["crew"],
            {
                "123": {"department": "Directing", "job": "Director"},
                "456": {"department": "Writing", "job": "Writer"},
            },
        )
        self.assertEqual(result["production_companies"], [101, 102])

    @patch("helper.api_requester.request_url")
    def test_request_movie_not_found(self, request) -> None:
        """
        Test the `request_movie` method when no movie is found.

        Parameters
        ----------
        request : unittest.mock.MagicMock
            A mock object to simulate the HTTP request.

        Raises
        ------
        Exception
            If the movie does not exist in the database.

        Asserts
        -------
        AssertionError
            If the exception message does not match the expected message.

        Returns
        -------
        None
        """

        movie_id = 1
        request.return_value = {}

        with self.assertRaises(Exception) as result:
            request_movie("https://www.test.com", movie_id)

        self.assertEqual(
            f"Movie {movie_id} does not exist in database!",
            str(result.exception),
        )

    @patch("helper.api_requester.request_url")
    def test_request_movie_single_page(self, request) -> None:
        """
        Test the `request_movie_reviews` method with a single page of reviews.

        Parameters
        ----------
        request : unittest.mock.MagicMock
            A mock object to simulate the HTTP request.

        Returns
        -------
        None
        """

        with open(self.__requested_movie_from_single_page) as json_file:
            test_json_file = json.load(json_file)

        request.return_value = test_json_file

        movie_reviews = request_movie_reviews("http://test.com/reviews?page=page_number", 123, 1)

        self.assertEqual(len(movie_reviews), 2)
        self.assertEqual(movie_reviews["user1"]["rating"], 8)
        self.assertEqual(
            movie_reviews["user2"]["content"],
            "Good, but could have been better.",
        )

    @patch("helper.api_requester.request_url")
    def test_request_movie_multiple_page(self, request) -> None:
        """
        Test the method `request_movie_reviews` with multiple pages.

        Parameters
        ----------
        request : unittest.mock.Mock
            A mock object to simulate the HTTP request.

        Returns
        -------
        None
        """

        with open(self.__requested_movie_from_multiple_pages) as json_file:
            test_json_file = json.load(json_file)

        request.side_effect = [test_json_file[0], test_json_file[1]]

        movie_reviews = request_movie_reviews("http://test.com/reviews?page=page_number", 123, 1)

        self.assertEqual(len(movie_reviews), 2)
        self.assertIn("user1", movie_reviews)
        self.assertIn("user2", movie_reviews)

    @patch("helper.api_requester.request_url")
    def test_request_movie_no_reviews(self, request) -> None:
        """
        Test the method `request_movie_reviews` function when no reviews are returned.

        Parameters
        ----------
        request : unittest.mock.MagicMock
            A mock object to simulate the HTTP request.

        Returns
        -------
        None
        """

        request.return_value = {}

        movie_reviews = request_movie_reviews("http://test.com/reviews?page=page_number", 123, 1)

        self.assertEqual(movie_reviews, {})


if __name__ == "__main__":
    unittest.main()
