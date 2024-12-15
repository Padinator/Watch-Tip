import json
import sys
import unittest

from pathlib import Path
from unittest.mock import patch, MagicMock

# ---------- Import own python files ----------
project_dir = Path(__file__).parents[2]
sys.path.append(str(project_dir))

from helper.api_requester import request_url, request_movie, request_movie_reviews


# Define constants
BASIC_PATH = project_dir / "tests/test_jsons_files/test_api_requester_jsons"


class TestApiRequester(unittest.TestCase):

    def setUp(self):
        self.__requested_url_with_status_200_path = BASIC_PATH / "test_request_url_status_200.json"
        self.__successful_requested_movie_path = BASIC_PATH / "test_request_movie_successfully.json"
        self.__requested_movie_from_single_page = BASIC_PATH / "test_request_movie_single_page.json"
        self.__requested_movie_from_multiple_pages = BASIC_PATH / "test_request_movie_multiple_page.json"

    @patch("helper.api_requester.requests.get")
    def test_request_url_status_200(self, request):
        """Test the method 'request_url' with status code 200"""
        response = MagicMock()
        response.status_code = 200

        with open(self.__requested_url_with_status_200_path) as json_file:
            test_json_file = json.load(json_file)

        response.json.return_value = test_json_file

        request.return_value = response

        result = request_url("https://www.test.com")

        self.assertEqual(result, test_json_file)

    @patch("helper.api_requester.requests.get")
    def test_request_url_status_404(self, request):
        """Test the method 'request_url' with status code 404"""
        response = MagicMock()
        response.status_code = 404

        request.return_value = response

        result = request_url("https://www.test.com")

        self.assertEqual(result, {})

    @patch('helper.api_requester.requests.get')
    def test_request_url_status_429(self, request):
        """ Test the method 'request_url' with status code 429 """
        response = MagicMock()
        response.status_code = 429

        request.return_value = response

        result = request_url("https://www.test.com", max_retries=1, connection_error_timeout=1)

        self.assertEqual(result, {})

    @patch("helper.api_requester.request_url")
    def test_request_movie_successfully(self, request):
        """Test the method 'request_movie' successfully"""
        movie_id = 1

        with open (self.__successful_requested_movie_path) as json_file:
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
    def test_request_movie_not_found(self, request):
        """Test the method 'request_movie' with no movie found"""
        movie_id = 1
        request.return_value = {}

        with self.assertRaises(Exception) as result:
            request_movie("https://www.test.com", movie_id)

        self.assertEqual(
            f"Movie {movie_id} does not exist in database!", str(result.exception)
        )

    @patch("helper.api_requester.request_url")
    def test_request_movie_single_page(self, request):
        """Test the method 'request_movie_reviews' with one single page"""

        with open (self.__requested_movie_from_single_page) as json_file:
            test_json_file = json.load(json_file)

        request.return_value = test_json_file

        movie_reviews = request_movie_reviews(
            "http://test.com/reviews?page=page_number", 123, 1
        )

        self.assertEqual(len(movie_reviews), 2)
        self.assertEqual(movie_reviews["user1"]["rating"], 8)
        self.assertEqual(
            movie_reviews["user2"]["content"], "Good, but could have been better."
        )

    @patch("helper.api_requester.request_url")
    def test_request_movie_multiple_page(self, request):
        """Test the method 'request_movie_reviews' with multiple pages"""

        with open (self.__requested_movie_from_multiple_pages) as json_file:
            test_json_file = json.load(json_file)

        request.side_effect = [test_json_file[0], test_json_file[1]]

        movie_reviews = request_movie_reviews(
            "http://test.com/reviews?page=page_number", 123, 1
        )

        self.assertEqual(len(movie_reviews), 2)
        self.assertIn("user1", movie_reviews)
        self.assertIn("user2", movie_reviews)

    @patch("helper.api_requester.request_url")
    def test_request_movie_no_reviews(self, request):
        """Test the methode 'request_movie_reviews' in the file
        api.requester.py
        """
        request.return_value = {}

        movie_reviews = request_movie_reviews(
            "http://test.com/reviews?page=page_number", 123, 1
        )

        self.assertEqual(movie_reviews, {})


if __name__ == "__main__":
    unittest.main()
