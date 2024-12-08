import unittest

from helper.api_requester import request_url, request_movie, request_movie_reviews
from unittest.mock import patch, MagicMock


class TestApiRequester(unittest.TestCase):

    @patch("helper.api_requester.requests.get")
    def test_request_url_status_200(self, request):
        """Test the method 'request_url' with status code 200"""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"test": "test"}

        request.return_value = response

        result = request_url("https://www.test.com")

        self.assertEqual(result, {"test": "test"})

    @patch("helper.api_requester.requests.get")
    def test_request_url_status_404(self, request):
        """Test the method 'request_url' with status code 404"""
        response = MagicMock()
        response.status_code = 404

        request.return_value = response

        result = request_url("https://www.test.com")

        self.assertEqual(result, {})

    # TODO: When pytest is executed, it hangs on this test here, why???
    # @patch('helper.api_requester.requests.get')
    # def test_request_url_status_429(self, request):
    #     """ Test the method 'request_url' with status code 429 """
    #     response = MagicMock()
    #     response.status_code = 429

    #     request.return_value = response

    #     with self.assertRaises(Exception) as result:
    #         request_url("https://www.test.com")

    #     self.assertEqual("Too many requests, try again", str(result.exception))

    @patch("helper.api_requester.request_url")
    def test_request_movie_successfully(self, request):
        """Test the method 'request_movie' successfully"""
        movie_id = 1
        data = {
            "credits": {
                "cast": [{"id": 1}, {"id": 2}],
                "crew": [
                    {"id": 123, "department": "Directing", "job": "Director"},
                    {"id": 456, "department": "Writing", "job": "Writer"},
                ],
            },
            "production_companies": [{"id": 101}, {"id": 102}],
        }

        request.return_value = data

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
        data = {
            "results": [
                {
                    "author_details": {"username": "user1", "rating": 8},
                    "id": 123,
                    "content": "Great movie!",
                    "created_at": "2024-12-12:12:12",
                    "updated_at": "2024-12-16:12:12",
                    "url": "http://test.com/review/123",
                },
                {
                    "author_details": {"username": "user2", "rating": 7},
                    "id": 456,
                    "content": "Good, but could have been better.",
                    "created_at": "2024-12-13:12:12",
                    "updated_at": "2024-12-17:12:12",
                    "url": "http://test.com/review/456",
                },
            ],
            "total_pages": 1,  # Only one page
        }
        request.return_value = data

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
        data_page_one = {
            "results": [
                {
                    "author_details": {"username": "user1", "rating": 8},
                    "id": 123,
                    "content": "Great movie!",
                    "created_at": "2024-12-12:12:12",
                    "updated_at": "2024-12-16:12:12",
                    "url": "http://test.com/review/123",
                }
            ],
            "total_pages": 2,  # Two pages
        }

        data_page_two = {
            "results": [
                {
                    "author_details": {"username": "user2", "rating": 7},
                    "id": 456,
                    "content": "Good, but could have been better.",
                    "created_at": "2024-12-13:12:12",
                    "updated_at": "2024-12-17:12:12",
                    "url": "http://test.com/review/456",
                }
            ],
            "total_pages": 2,  # Two pages
        }

        request.side_effect = [data_page_one, data_page_two]

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
