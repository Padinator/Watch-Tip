import re
import requests
import sys
import time

from pathlib import Path
from typing import Any, Dict

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[1]
sys.path.append(str(project_dir))

import helper.variables as vars


def request_url(url: str, headers: Dict[str, str]=vars.headers, max_retries:int = 10, connection_error_timeout: int=10) -> Dict[Any, Any]:
    """
        Requests the passed URL and returns the response as json = dict,
        if the request was successsful, else return en empty dict '{}'
    """

    for i in range(max_retries):
        try:
            response = requests.get(url=url, headers=headers)

            if response.status_code == 200:  # Request was okay
                return response.json()
            elif response.status_code == 404:  # Ressource not found/existing
                break
            elif response.status_code == 429:  # Too many requests
                raise Exception("Too many requests, try again")
        except Exception as ex:  # Too many requests or connection ended by server without response
            print(ex, "sleep(10)")
            time.sleep(connection_error_timeout)
    return {}  # Error case: return an empty dict


def request_movie(url: str, movie_id: str, headers: Dict[str, str]=vars.headers) -> Dict[str, Any]:
    """
        Request API with passed URL "url" and passed ID "movie_id". Use for
        this request the passed headers "headers". Return the resulting data
        as dict.
    """

    # Request API
    movie = request_url(url, headers)

    # Only keep necessary keys
    if movie == {}:  # Skip not existing movie
        raise Exception(f"Movie {movie_id} does not exist in database!")  # Terminate thread (parallel execution with multiple threads)

    # Remove duplicate infos from movie
    movie["credits"]["cast"] = [person["id"] for person in movie["credits"]["cast"]]
    movie["credits"]["crew"] = dict([(str(person["id"]), {"department": person["department"], "job": person["job"]})
                                for person in movie["credits"]["crew"]])
    movie["production_companies"] = [company["id"] for company in movie["production_companies"]]

    return movie


def request_movie_reviews(url: str, movie_id: int, page_number: int) -> Dict[str, Any]:
    """
        Requests the passed URL for movie reviews of a movie.
        Save results in a dict, request other pages, if they exist,
        else return a dictionary containg all reviews of a movie.
    """

    page_specific_url = re.sub("page_number", f"{page_number}", url)
    requested_data = request_url(page_specific_url, vars.headers)

    if requested_data == {}:
        return {}
    else:
        # Save only important parts of recommendations (rc)
        movie_reviews = dict([(rc["author_details"]["username"], {"movie_id": int(movie_id),
            "rating": rc["author_details"]["rating"], "id": rc["id"], "content": rc["content"],
            "created_at": rc["created_at"], "updated_at": rc["updated_at"], "url": rc["url"]})
            for rc in requested_data["results"]])
        total_pages = int(requested_data["total_pages"])

        if page_number + 1 <= total_pages:
            further_reviews = request_movie_reviews(url, movie_id, page_number + 1)
            movie_reviews.update(further_reviews)

        return movie_reviews
