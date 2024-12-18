import copy as cp
import pickle
import re
import sys
import time

from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[1]
sys.path.append(str(project_dir))

import helper.variables as vars

from collections import OrderedDict
from database.movie import Movies
from helper.file_system_interaction import load_object_from_file, save_object_in_file


# Define some constants
COUNT_OF_MAX_RUNNING_THREADS = 16
MAX_YEAR_DIFFERENCE = 1


def get_first_letter_or_number_of_a_title(title: str) -> str:
    """
    Returns to a passed title the first letter as uppercase letter or
    if it's a digit find the whole number and return it as string.
    """

    result = re.search("[a-zA-Z0-9]+", title)

    if result != None:
        return result.group()
        # first_char = result.group()

        # if first_char.isalpha():  # Return first char => letter
        #     return first_char.upper()
        # else:  # First letter is a digit -> find whole number
        #     return re.search("[0-9]+", title).group()
    return None


def format_int(value: str, default: int) -> int:
    """
    Formats passed str to int. If it is not possible, return default value.
    """

    try:
        return int(value)
    except:
        return default


def compareStrings(s1: str, s2: str, min_ratio: float=0.8) -> Tuple[bool, float]:
    """
    Returns, if both strings are similar with comparing them like
    Levenshtein distance, but here:\
    2 * <#matching chars> / ([len(s1) + len(s2)]^2)\
    Also returns the ratio, which was computed, so caller can use/proof
    it.
    """

    ratio = SequenceMatcher(None, s1, s2).ratio()
    return min_ratio < ratio, ratio


def find_netflix_movie_in_database(netflix_movie: Dict[str, Any], all_movies: List[Dict[int, Any]],
                                    max_year_difference: int=1, min_ratio:int=0.8) -> Tuple[int, Dict[str, Any]]:
    """
    Compares passed Netlfix movie with movies from database.\
    Returns (0, <passed netflix movie>), if no movie matches (maybe a series given or
    a movie, for which no movies with similiar names are in database.\
    Returns the movie (1, Dict[str, Any]), if one or more movies were found in database,
    which have a similiar names (matching with "min_ratio"), but different release years
    (+- max_year_difference). This also could happen, if the name of a series is very
    likely to a name of a movie or several movies.\
    Returns (2, <passed netflix movie>), if at least Netflix movie matches more or less
    (with "min_ratio") another movie from database. Then the best matching will be cosen.
    """

    id, year, name = netflix_movie["netflix_id"], netflix_movie["year"], netflix_movie["title"]
    movie_in_database = {}

    # Search netflix movie in database
    for movie in all_movies:
        similarity_o_title, ratio_o_title = compareStrings(name, movie["original_title"], min_ratio)
        similarity_title, ratio_title = compareStrings(name, movie["title"], min_ratio)

        if name == movie["original_title"] or name == movie["title"]\
            or similarity_o_title or similarity_title:
            movie_in_database[(movie["id"], max(ratio_o_title, ratio_title))] = cp.copy(movie)

    # Check, if no movie, one movie or multiple movies were found
    if len(movie_in_database) == 0:  # No movie found -> new movie or series from Netflix given
        return (0, netflix_movie)
    elif len(movie_in_database) == 1:  # Only one matching movie found
        found_netflix_movie = list(movie_in_database.values())[0]
    elif 1 < len(movie_in_database):  
        # Filter for movies with highest ratio to title or original_title of the movie in database
        max_ratio = max(ratio for _, ratio in movie_in_database.keys())
        movie_in_database = dict([(id, movie) for (id, ratio), movie in movie_in_database.items() if ratio >= (max_ratio - 1e-3)])

        # Find movie with closest release_date
        found_netflix_movie = min(movie_in_database.items(), key=lambda x: abs(year - x[1]["release_year"]))[1]

        if max_year_difference < abs(year - found_netflix_movie["release_year"]):  # Difference of maximal max_year_difference years is okay
            return (1, netflix_movie)

    # Save nextlfix data in movie and return it
    found_netflix_movie["netflix_movie_id"] = id
    found_netflix_movie["netflix_title"] = name
    found_netflix_movie["netflix_year"] = year
    return (2, found_netflix_movie)


if __name__ == "__main__":
    # Define variables
    all_movies, netflix_movies_and_series = {}, {}
    all_movies_grouped, netflix_movies_and_series_grouped = defaultdict(OrderedDict), defaultdict(OrderedDict)
    found_netflix_movies, netflix_series = [], []
    no_matching_movie_found, missing_movies_in_db = [], []

    # # Connect to database and read all movies
    # all_movies_table = Movies()
    # all_movies = [(get_first_letter_or_number_of_a_title(movie["title"]),
    #                     {
    #                         "original_title": movie["original_title"], "title": movie["title"],
    #                         "release_year": format_int(movie["release_date"].split("-")[0], -100),
    #                         "id": movie_id
    #                     }
    #                     ) for movie_id, movie in list(all_movies_table.get_all().items()) if get_first_letter_or_number_of_a_title(movie["title"][0]) != None]
    # print(f"Found {len(all_movies)} movies in database")

    # # Read and save Netflix movies
    # with open(vars.local_netflix_movies_file_path, "r") as netflix_movies_file:
    #     netflix_movies_and_series = [(get_first_letter_or_number_of_a_title(movie.split(",", 2)[2]),
    #         {
    #             "netflix_id": movie.split(",", 2)[0],
    #             "title": movie.split(",", 2)[2].strip(),
    #             "year": format_int(movie.split(",", 2)[1], -200)
    #         })
    #         for movie in netflix_movies_file.readlines() if get_first_letter_or_number_of_a_title(movie.split(",", 2)[2]) != None]
    #     print(f"Found {len(netflix_movies_and_series)} Netflix movies")

    # # Group all movies from database by letters and numbers
    # for first_letter_or_number, movie in all_movies:
    #     # Find second letter/number
    #     movie_title = movie["title"]
    #     first_letter_or_number = get_first_letter_or_number_of_a_title(movie_title)
    #     letters_to_skip = len(first_letter_or_number)

    #     # Check, if movie is long enough to split it again
    #     if letters_to_skip < len(movie_title):  # Movie title consist of multiple words
    #         sec_letter_or_number = get_first_letter_or_number_of_a_title(movie_title[letters_to_skip + 1:])
    #     else:  # Movie title consists only of one word
    #         sec_letter_or_number = ""

    #     # Create dictionray for new letter/number
    #     if sec_letter_or_number not in all_movies_grouped[first_letter_or_number]:
    #         all_movies_grouped[first_letter_or_number][sec_letter_or_number] = []

    #     # Add movie to dict of all mvoies
    #     all_movies_grouped[first_letter_or_number][sec_letter_or_number].append(movie)
    # print(f"Grouped {sum([sum([len(ms) for ms in d.values()]) for d in all_movies_grouped.values()])} movies from database")

    # for first_letter_or_number, movie in netflix_movies_and_series:
    #     # Find second letter/number
    #     movie_title = movie["title"]
    #     first_letter_or_number = get_first_letter_or_number_of_a_title(movie_title)
    #     letters_to_skip = len(first_letter_or_number)

    #     # Check, if movie/series is long enough to split it again
    #     if letters_to_skip < len(movie_title):  # Movie/series title consist of multiple words
    #         sec_letter_or_number = get_first_letter_or_number_of_a_title(movie_title[letters_to_skip + 1:])
    #     else:  # Movie/series title consists only of one word
    #         sec_letter_or_number = ""

    #     # Create dictionray for new letter/number
    #     if sec_letter_or_number not in netflix_movies_and_series_grouped[first_letter_or_number]:
    #         netflix_movies_and_series_grouped[first_letter_or_number][sec_letter_or_number] = []

    #     # Add movie to dict of all mvoies/series of Netflix
    #     netflix_movies_and_series_grouped[first_letter_or_number][sec_letter_or_number].append(movie)

    # print(f"Grouped {sum([sum([len(ms) for ms in d.values()]) for d in netflix_movies_and_series_grouped.values()])} Netflix movies")

    # # Sort movies (only first order -> for outputting results)
    # all_movies_grouped = dict(sorted(all_movies_grouped.items(), key=lambda x: x[0]))
    # netflix_movies_and_series_grouped = dict(sorted(netflix_movies_and_series_grouped.items(), key=lambda x: x[0]))

    # # Save prepared movies files in file
    # save_object_in_file("updated_data/all_movies_db.pickle", all_movies_grouped)
    # save_object_in_file("updated_data/all_netflix_movies_and_series_db.pickle", netflix_movies_and_series_grouped)


    # Read all movies from file
    all_movies_grouped = load_object_from_file("updated_data/all_movies_db.pickle")
    netflix_movies_and_series_grouped = load_object_from_file("updated_data/all_netflix_movies_and_series_db.pickle")

    # Print results
    print("All movies:", type(all_movies_grouped), len(all_movies_grouped))
    print(len(all_movies_grouped.keys()))

    for first_letter_or_number, sec_letter_dict in list(all_movies_grouped.items())[::len(all_movies_grouped) // 26 + 1]:
        print(f"{first_letter_or_number}: ", end="")
        for sec_letter_or_number, movies in list(sec_letter_dict.items())[::len(sec_letter_dict) // 26 + 1]:
            print(f"{sec_letter_or_number}: {len(movies)}", end=", ")
        print()
    print()

    print("All netflix movies and series:", type(netflix_movies_and_series_grouped), len(netflix_movies_and_series_grouped))
    print(len(netflix_movies_and_series_grouped.keys()))

    for first_letter_or_number, sec_letter_dict in list(netflix_movies_and_series_grouped.items())[::len(netflix_movies_and_series_grouped) // 26 + 1]:
        print(f"{first_letter_or_number}: ", end="")
        for sec_letter_or_number, movies in list(sec_letter_dict.items())[::len(sec_letter_dict) // 26 + 1]:
            print(f"{sec_letter_or_number}: {len(movies)}", end=", ")
        print()
    print()

    # Create arguments for parallelized task
    print("Create arguments for parallelized task")
    i = 0
    args_for_parallel_execution = []

    start_time = time.time()
    for first_letter_or_number, netflix_movies_dict in list(netflix_movies_and_series_grouped.items())[86:]:
        for sec_letter_or_number, netflix_movies in netflix_movies_dict.items():
            if first_letter_or_number in all_movies_grouped and sec_letter_or_number in all_movies_grouped[first_letter_or_number]:  # Only letters/numbers in both movie dicts
                all_movies_of_two_words = all_movies_grouped[first_letter_or_number][sec_letter_or_number]

                for netflix_movie in netflix_movies:
                    if i % 1000 == 0:
                        print(f"Iteraion: {i}")

                    movie_is_a_movie, movie = find_netflix_movie_in_database(netflix_movie, all_movies_of_two_words, min_ratio=0.95)

                    if movie_is_a_movie == 0:  # Movie is very likely a series
                        no_matching_movie_found.append(movie)
                    elif movie_is_a_movie == 1:  # Netflix movie is very likely missing in database
                        missing_movies_in_db.append(movie)
                    else:  # 2: Found Netflix movie in database
                        found_netflix_movies.append(movie)
                        index_of_movie_to_delete = [m["id"] for m in all_movies_of_two_words].index(movie["id"])
                        all_movies_of_two_words.pop(index_of_movie_to_delete)  # Drop movie from list, so that no other movie can match to it

                    i += 1
            else:
                no_matching_movie_found.extend(netflix_movies)
    print(f"Created {len(args_for_parallel_execution)} arguments for parallelized task")
    print(f"Time: {time.time() - start_time}")

    save_object_in_file("updated_data/insert_user_histories_in_db/found_netflix_movies.pickle", found_netflix_movies)
    save_object_in_file("updated_data/insert_user_histories_in_db/missing_movies_in_db.pickle", missing_movies_in_db)
    save_object_in_file("updated_data/insert_user_histories_in_db/no_matching_movie_found.pickle", no_matching_movie_found)


    found_netflix_movies = load_object_from_file("updated_data/insert_user_histories_in_db/found_netflix_movies.pickle")
    missing_movies_in_db = load_object_from_file("updated_data/insert_user_histories_in_db/missing_movies_in_db.pickle")
    no_matching_movie_found = load_object_from_file("updated_data/insert_user_histories_in_db/no_matching_movie_found.pickle")

    print(f"found_netflix_movies: {len(found_netflix_movies)}")
    print(f"missing_movies_in_db: {len(missing_movies_in_db)}")
    print(f"no_matching_movie_found: {len(no_matching_movie_found)}")
    print(f"All in all there are {len(found_netflix_movies) + len(missing_movies_in_db) + len(no_matching_movie_found)} movies")

    # vars.map_for_netflix_movies_to_db_movies
