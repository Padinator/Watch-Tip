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
MIN_MOVIE_RATIO = 0.9  # Relationship between movies while finding a matching between Netflix movies and movies from database
MAX_YEAR_DIFFERENCE = 2  # Maximal acceptable difference between a Netflix movie and one from database


def format_int(value: str, default: int) -> int:
    """
    Formats passed str to int.
    
    Parameters
    ----------
    value : int
        Value to convert/format to str
    default : int
        Default value, if converting fails

    Returns
    -------
    int
        The formatted/converted value or the passed default value, if conversion failed
    """

    try:
        return int(value)
    except:
        return default


def build_grouping_expression(word: str, group_n_critria: List[bool]=[True]) -> str:
    """
    Builds an expression, which macthes the regex pattern "[a-zA-Z0-9]+" (word or number)
    and "[a-zA-Z]|[0-9]+" (letter or number).\n
    All sub expressions matching the regex patterns will be concatenated. Other characters
    not in the pattern won't be included as well.

    Parameters
    ----------
    word : str
        A word, which will be splitted into multiple words or letters and numbers.
    group_n_critria : List[bool], default [True]
        Contains criteria for splitting the passed word. True means split by word/number,
        False means split by letter/number and length of this List means number of splits
    
    Returns
    -------
    str
        The whole matching string
    """

    assert 0 < len(group_n_critria)

    # Find pattern in passed word
    pattern = "[a-zA-Z0-9]+" if group_n_critria[0] else "[a-zA-Z]|[0-9]+"  # Use word/number or character/number
    res = re.search(pattern, word)
    expression = res.group() if res != None else ""

    if len(group_n_critria) == 1 or len(word) == len(expression):  # No group criteria is remaining or word is already empty/worked off
        return expression
    else:
        occurence_in_word = word.index(expression)  # Skip other characters like "."
        return expression + build_grouping_expression(word[occurence_in_word + len(expression):], group_n_critria[1:])


def group_movies_by_attr(movies: List[Dict[str, Any]], attr: str, group_n_critria: bool=[True]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Groups passed array of movies by passed attribute, e.g. movie titles.
    This will be done n (= length of group_n_critria) times. Grouping will
    be done based on function "build_grouping_expression".

    Parameters
    ----------
    movies : List[Dict[str, Any]]
        List of movie dicts to group with grouping criteria group_n_critria
    attr : str
        Attribute of a movie dictionary to use for grouping
    group_n_critria : bool=[True]
        Contains criteria for splitting the passed word. True means split by word/number,
        False means split by letter/number and length of this List means number of splits

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        The grouped dictionary with all str expressions as keys and the corresponding
        list of movies as values
    """

    grouped_movies = defaultdict(list)

    for movie in movies:
        movie_attr = movie[attr]
        groupding_expression = build_grouping_expression(movie_attr, group_n_critria)
        grouped_movies[groupding_expression].append(movie)

    return grouped_movies


def compareStrings(s1: str, s2: str, min_ratio: float=0.8) -> Tuple[bool, float]:
    """
    Compares two strings and finds relationship as number between them.

    Parameters
    ----------
    s1 : str
        First string to compare with second one
    s2 : str
        Second string to compare with first one
    min_ratio : float,d efault 0.8
        Minimal relationship between movies for declaring them as similiar

    Returns
    -------
    Tuple[bool, float]
        True, if both strings are similar with comparing them like Levenshtein distance, but here:
        2 * <#matching chars> / ([len(s1) + len(s2)]^2)\n
        Also returns the ratio, which was computed, so caller can use/proof it
    """

    ratio = SequenceMatcher(None, s1, s2).ratio()
    return min_ratio < ratio, ratio


def find_netflix_movie_in_database(netflix_movie: Dict[str, Any], all_movies: List[Dict[int, Any]],
                                    evaluate_year_difference: bool=True, max_year_difference: int=1,
                                    min_ratio:int=0.8, use_max_ratio_filter: bool=True)\
                                        -> Tuple[int, Dict[str, Any]]:
    """
    Compares passed Netlfix movie with movies from database by ratio and difference of years.

    Parameters
    ----------
    netflix_movie : Dict[str, Any]
        Netflix movie or series to find a correspsonding movie in database
    all_movies : List[Dict[int, Any]]
        List of all movies in database
    evaluate_year_difference : bool, default True
        If True evaluate maximal difference of movies in criterium year, else skip this filtering
    max_year_difference : int, default 1
        Maximal maximal temproal/year difference between a Netflix movie and a movie from database
    min_ratio : int, default 0.8
        Minimal ratio for declaring a Netflix movie and a movie from database as similiar
    use_max_ratio_filter : bool, default True
        If True, find movie with highest ratio and filter all movies, which have a lower ratio

    Returns
    -------
    Tuple[int, Dict[str, Any]]
        First value is 0, if no matching movie was found in database, 1 if some matching movies
        were found, but they have a too big year difference and 2, if a movie from database 
        matching the Netflix movie was found.\n
        Second value is the movie, in the first two cases it's the original Netflix movie object,
        else the combination of both movie (Netflix movie and movie from database) object
        keys/values in one dict.
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
    else:  
        # Filter for movies with highest ratio to title or original_title of the movie in database
        if use_max_ratio_filter:
            max_ratio = max(ratio for _, ratio in movie_in_database.keys())
            movie_in_database = dict([(id, movie) for (id, ratio), movie in movie_in_database.items() if ratio >= (max_ratio - 1e-3)])

        # Find movie with closest release_date
        found_netflix_movie = min(movie_in_database.items(), key=lambda x: abs(year - x[1]["release_year"]))[1]

        if evaluate_year_difference and (max_year_difference < abs(year - found_netflix_movie["release_year"])):  # Difference of maximal max_year_difference years is okay
            return (1, netflix_movie)

    # Save nextlfix data in movie and return it
    found_netflix_movie["netflix_movie_id"] = id
    found_netflix_movie["netflix_title"] = name
    found_netflix_movie["netflix_year"] = year
    return (2, found_netflix_movie)


def match_netflix_movies_with_movies_from_database(movies_from_database: Dict[str, Dict[str, Any]], netflix_movies: Dict[str, Dict[str, Any]],
                                                   min_ratio: float=0.8, use_max_ratio_filter: bool=True, evaluate_year_difference: bool=True,
                                                   max_year_difference: int=1, output_iterations: bool=True, output_time: bool=True)\
                                                    -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Matches Netflix movies and movies from database. This function calls
    function "find_netflix_movie_in_database".

    Parameters
    ----------
    movies_from_database : Dict[str, Dict[str, Any]]
        Grouped dict of all movies in database, e.g.\n
        {
            "The lord": [<movies starting with "The lord">],
            "The emperor": [<movies starting with "The emperor">],
            ...
        }
    netflix_movies : Dict[str, Dict[str, Any]]
        Grouped dict of all Netflix movies, e.g.\n
        {
            "The lord": [<movies starting with "The lord">],
            "The emperor": [<movies starting with "The emperor">],
            ...
        }
    min_ratio : float, default 0.8
        Minimal ratio for declaring a Netflix movie and a movie from database as similiar
    use_max_ratio_filter : bool, default True
        If True, find movie with highest ratio and filter all movies, which have a lower ratio
    evaluate_year_difference : bool, default True
        If True evaluate maximal difference of movies in criterium year, else skip this filtering
    max_year_difference : int, default 1
        Maximal maximal temproal/year difference between a Netflix movie and a movie from database
    output_iterations : bool, default True
        If True output iterations of matching process, else not
    output_time : bool, default True
        If True output time the matching process took, else not

    Returns
    -------
    Tuple[
                List[Dict[str, Any]],\n
                List[Dict[str, Any]],\n
                List[Dict[str, Any]]
    ]
        Returns three list of movie dicts: found/matching Netflix movies, movies without any matching
        (probably  series) and movies which had a matching, but the temporal difference was to big
        (maybe a prequel/sequel or a series to a movie)
    """

    # Define variables for matching step
    found_netflix_movies, no_matching_movie_found, missing_movies_in_db = [], [], []
    i = 0

    # Match Netflix movies and movies from database
    if output_time:
        start_time = time.time()

    for grouping_expression, netflix_movies in netflix_movies.items():
        if grouping_expression in movies_from_database:  # Grouping expressions must be in both movie dicts
            all_movies_of_two_words = movies_from_database[grouping_expression]

            for netflix_movie in netflix_movies:
                if output_iterations and i % 1000 == 0:
                    print(f"Iteration: {i}")
                i += 1

                movie_is_a_movie, movie = find_netflix_movie_in_database(netflix_movie, all_movies_of_two_words,
                                                    min_ratio=min_ratio, use_max_ratio_filter=use_max_ratio_filter,
                                                    evaluate_year_difference=evaluate_year_difference, max_year_difference=max_year_difference)

                if movie_is_a_movie == 0:  # Movie is very likely a series
                    no_matching_movie_found.append(movie)
                elif movie_is_a_movie == 1:  # Netflix movie is very likely missing in database
                    missing_movies_in_db.append(movie)
                else:  # 2: Found Netflix movie in database
                    found_netflix_movies.append(movie)
                    index_of_movie_to_delete = [m["id"] for m in all_movies_of_two_words].index(movie["id"])
                    all_movies_of_two_words.pop(index_of_movie_to_delete)  # Drop movie from list, so that no other movie can match to it
        else:
            no_matching_movie_found.extend(netflix_movies)

    if output_time:
        print(f"Time: {time.time() - start_time} sec")
    return found_netflix_movies, no_matching_movie_found, missing_movies_in_db


def output_groups(all_movies_grouped: Dict[str, List[Dict[str, Any]]],
                  netflix_movies_and_series_grouped: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Outputs groups of movies.

    Parameters
    ----------
    all_movies_grouped : Dict[str, List[Dict[str, Any]]]
        Grouped dict of all movies from database, e.g.\n
        {
            "The lord": [<movies starting with "The lord">],
            "The emperor": [<movies starting with "The emperor">],
            ...
        }
    netflix_movies_and_series_grouped: Dict[str, List[Dict[str, Any]]]
        Grouped dict of Netflix movies and series, e.g.\n
        {
            "The lord": [<movies starting with "The lord">],
            "The emperor": [<movies starting with "The emperor">],
            ...
        }
    """

    number_of_movies_in_database = sum([len(movies) for movies in all_movies_grouped.values()])
    number_of_netflix_movies = sum([len(movies) for movies in netflix_movies_and_series_grouped.values()])
    print(f"Grouped {number_of_movies_in_database} movies from database into {len(all_movies_grouped.keys())} groups")
    print(f"Grouped {number_of_netflix_movies}  missing in database Netflix movies into {len(netflix_movies_and_series_grouped.keys())} groups")


def output_matching_results(found_netflix_movies: List[Dict[str, Any]], no_matching_movies_found: List[Dict[str, Any]],
                            missing_movies_in_db: List[Dict[str, Any]], number_of_movies_in_database_before: int,
                            number_of_movies_in_database_after: int) -> None:
    """
    Outputs statistics of a matching run.

    Parameters
    ----------
    found_netflix_movies : List[Dict[str, Any]]
        List of all Netflix movies found while matching them to all database movies
    no_matching_movies_found : List[Dict[str, Any]]
        List of all Netflix movies that don't match movies form database
    missing_movies_in_db : List[Dict[str, Any]]
        List of all Netflix movies that are missing in database
    number_of_movies_in_database_before : int,
        Number of movies in database before processing matching
    number_of_movies_in_database_after : int
        Number of movies in database after processing matching
    """

    print(f"found_netflix_movies: {len(found_netflix_movies)}")
    print(f"no_matching_movies_found: {len(no_matching_movies_found)}")
    print(f"missing_movies_in_db: {len(missing_movies_in_db)}")
    print(f"All in all there are {len(found_netflix_movies) + len(missing_movies_in_db) + len(no_matching_movies_found)} movies")
    print(f"Movies in database before: {number_of_movies_in_database_before}, number of movies in database after: {number_of_movies_in_database_after}")
    print(f"Movies removed from database list: {number_of_movies_in_database_before - number_of_movies_in_database_after}")


def do_one_matching_run(grouped_database_movies: Dict[str, List[Dict[str, Any]]], ungrouped_netflix_movies: List[Dict[str, Any]],
                        min_ratio: float, max_year_diff: int, group_criteria: List[bool], database_movies_criterium: str="title",
                        netflix_movie_criterium: str="title")\
                            -> Tuple[Dict[str, List[Dict[str, Any]]],
                                     Tuple[List[Dict[str, Any]],
                                           List[Dict[str, Any]],
                                           List[Dict[str, Any]]]]:
    """
    Do a matching run, which means try to match Netflix movies to movies from database.
    This will be done by calling the function "match_netflix_movies_with_movies_from_database".

    Parameters
    ----------
    grouped_database_movies : Dict[str, List[Dict[str, Any]]]
        Grouped dict of all movies in database, e.g.\n
        {
            "The lord": [<movies starting with "The lord">],
            "The emperor": [<movies starting with "The emperor">],
            ...
        }
    ungrouped_netflix_movies : List[Dict[str, Any]]
        List of all Netflix movies to group and to find a matching with movies from database
    min_ratio : float
        Minimal ratio for declaring a Netflix movie and a movie from database as similiar
    max_year_diff : int
        Maximal maximal temproal/year difference between a Netflix movie and a movie from database
    group_criteria : List[bool]
        Contains criteria for grouping (= splitting) the passed Netflix movie titles. True means
        split by word/number, False means split by letter/number and length of this List means number
        of splits
    database_movies_criterium : str, default "title"
        Use this as entry to find values in movie dicts from database. Based on these values the
        grouping will be done.
    netflix_movie_criterium : str, default "title"
        Use this as entry to find values in Netflix movie dicts. Based on these values the grouping
        will be done.

    Returns
    -------
    Tuple[
                Dict[str, List[Dict[str, Any]]],\n
                Tuple[
                            List[Dict[str, Any]],\n
                            List[Dict[str, Any]],\n
                            List[Dict[str, Any]]
                ]
    ]
        Returns as first value a modified, grouped dict of all movies from database. Movies, which are
        found in database, are here missing/left out.\n
        Second entry is a dict containing three lists: found/matching Netflix movies, movies without
        any matching, movies which had a matching, but the temporal difference
    """

    # Group movies by a word/number and a char/number two words/numbers
    all_remaining_movies = [movie for movies in grouped_database_movies.values() for movie in movies]  # Ravel movie groups
    new_grouped_database_movies = group_movies_by_attr(all_remaining_movies, database_movies_criterium, group_criteria)  # Be more relaxed with word grouping
    new_grouped_netflix_movies = group_movies_by_attr(ungrouped_netflix_movies, netflix_movie_criterium, group_criteria)  # Be more relaxed with word grouping
    output_groups(new_grouped_database_movies, new_grouped_netflix_movies)

    # Find matches
    return new_grouped_database_movies, match_netflix_movies_with_movies_from_database(new_grouped_database_movies,
                                            new_grouped_netflix_movies, min_ratio=min_ratio, max_year_difference=max_year_diff)


def optimize_matchings(database_movies: Dict[str, List[Dict[str, Any]]], netflix_movies_and_series: Dict[str, List[Dict[str, Any]]],
                        ratio_max_year_combis: List[Tuple[float, int]], group_criteria: List[Tuple[bool, bool]])\
                        -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Optimize matchings by doing many iterations of matching Netflix movies with movies from database.
    For each matching iteration the function "do_one_matching_run" will be called.\n
    The first matching iteration will is most restrictive, because of finding first all "100 % safe"
    matches, so the ratio between two movies must be >= 0.99 and the temporal/year difference must
    be 0.\n
    Then do other iterations based on the passed parameters. Typically the restrictions will be
    lowered to find more matches, but it's guaranteed that obvious matches between Netflix movies
    and movies from database will be always correct matched.

    Parameters
    ----------
    database_movies : Dict[str, List[Dict[str, Any]]]
        Grouped dict of all movies in database, e.g.\n
        {
            "The lord": [<movies starting with "The lord">],
            "The emperor": [<movies starting with "The emperor">],
            ...
        }
    netflix_movies_and_series : Dict[str, List[Dict[str, Any]]]
        Grouped dict of Netflix movies and series, e.g.\n
        {
            "The lord": [<movies starting with "The lord">],
            "The emperor": [<movies starting with "The emperor">],
            ...
        }
    ratio_max_year_combis : List[Tuple[float, int]]
        List of all combinations of ratio and may year difference (temporal differene), which will
        be executed/tried in an matching iteration.
    group_criteria: List[Tuple[bool, bool]]
        Execute all "ratio_max_year_combis" with a group criterium (for loop over for loop). A group
        criterium specifies, how too split words (here movie titles) for grouping them.

    Returns
    -------
    Tuple[
                List[Dict[str, Any]],\n
                List[Dict[str, Any]],\n
                List[Dict[str, Any]]
    ]
        Returns three list of movie dicts: found/matching Netflix movies, movies without any matching
        (probably  series) and movies which had a matching, but the temporal difference was to big
        (maybe a prequel/sequel or a series to a movie)
    """

    # Define variables
    found_netflix_movies, no_matching_movies_found, missing_movies_in_db = [], [], []
    database_movies_criterium, netflix_movie_criterium = "title", "title"
    start_group_criterium = [True, True]  # Be most restrictive with splitting on whole words/numbers

    # Group all movies from database in first iteration manually
    all_movies_grouped = group_movies_by_attr(database_movies, database_movies_criterium, start_group_criterium)  # Group manually
    all_movies_grouped = dict(sorted(all_movies_grouped.items(), key=lambda x: x[0]))  # Sort groups alphabetically

    # Find matching between Netlflix movies and movies from database (first iteration)
    print("1. iteration: Find matching between Netlflix movies and movies from database")
    number_of_movies_in_database_before = sum([len(movies) for movies in all_movies_grouped.values()])  # For statistics

    # Find matchings
    all_movies_grouped, (found_netflix_movies, no_matching_movies_found, missing_movies_in_db) =\
        do_one_matching_run(all_movies_grouped, netflix_movies_and_series, 1.0, 0,
                            start_group_criterium, database_movies_criterium, netflix_movie_criterium)

    # Outputs statistics
    number_of_movies_in_database_after = sum([len(movies) for movies in all_movies_grouped.values()])
    output_matching_results(found_netflix_movies, no_matching_movies_found, missing_movies_in_db,
                            number_of_movies_in_database_before, number_of_movies_in_database_after)

    # Group, find matchings and output results in further iterations
    i = 2

    for group_criteria_for_one_run in group_criteria:  # Iterate over all groupings/word splitting arrays
        for min_ratio, max_year_diff in ratio_max_year_combis:
            print(f"\n\nIteration: {i}")
            print(f"Groups (split movies: True = word/number, False = letter/number): {group_criteria_for_one_run}")
            print(f"Minimal ratio for matching movies (= similiar movies): {min_ratio}")
            print(f"Max year/temporal differene between movies: {max_year_diff}")
            i += 1

            # Find matches between missing in database movies and database movies
            print("\nFind matches between missing in database movies and database movies:")
            number_of_movies_in_database_before = sum([len(movies) for movies in all_movies_grouped.values()])  # For statistics
            all_movies_grouped, (new_found_netflix_movies, new_no_matching_movies_found, missing_movies_in_db) =\
                do_one_matching_run(all_movies_grouped, missing_movies_in_db, min_ratio, max_year_diff,
                                    group_criteria_for_one_run, database_movies_criterium, netflix_movie_criterium)
            number_of_movies_in_database_after = sum([len(movies) for movies in all_movies_grouped.values()])  # For statistics

            # Extend list of found Netflix movies and not found movies (= probably series)
            found_netflix_movies.extend(new_found_netflix_movies)
            no_matching_movies_found.extend(new_no_matching_movies_found)

            output_matching_results(found_netflix_movies, no_matching_movies_found, missing_movies_in_db,
                                    number_of_movies_in_database_before, number_of_movies_in_database_after)

            # Find matches between movie or series (not in database) and database movies
            print("\nFind matches between movie or series (not in database) and database movies:")
            number_of_movies_in_database_before = sum([len(movies) for movies in all_movies_grouped.values()])  # For statistics
            all_movies_grouped, (new_found_netflix_movies, no_matching_movies_found, new_missing_movies_in_db) =\
                do_one_matching_run(all_movies_grouped, no_matching_movies_found, min_ratio, max_year_diff,
                                    group_criteria_for_one_run, database_movies_criterium, netflix_movie_criterium)
            number_of_movies_in_database_after = sum([len(movies) for movies in all_movies_grouped.values()])  # For statistics

            # Extend list of found Netflix movies and not found movies (= probably series)
            found_netflix_movies.extend(new_found_netflix_movies)
            missing_movies_in_db.extend(new_missing_movies_in_db)

            output_matching_results(found_netflix_movies, no_matching_movies_found, missing_movies_in_db,
                                    number_of_movies_in_database_before, number_of_movies_in_database_after)

    return found_netflix_movies, no_matching_movies_found, missing_movies_in_db


if __name__ == "__main__":
    # Define variables
    all_movies, netflix_movies_and_series = {}, {}
    all_movies_grouped, netflix_movies_and_series_grouped = defaultdict(OrderedDict), defaultdict(OrderedDict)
    found_netflix_movies, netflix_series, missing_movies_in_db = [], [], []

    # # Connect to database and read all movies
    # all_movies_table = Movies()
    # all_movies_from_database = [
    #     {
    #         "id": movie_id,
    #         "original_title": movie["original_title"],
    #         "title": movie["title"],
    #         "release_year": format_int(movie["release_date"].split("-")[0], -100)
    #     }
    #     for movie_id, movie in list(all_movies_table.get_all().items())]
    # print(f"Found {len(all_movies)} movies in database")

    # # Read Netflix movies
    # with open(vars.local_netflix_movies_file_path, "r") as netflix_movies_file:
    #     netflix_movies_and_series = [
    #         {
    #             "netflix_id": movie.split(",", 2)[0],
    #             "title": movie.split(",", 2)[2].strip(),
    #             "year": format_int(movie.split(",", 2)[1], -200)
    #         }
    #         for movie in netflix_movies_file.readlines()]
    #     print(f"Found {len(netflix_movies_and_series)} Netflix movies")

    # # Save prepared movies files in file
    # save_object_in_file("updated_data/all_movies_db.pickle", all_movies_from_database)
    # save_object_in_file("updated_data/all_netflix_movies_and_series_db.pickle", netflix_movies_and_series)


    # Read all movies from file
    all_movies_from_database = load_object_from_file("updated_data/all_movies_db.pickle")
    netflix_movies_and_series = load_object_from_file("updated_data/all_netflix_movies_and_series_db.pickle")

    """
    Find matching between missing in database Netlflix movies and movies from database
    First group by first second words/numbers and the bey first word/number and second letter/number
    """
    print("Find matching between missing in database Netflix movies and remaining movies from database:")
    ratio_max_year_combis = [(0.99, MAX_YEAR_DIFFERENCE), (0.9, 0), (0.99, 3), (0.8, 0), (0.9, MAX_YEAR_DIFFERENCE), (0.8, 3)]
    group_criteria = [(True, True), (True, False)]
    found_netflix_movies, netflix_series, missing_movies_in_db =\
        optimize_matchings(all_movies_from_database, netflix_movies_and_series, ratio_max_year_combis, group_criteria)
    
    # TODO: Aufrufhierarchie dokumentieren + Parameter besser dokumentieren!!!

    print("\nSample of found/matched Netlfix movies:")
    for movie in found_netflix_movies[:10]:
        print(movie)

    print("\nSample of still in database missing Netlfix movies:")
    for movie in missing_movies_in_db[:10]:
        print(movie)

    print("\nSample of movies that are very likely series:")
    for movie in netflix_series[:10]:
        print(movie)

    # Save movies statistics (= found/not found movies)
    save_object_in_file(vars.map_for_netflix_movies_to_db_movies_path, found_netflix_movies)
    save_object_in_file(vars.missing_netflix_movies_in_database_path, missing_movies_in_db)
    save_object_in_file(vars.netflix_series_path, netflix_series)

    found_netflix_movies = load_object_from_file(vars.map_for_netflix_movies_to_db_movies_path)
    missing_movies_in_db = load_object_from_file(vars.missing_netflix_movies_in_database_path)
    netflix_series = load_object_from_file(vars.netflix_series_path)

    with open("updated_data/insert_user_histories_in_db/found_netflix_movies.txt", "w", encoding="utf-8") as file:
        for movie in found_netflix_movies:
            file.write(str(movie) + "\n")

    with open("updated_data/insert_user_histories_in_db/netflix_series.txt", "w", encoding="utf-8") as file:
        for movie in netflix_series:
            file.write(str(movie) + "\n")

    with open("updated_data/insert_user_histories_in_db/missing_movies_in_db.txt", "w", encoding="utf-8") as file:
        for movie in missing_movies_in_db:
            file.write(str(movie) + "\n")

    # vars.map_for_netflix_movies_to_db_movies
