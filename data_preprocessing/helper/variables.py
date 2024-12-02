from pathlib import Path


# Define some basic paths, where (updated) data will be stored
original_data_path = Path("data")
updated_data_path = Path("updated_data")
fil_db_with_test_data_path = Path(f"{updated_data_path}/fil_db_with_test_data")
count_genres_data_path = Path(f"{updated_data_path}/count_genres")
calculate_real_genres_data_path = Path(f"{updated_data_path}/calculate_real_genres")

# Define location of data sets
data_set_date = "11_28_2024"
local_movie_data_set_path = original_data_path / f"movie_ids_{data_set_date}.json"
local_producers_and_actors_data_set_path = original_data_path / f"person_ids_{data_set_date}.json"
local_producer_company_data_set_path = original_data_path / f"production_company_ids_{data_set_date}.json"
local_netflix_movies_file_path = original_data_path / f"netflix_user_data/movie_titles.csv"

# Define name of data sets after a specific convention
movie_data_set = f"movie_ids_{data_set_date}.json.gz"
producers_and_actors_data_set = f"person_ids_{data_set_date}.json.gz"
producer_company_data_set = f"production_company_ids_{data_set_date}.json.gz"

# Define file paths for interrupting script (fil_db_with_test_data.py)
missing_movies_file = fil_db_with_test_data_path / "missing_movies.txt"
index_of_next_movie_file = fil_db_with_test_data_path / "index_of_next_movie.txt"
all_reviews_tmp_data_file = fil_db_with_test_data_path / "all_reviews.pickle"

# Define file paths for collected actors, producers and production companies (fil_db_with_test_data.py)
local_actors_file_path = fil_db_with_test_data_path / "/all_actor.json"
local_producers_file_path = fil_db_with_test_data_path / "all_producer.json"
local_production_companies_file_path = fil_db_with_test_data_path / "all_production_company.json"

# Define error file paths (count_genres.py)
skipped_movies_file_path = count_genres_data_path / "skipped_movies.txt"

# Define error file paths (calculate_real_genres.py)
calc_real_genres_error_file_actors =  calculate_real_genres_data_path / "error_calc_real_genres_actors.txt"
calc_real_genres_error_file_producers =  calculate_real_genres_data_path / "error_calc_real_genres_producers.txt"
calc_real_genres_error_file_production_companies =  calculate_real_genres_data_path / "error_calc_real_genres_production_companies.txt"

# Define path of storaging temporary dict between netflix movies and movies in database
map_for_netflix_movies_to_db_movies = updated_data_path / "insert_user_histories_in_db/netflix_movies_map.pickle"

# Define URLs to data sets of all movies, producers and production companies
url_data_all_movies = f"http://files.tmdb.org/p/exports/{movie_data_set}"
url_data_all_producers_and_actors = f"http://files.tmdb.org/p/exports/{producers_and_actors_data_set}"
url_data_all_companies = f"http://files.tmdb.org/p/exports/{producer_company_data_set}"

# Define api token
api_key = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiNGY1YTQ0ZWM2NGM0YTY3NWY0NTJmMTFmMmVhY2QxYyIsIm5iZiI6MTczMTI0NzU1OC4wOTcxOTk0LCJzdWIiOiI2NmVkNjI0MzU3NmUyY2NhMWFmZTA2NWUiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.WPajvzzzoD8U1vqoJuwpkKFcczmFHZOFzb_1xhpDryM"

# Define URLs and for fetching detail data of movies and genres
abstract_movie_url = "https://api.themoviedb.org/3/movie/replace_id?append_to_response=credits&language=en-US"
abstract_movie_reviews_url = "https://api.themoviedb.org/3/movie/replace_id/reviews?language=en-US&page=page_number"
abstract_person_url = "https://api.themoviedb.org/3/person/replace_id?language=en-US"
abstract_production_company_url = "https://api.themoviedb.org/3/company/replace_id"
genre_url = "https://api.themoviedb.org/3/genre/movie/list?language=en"
headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {api_key}"
}
