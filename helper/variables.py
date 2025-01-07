from pathlib import Path

# Define project path
project_dir_vars = Path(__file__).parents[1]


# --------------- Define some variables for pre processing ---------------
data_preprocessing_path = project_dir_vars / "data_preprocessing"

# Define some basic paths, where (updated) data will be stored
original_data_path = data_preprocessing_path / "data"
updated_data_path = data_preprocessing_path / "updated_data"
fil_db_with_test_data_path = updated_data_path / "fil_db_with_test_data"
count_genres_data_path = updated_data_path / "count_genres"
calculate_real_genres_data_path = updated_data_path / "calculate_real_genres"
insert_user_histories_in_db_data_path = updated_data_path / "insert_user_histories_in_db"
netflix_user_data_path = original_data_path / "netflix_user_data"

# Define location of data sets
data_set_date = "11_28_2024"
local_movie_data_set_path = original_data_path / f"movie_ids_{data_set_date}.json"
local_producers_and_actors_data_set_path = (
    original_data_path / f"person_ids_{data_set_date}.json"
)
local_producer_company_data_set_path = (
    original_data_path / f"production_company_ids_{data_set_date}.json"
)
local_netflix_movies_file_path = (
    netflix_user_data_path / "movie_titles.csv"
)
local_netflix_movies_watched_file_paths = [
    netflix_user_data_path / f"combined_data_{i}.txt"
    for i in range(1, 5)
]

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
local_production_companies_file_path = (
    fil_db_with_test_data_path / "all_production_company.json"
)

# Define error file paths (count_genres.py)
missing_actors_file_path = count_genres_data_path / "missing_actors.txt"
missing_producers_file_path = count_genres_data_path / "missing_producers.txt"
missing_production_companies_file_path = (
    count_genres_data_path / "missing_production_companies.txt"
)

# Define error file paths (calculate_real_genres.py)
calc_real_genres_error_file_actors = (
    calculate_real_genres_data_path / "error_calc_real_genres_actors.txt"
)
calc_real_genres_error_file_producers = (
    calculate_real_genres_data_path / "error_calc_real_genres_producers.txt"
)
calc_real_genres_error_file_production_companies = (
    calculate_real_genres_data_path / "error_calc_real_genres_production_companies.txt"
)

# Define path for storaging temporary dict between Netflix movies and movies in database
map_for_netflix_movies_to_db_movies_path = (
    insert_user_histories_in_db_data_path / "netflix_movies_mapped_to_database.pickle"
)
missing_netflix_movies_in_database_path = (
    insert_user_histories_in_db_data_path / "missing_netflix_movies_in_database.pickle"
)
netflix_series_path = (
    insert_user_histories_in_db_data_path / "netflix_series.pickle"
)
map_for_netflix_movies_to_db_movies_path_txt = (
    insert_user_histories_in_db_data_path / "netflix_movies_mapped_to_database.txt"
)
missing_netflix_movies_in_database_path_txt = (
    insert_user_histories_in_db_data_path / "missing_netflix_movies_in_database.txt"
)
netflix_series_path_txt = (
    insert_user_histories_in_db_data_path / "netflix_series.txt"
)

# Define path for storaging temporary dict of all watchings of Netflix movies
netflix_movies_watchings_path = (
    insert_user_histories_in_db_data_path / "netflix_movies_watchings.pickle"
)

# Define URLs to data sets of all movies, producers and production companies
url_data_all_movies = f"http://files.tmdb.org/p/exports/{movie_data_set}"
url_data_all_producers_and_actors = (
    f"http://files.tmdb.org/p/exports/{producers_and_actors_data_set}"
)
url_data_all_companies = f"http://files.tmdb.org/p/exports/{producer_company_data_set}"

# Define api token
api_key = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiNGY1YTQ0ZWM2NGM0YTY3NWY0NTJmMTFmMmVhY2QxYyIsIm5iZiI6MTczMTI0NzU1OC4wOTcxOTk0LCJzdWIiOiI2NmVkNjI0MzU3NmUyY2NhMWFmZTA2NWUiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.WPajvzzzoD8U1vqoJuwpkKFcczmFHZOFzb_1xhpDryM"

# Define URLs and for fetching detail data of movies and genres
abstract_movie_url = "https://api.themoviedb.org/3/movie/replace_id?append_to_response=credits&language=en-US"
abstract_movie_reviews_url = "https://api.themoviedb.org/3/movie/replace_id/reviews?language=en-US&page=page_number"
abstract_person_url = "https://api.themoviedb.org/3/person/replace_id?language=en-US"
abstract_production_company_url = "https://api.themoviedb.org/3/company/replace_id"
genre_url = "https://api.themoviedb.org/3/genre/movie/list?language=en"
headers = {"accept": "application/json", "Authorization": f"Bearer {api_key}"}


# -------- Define some variables for preprocessing and model training --------
model_path = project_dir_vars / "model"

# Define some variables for data preparation
prepared_data_folder = model_path / "prepared_data"
user_history_file_path_with_real_genres = (
    prepared_data_folder / "user_histories_with_real_genres.pickle"
)
user_history_file_path_with_real_genres_visualization = (
    prepared_data_folder / "user_histories_with_real_genres.dataframe"
)
user_history_file_path_with_real_genres_and_reduced_dimensions = (
    prepared_data_folder
    / "user_histories_with_real_genres_and_reduced_dimensions.pickle"
)
user_history_file_path_with_real_genres_and_reduced_dimensions_visualization = (
    prepared_data_folder
    / "user_histories_with_real_genres_and_reduced_dimensions.dataframe"
)

# Define some variables for feature extraction
extracted_features_folder = model_path / "extracted_features"
extracted_features_file_path = extracted_features_folder / "extracted_features.pickle"
