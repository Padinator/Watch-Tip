# Define date of data sets
data_set_date = "11_28_2024"
local_movie_data_set_path = f"data/movie_ids_{data_set_date}.json"
local_producers_and_actors_data_set_path = f"data/person_ids_{data_set_date}.json"
local_producer_company_data_set_path = f"data/production_company_ids_{data_set_date}.json"
local_netflix_movies_file_path = "data/netflix_user_data/movie_titles.csv"

# Define name of data sets after a specific convention
movie_data_set = f"movie_ids_{data_set_date}.json.gz"
producers_and_actors_data_set = f"person_ids_{data_set_date}.json.gz"
producer_company_data_set = f"production_company_ids_{data_set_date}.json.gz"

# Define URLs to data sets of all movies, producers and production companies
url_data_all_movies = f"http://files.tmdb.org/p/exports/{movie_data_set}"
url_data_all_producers_and_actors = f"http://files.tmdb.org/p/exports/{producers_and_actors_data_set}"
url_data_all_companies = f"http://files.tmdb.org/p/exports/{producer_company_data_set}"

# Define api token
api_key = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiNGY1YTQ0ZWM2NGM0YTY3NWY0NTJmMTFmMmVhY2QxYyIsIm5iZiI6MTczMTI0NzU1OC4wOTcxOTk0LCJzdWIiOiI2NmVkNjI0MzU3NmUyY2NhMWFmZTA2NWUiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.WPajvzzzoD8U1vqoJuwpkKFcczmFHZOFzb_1xhpDryM"

# Define URLs and for fetching detail data of movies and genres
abstract_movie_url = "https://api.themoviedb.org/3/movie/movie_id?append_to_response=credits&language=en-US"
genre_url = "https://api.themoviedb.org/3/genre/movie/list?language=en"
headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {api_key}"
}
