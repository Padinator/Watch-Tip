# Description
Here will be described, how you can get the data for reproducing the data collection step, so that you can train the models on your own.

# Data preprocessing directory structure
data_preprocessing<br>
├── fil_db_with_test_data.py - Search in TMDB for movie detail data, movie reviews, movie producers and production companies --> insert in database<br>
├── count_genres.py - Count genres in which each actor plays, each producer produces and each company finances<br>
├── calculate_real_gernes.py - Compute real genres based on counted genres --> insert in database<br>
├── match_netflix_prize_data_to_database.py - Read Netflix prize data and match movies (not series) from it to TMDB data/movies --> write to file

# Steps for collecting data from TMDB
1. Setup docker container for the database (mongodb) with the [docker-compose.yml](docker-compose.yml). For this use the command "docerk-compose up -d"<br>
   If you already have a the collected and prepared data, you can uncomment the lines in the "docker-compose.yml" for restoring this file to the database.
2. Edit variable "api_key" storing the API token for TMDB in [helper/variables.py](../helper/variables.py)
3. Execute [fil_db_with_test_data.py](fil_db_with_test_data.py)
4. Execute [count_genres.py](count_genres.py)
5. Execute [calculate_real_gernes.py](calculate_real_gernes.py)
6. Now you can execute the model files, e.g. [First model](../model/model_target_prob_distr.py), [Second model](../model/model_target_prob_distr.py)<br>
   Read more about models in [model/README.md](../model/README.md).<br>
   You can only use the data from TMDB for the model training and testing. For more data use the Netflix Prize data (see below).

# Steps for collecting Netflix Prize data (only possible after collecting TMDB data)
7. Execute [match_netflix_prize_data_to_database.py](match_netflix_prize_data_to_database.py)
8. Now you can execute the model files, e.g. [First model](../model/model_target_prob_distr.py), [Second model](../model/model_target_prob_distr.py)<br>
   Read more about models in [model/README.md](../model/README.md)