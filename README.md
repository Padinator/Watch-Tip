# Watch-Tip
Recommendation engine for movies

# How to?
This recommendation engine is based on the psychological aspect that people are watching always the same king of movies and actors playing always the smae kind of movies, same for producers.<br>
To make this aspect useful and create an AI learning from these patterns, the "real genres" (the real contents of a movie) of movies of will be computed. These will be done by counting in which genres (per movie) an actor has played, a producer has produced and a company has financed. Normalize it and you know the real content of a movie.<br>
The second step is to sort the proposed content based matching movies by another AI e.g. sentimal analysis. So we evaluate not only the users' votings. We also take the reviews into account!
The UI is coming soon.

# Project structure
WATCH-TIP<br>
├── data_preprocessing - Preprocess data<br>
&emsp;&emsp; ├── fil_db_with_test_data.py - Search in TMDB for movie detail data, movie reviews, movie producers and production companies --> insert in database<br>
&emsp;&emsp; ├── count_genres.py - Count genres in which each actor plays, each producer produces and each company finances<br>
&emsp;&emsp; ├── calculate_real_gernes.py - Compute real genres based on counted genres --> insert in database<br>
&emsp;&emsp; ├── match_netflix_prize_data_to_database.py - Read Netflix prize data and match movies (not series) from it to TMDB data/movies --> write to file<br>
├── database - Interaction with database (create, read update delete)<br>
├── docs - Some documentation about the project<br>
├── helper - Helper files with some helpful functions etc.<br>
├── model - Implementation/Usages of AI techniques/models to predict the real genres of the next movie or the next movie itself<br>
&emsp;&emsp; ├── data_visualization.ipynb - Visualtization of real genres of all movies and movie histories of users (TMDB dataset and Netflix prize dataset)<br>
&emsp;&emsp; ├── model_target_real_genres.py - Trains and test an AI model: input: movie (real genres) histories of users; output/target/label: real genres<br>
&emsp;&emsp; ├── results_visualization.ipynb - Visualtization of real genres of all movies and movie histories of users (TMDB dataset and Netflix prize dataset)<br>
&emsp;&emsp; ├── target_real_genres - Contains result files of model (predicting real genres) training and testing<br>
&emsp;&emsp; │ &emsp; ├── [Evaluation real genres](model/results/target_real_genres/evaluation.md) - Evaluates the model predicting real genres<br>
&emsp;&emsp; │<br>
&emsp;&emsp; ├── model_target_prob_distr.py - Trains and test an AI model: input: one hot encoded movies or real genres of a movie + different embeddings; output/target label: probability distribution of all movies<br>
&emsp;&emsp;&emsp;&emsp; ├── [Evaluation probability distribution](model/results/target_prob_distr/evaluation.md) - Evaluates the model predicting possible similiar movies<br>
├── ranking - Rank proposed/suggested movies (from AIs from directory model) to show the best at first<br>
├── slides-decks - Slides and presentations<br>
├── tests- Tests ensuring quality of several modules<br>
├── webserver - Contains not fully implemented UI --> coming soon

# Setup anaconda environment for using GPU under Windows
Necessary packages:
- python=3
- numba
- cudnn
- nvidia::cuda-cupti
- cudatoolkit=10.1.243

# Commands for building anaconda environment under Windows
conda create --name kint python=3 pymongo pandas numba numpy matplotlib plotly pathlib scipy seaborn statsmodels numba scikit-learn cudnn nvidia::cuda-cupti cudatoolkit=10.1.243 word2vec gensim<br>
conda activate kint<br>
python -m pip install "tensorflow<2.11"<br>
conda install conda-forge::multicore-tsne

# Documentation
The code documentation is written with the numpy-style: [Numpy-Style](https://numpydoc.readthedocs.io/en/latest/format.html)<br>
There are also some diagrams representing the runtime behaviour of the files/algorithms. See for this in the "docs" directory.
