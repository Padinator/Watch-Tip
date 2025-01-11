# Watch-Tip
Recommendation engine for movies

# How to?
This recommendation engine is based on the psychological aspect that people are watching always the same king of movies and actors playing always the smae kind of movies, same for producers.<br>
To make this aspect useful and create an AI learning from these patterns, the "real genres" (the real contents of a movie) of movies of will be computed. These will be done by counting in which genres (per movie) an actor has played, a producer has produced and a company has financed. Normalize it and you know the real content of a movie.<br>
The second step is to sort the proposed content based matching movies by another AI e.g. sentimal analysis. So we evaluate not only the users' votings. We also take the reviews into account!
The UI is coming soon.

# Project structure
WATCH-TIP<br>
|---- data_preprocessing - Preprocess data<br>
&emsp;&ensp; |---- fil_db_with_test_data.py &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;- Search in TMDB for movie detail data, movie reviews, movie producers and production companies --> insert in database<br>
&emsp;&ensp; |---- count_genres.py &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp; - Count genres in which each actor plays, each producer produces and each company finances<br>
&emsp;&ensp; |---- calculate_real_gernes.py &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;- Compute real genres based on counted genres --> insert in database<br>
&emsp;&ensp; |---- match_netflix_prize_data_to_database.py - Read Netflix prize data and match movies (not series) from it to TMDB data/movies --> write to file<br>
|---- database - Interaction with database (create, read update delete)<br>
|---- docs &emsp;&ensp;&nbsp; - Some documentation about the project<br>
|---- helper &emsp; - Helper files with some helpful functions etc.<br>
|---- model &emsp; - Implementation/Usages of AI techniques/models to predict the real genres of the next movie or the next movie itself<br>
&emsp;&ensp; |---- data_visualization.ipynb &emsp;&emsp;&nbsp;- Visualtization of real genres of all movies and movie histories of users (TMDB dataset and Netflix prize dataset)<br>
&emsp;&ensp; |---- model_target_real_genres.py - Trains and test an AI model: input: movie (real genres) histories of users; output/target/label: real genres<br>
&emsp;&ensp; |---- results_visualization.ipynb &emsp;&nbsp;- Visualtization of real genres of all movies and movie histories of users (TMDB dataset and Netflix prize dataset)<br>
&emsp;&ensp; |---- model_target_prob_distr.py &nbsp; - Trains and test an AI model: input: one hot encoded movies or real genres of a movie + different embeddings; output/target label: probability distribution of all movies<br>
|---- ranking &emsp;&ensp;&nbsp; - Rank proposed/suggested movies (from AIs from directory model) to show the best at first<br>
|---- slides-decks - Slides and presentations<br>
|---- tests &emsp;&emsp;&emsp; - Tests ensuring quality of several modules<br>
|---- webserver &ensp; - Contains not fully implemented UI --> coming soon

# Setup anaconda environment for using GPU under Windows
Necessary packages:
- python=3
- pymongo
- pandas
- numba
- numpy
- matplotlib
- pathlib
- scipy
- seaborn
- statsmodels
- numba
- scikit-learn
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
