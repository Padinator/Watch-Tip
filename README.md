# Watch-Tip
Recommendation engine for movies

# How to?
This recommendation engine is based on the psychological aspect that people are watching always the same king of movies and actors playing always the smae kind of movies, same for producers.
To make this aspect useful and create an AI learning from these patterns, the "real genres" (the real contents of a movie) of movies of will be computed. These will be done by counting in which genres (per movie) an actor has played, a producer has produced and a company has financed. Normalize it and you know the real content of a movie.
The second step is to sort the proposed content based matching movies by another AI e.g. sentimal analysis. So we evaluate not only the users' votings. We also take the reviews into account!
The UI is coming soon.

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
conda create --name kint python=3 pymongo pandas numba numpy matplotlib plotly pathlib scipy seaborn statsmodels numba scikit-learn cudnn nvidia::cuda-cupti cudatoolkit=10.1.243 word2vec gensim\
conda activate kint\
python -m pip install "tensorflow<2.11"
conda install conda-forge::multicore-tsne

# Documentation
The code documentation is written with the numpy-style: [text](https://numpydoc.readthedocs.io/en/latest/format.html)
There are also some diagrams representing the runtime behaviour of the files/algorithms. See for this in the "docs" directory.
