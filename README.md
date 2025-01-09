# Watch-Tip
Recommendation engine for movies

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
