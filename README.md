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

Commands for building anaconda environment under Windows:

conda create --name kint python=3 pymongo pandas numba numpy matplotlib pathlib scipy seaborn statsmodels numba scikit-learn cudnn nvidia::cuda-cupti cudatoolkit=10.1.243
conda activate kint
python -m pip install "tensorflow<2.11"
