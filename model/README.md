# Description
The model directory consists of the data visualizations, feature extractions, preparing data, training models and evaluating the results. Because of large file size, mosts of this topics aren't uploaded to GitHub.

# Model directory structure
model<br>
├── data_visualization.ipynb - Visualtization of real genres of all movies and movie histories of users (TMDB dataset and Netflix prize dataset)<br>
├── model_target_real_genres.py - Trains and test an AI model: input: movie (real genres) histories of users; output/target/label: real genres<br>
├── results_visualization.ipynb - Visualtization of real genres of all movies and movie histories of users (TMDB dataset and Netflix prize dataset)<br>
├── target_real_genres - Contains result files of model (predicting real genres) training and testing<br>
│ &emsp; ├── [Evaluation of real genres](model/results/target_real_genres/EVALUATION.md) - Evaluates the model predicting real genres<br>
│<br>
├── model_target_prob_distr.py - Trains and test an AI model: input: one hot encoded movies or real genres of a movie + different embeddings; output/target label: probability distribution of all movies<br>
├── [Evaluation of probability distribution](model/results/target_prob_distr/EVALUATION.md) - Evaluates the model predicting possible similiar movies
