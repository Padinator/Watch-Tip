# Table of contents
- [Table of contents](#table-of-contents)
- [Model behaviour](#model-behaviour)
- [Model structure](#model-structure)
- [Evaluation](#evaluation)
  - [Evaluation with PCA](#evaluation-with-pca)
  - [Evaluation with t-SNE](#evaluation-with-t-sne)
  - [Raise the number of movies to predict](#raise-the-number-of-movies-to-predict)
- [Model parameterizations](#model-parameterizations)
- [Self evaluation of models](#self-evaluation-of-models)

# Model behaviour
The model uses the a user's last n movies as movie history and predicts the next movie, which a user will watch.<br>
So the grounded truth looks like the following:<br>
**movie history (= context) --> next movie (= target)**

# Model structure
The used model has the following structure:<br>
The input dimension depends on the number of hstories used to predict the next movie (10x19 for 10 movies in history, 30x19 for 30 movies in history and 100x19 for 100 movies in history).
![screenshot](images/model.png)

# Evaluation
## Evaluation with PCA
The results of the model are stored in the directories laying in the current/this directory.<br>
The accuracy of the model can be displayed with PCA:
![screenshot](images/PCA_Predictions_for_test.png)
The plot shows the predictions (orange) to the test data, the movies which were only used as target (rosa), the movies which were only used in the movie history (= context, puple), the movies which were used as target and in history (blue) and the movies, which weren't used for testing at all (yellow.)<br>
For this training process the targets/labels were filtered, so that only movies which occur at least 100 times as target/label, will be used. This reduces the variance and hinders the model trying to overfit. But in this case the model was stuck and predicted only more or less the same movie. Filtering target movies, relaxes the output/model.<br>
Overall the variance in the plot is too high and and that the model can only predict the values around the mean values/real genres (orange points) (although filtering movies with less occurence was used). So the amount of different data is too big. An improvement could be, reducing the amount of variance by splitting the data into different parts (comming soon, see [technical presentation](../../../slides-decks/technical-presentation/technical-presentation.pptx)).<br>

## Evaluation with t-SNE
It's also possible to visualize the data with t-SNE:
![screenshot](images/t-SNE_Predictions_for_test.png)
Now you can see that the predictions are far away from the real values more than the ones computed with PCA. This happens, because t-SNE reduces the data with non linear transofrmations. So it takes only the neighbours with probability distributions into account, so that different figures origins, because there are many more predictions than targets.<br>
Overall PCA shows the correct figure/accuracy.<br>

## Raise the number of movies to predict
We raised the number of possible target movies, so that the model must not predict exactly the next movie, which leads to more flexibility. Then the loss can be computed for example to the movie closest/furthest away or to the mean of all target movies. However filtering the less occuring movies as targets isn't anymore possible after feature extraction, else targets/labels will have different shapes. This must be done beforehand (comming soon). Without filtering the model gets worse again and with raising the number of predictable movies, it gets better again, but still worse (see [serveral predictable movies](100000_0.85_100_50_10_True_False_50_89096_32/)). Adding filtering target movies probably would improve the result.

# Model parameterizations
The file [progress.txt](progress.txt) contains different parameterizations with different layers of the described model/variants of the model.

# Self evaluation of models
The different result directories e.g. "100000_0.85_30_20_1_False_False_50_89127_32" orignis from the following criteria:
- Maximal amount of data, e.g. *100000* movie histories
- Amount of train data, e.g. *0.85* (so 85 %)
- Number of movies of a movie history to use for predicting the next movie, e.g. *30* movies
- Minimal number of movies in a "movie history" that must exist and will be filled with "zero movies" (all real genres are zero) for reaching the requested number of movies per history, e.g. *20*:<br>
  *10 zero movies + 20 movies = 30 movies = 1 input feature*
- Number of movies to predict e.g. *1* or *10*
- Boolean parameter whether a movie history will be filled with zero movies, e.g. *False* then no filling with zero movies will be done.
- Boolean parameter whether the movie histories wll be extracted fine-grained, e.g. *True* then following example is relevant:<br>
  *movie history h = [m1, m2, m3, m4, m5, m6]*<br>
  *Number of movies for predicting next one = 3*<br><br>
  **Input features**:<br>
  &ensp;[<br>
    &emsp;&ensp; [m1, m2, m3],<br>
    &emsp;&ensp; [m2, m3, m4],<br>
    &emsp;&ensp; [m3, m4, m5],<br>
  &ensp;]<br><br>
  **Targets/Labels**:<br>
  &ensp;[m4, m5, m6]