import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import google.generativeai as genai
import numpy as np
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[1]
sys.path.append(str(project_dir))

from database.movie import Movies
from database.user import Users
from helper import variables as vars
from helper.api_requester import request_url
from helper.file_system_interaction import load_object_from_file, save_object_in_file

all_users_table = Users()
all_movies_table = Movies()

# all_movies = all_movies_table.get_all()
# all_users = all_users_table.get_all()

# save_object_in_file(Path("ranking/data/all_movies.json"), all_movies)
# save_object_in_file(Path("ranking/data/all_users.json"), all_users)

# all_movies = load_object_from_file(Path("ranking/data/all_movies.json"))
# all_users = load_object_from_file(Path("ranking/data/all_users.json"))

keybert = KeyBERT()

# The following code is not a official part of the ranking system!
# This is a list of different movies, that matches the best on the user preferences
movies = {
    13: "Forrest Gump",
    120: "The Lord of the Rings: The Fellowship of the Ring",
    330: "The Lost World: Jurassic Park",
    480105: "47 Meters Down: Uncaged",
    568: "Apollo 13",
    436270: "Black Adam",
}


all_users = None
with open("test.json", "r") as file:
    all_users = json.load(file)


def get_reviews_by_movie_ids(reviews, movie_ids):
    results = {}

    for _, user_reviews in reviews.items():
        for review_data in user_reviews.values():
            movie_id = review_data["movie_id"]

            if movie_id in movie_ids:
                movie_name = movie_ids[movie_id]

                if movie_name not in results:
                    results[movie_name] = []

                results[movie_name].append(review_data)

    return results


movie_ids = list(movies.keys())

filtered_reviews = get_reviews_by_movie_ids(all_users, movies)


def set_up_llm(model_name: str, system_instruction: str) -> genai.GenerativeModel:
    """
    Set up a generative large language model with the specified model name and system instruction.

    Parameters
    ----------
    model_name : str
        The name of the large language model to be used.
    system_instruction : str
        The system instruction to configure the large language model.

    Returns
    -------
    genai.GenerativeModel
        An instance of the generative model configured with the specified parameters.
    """

    return genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction,
    )


def generate_ai_response(
    filtered_reviews: Dict[str, str], model: genai.GenerativeModel, delay: int
) -> Dict[str, Dict[str, Any]]:
    """
    Generate AI responses for a set of best matching movies on the user preferences

    Parameters
    ----------
    filtered_reviews : dict of str
        A dictionary where the keys are movie titles and the values are their corresponding reviews.
    model: genai.GenerativeModel
        An instance of the generative model used to generate responses.

    Returns
    -------
    dict of dict
        A dictionary where the keys are movie titles and the values are dictionaries containing the AI-generated responses.
        If the response cannot be parsed as JSON, the value will be None.

    Notes
    -----
    This function uses a model to generate content based on the provided reviews. It expects the model's response to contain
    a JSON object. If the JSON object cannot be parsed, an error message will be printed and the corresponding movie's value
    in the returned dictionary will be None.
    """

    movies_with_their_responses = {}

    for movie, reviews in filtered_reviews.items():
        responses = []

        print(f"Processing movie: {movie} with {len(reviews)} reviews")
        for index, review_data in enumerate(reviews, start=1):
            review = review_data.get("content", "")
            if not review:
                print(f"Skipping empty review for movie: {movie}, Review nr.{index}")
                continue

            try:
                print(f"Analyzing review nr.{index} for '{movie}'")
                response = model.generate_content(f"Analyze the following review: {review}")

                cleaned_response = re.search(r"\{.*?\}", response.text, re.DOTALL)
                if cleaned_response:
                    try:
                        response_data = json.loads(cleaned_response.group())
                        responses.append(response_data)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing cleaned JSON for {movie}, Review  nr.{index}: {e}")
                        responses.append(None)
                else:
                    print(f"No valid JSON found in response for {movie}, Review nr.{index}")
                    responses.append(None)

            except Exception as e:
                print(f"Error processing review nr.{index} for '{movie}': {e}")
                responses.append(None)

            print(f"Waiting for {delay} seconds before the next request")
            time.sleep(delay)

        movies_with_their_responses[movie] = responses

    return movies_with_their_responses


def print_in_clean_format(movies_with_their_responses: Dict[str, Dict[str, Any]]) -> None:
    """
    Prints the movies and their responses in a clean format.

    Parameters
    ----------
    movies_with_their_responses : dict of dict
        A dictionary where the keys are movie titles (str) and the values are dictionaries containing responses (dict).

    Returns
    -------
    None
    """

    for movie, response in movies_with_their_responses.items():
        print(f"Movie: {movie}\nResponse: {response}")
        print("-" * 100)


def create_movie_ranking(movies_with_responses: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Create a ranking of movies based on their scores and number of highlights.

    Parameters
    ----------
    movies_with_responses : dict
        A dictionary where the keys are movie titles and the values are dictionaries containing
        the movie's score and optionally a list of highlights.

    Returns
    -------
    dict
        A dictionary where the keys are movie titles and the values are dictionaries containing
        the movie's rank, final score, original score, and number of highlights.
    """

    movies_with_their_values = {}

    for movie, response in movies_with_responses.items():
        tfidf_score = calculate_tfidf(response["highlights"])
        movies_with_their_values[movie] = {"sentiment_score": response["sentiment_score"], "tfidf_score": tfidf_score}

    print(movies_with_their_values)

    sorted_ranked_movies = calculate_and_normalise_final_score(movies_with_their_values)

    return sorted_ranked_movies


def calculate_tfidf(highlights: List[str]) -> float:
    """
    Calculate the weighted TF-IDF score for a list of text highlights.

    This function computes the TF-IDF scores for the given highlights and
    weights them by their sentiment scores. The final score is the mean of
    these weighted TF-IDF scores.

    Parameters
    ----------
    highlights : list of str
        A list of text highlights to be analyzed.

    Returns
    -------
    float
        The mean of the weighted TF-IDF scores. If the input list is empty,
        returns 0
    """

    if not highlights:
        return 0

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(highlights)
    tfidf_scores = np.sum(tfidf_matrix.toarray(), axis=1)

    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [sentiment_analyzer.polarity_scores(h)["compound"] for h in highlights]

    weighted_tfidf_scores = [
        tfidf_score * sentiment_score for tfidf_score, sentiment_score in zip(tfidf_scores, sentiment_scores)
    ]

    return np.mean(weighted_tfidf_scores)


def calculate_and_normalise_final_score(movies_with_their_values: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate and normalize the final score for a list of movies based on their sentiment and tfidf scores.

    Parameters
    ----------
    movies_with_their_values : dict
        A dictionary where the keys are movie identifiers and the values are dictionaries containing
        'sentiment_score' and 'tfidf_score' for each movie.

    Returns
    -------
    dict
        A dictionary of movies sorted by their final normalized score in descending order. Each movie's
        dictionary will include an additional key 'final_score' representing the calculated final score.

    Notes
    -----
    The final score is calculated as a weighted sum of the sentiment score and tfidf score, with weights
    of 0.7 and 0.3 respectively. The scores are normalized using MinMaxScaler before calculating the final score.
    """

    weights = {"sentiment_score": 0.6, "tfidf_score": 0.4}

    scores_matrix = []
    for response in movies_with_their_values.values():
        scores_matrix.append([response["sentiment_score"], response["tfidf_score"]])

    scaler = MinMaxScaler()
    scaled_scores = scaler.fit_transform(scores_matrix)

    for idx, (_, response) in enumerate(movies_with_their_values.items()):
        final_score = np.dot(scaled_scores[idx], [weights["sentiment_score"], weights["tfidf_score"]])
        response["final_score"] = final_score

    return dict(
        sorted(
            movies_with_their_values.items(),
            key=lambda item: (item[1]["final_score"]),
            reverse=True,
        )
    )


def analyze_aspects_with_tfidf(review, top_n=5):

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform([review])
    tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1))
    # print(f"\nReview: {review}\n\nTF-IDF Score: {tfidf_scores}")

    sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    # print(f"\nSorted_keywords: {sorted_keywords}")

    return [keyword for keyword, score in sorted_keywords]


def analyze_aspects_with_bert(review: str, top_n: int) -> List[str]:
    """
    Analyze aspects of a review using BERT model to extract keywords.

    Parameters
    ----------
    review : str
        The text of the review to analyze.
    top_n : int
        The number of top keywords to extract.

    Returns
    -------
    list of str
        A list of extracted keywords.
    """
    keywords = keybert.extract_keywords(review, keyphrase_ngram_range=(1, 2), top_n=top_n, stop_words="english")
    return [keyword[0] for keyword in keywords]


def analyze_sentiment_for_keywords(review: str, top_n: int) -> Dict[str, int]:
    """
    Analyze the sentiment for the top N keywords in a review.

    This function extracts the top N keywords from a review using a BERT-based
    aspect analysis method and then analyzes the sentiment of each keyword using
    the VADER sentiment analysis tool.

    Parameters
    ----------
    review : str
        The text of the review to analyze.
    top_n : int
        The number of top keywords to extract and analyze.

    Returns
    -------
    Dict[str, int]
        A dictionary where the keys are the extracted keywords and the values
        are the compound sentiment scores for each keyword.
    """
    # keywords = analyze_aspects_with_tfidf(review, top_n)
    keywords = analyze_aspects_with_bert(review, top_n)
    # print(f"Review: {review},\nkeywords: {keywords}")
    # print(f"\nKeywords: {keywords}")

    analyzer = SentimentIntensityAnalyzer()
    keyword_sentiment = {}

    for keyword in keywords:
        sentiment = analyzer.polarity_scores(keyword)
        # print(f"\nKeyword: {keyword}: sentiment: {sentiment}")
        keyword_sentiment[keyword] = sentiment["compound"]

    # print(f"\nKeyword_sentiment: {keyword_sentiment}")
    return keyword_sentiment


def rank_movies(movies_with_their_reviews: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
    """
    Ranks movies based on their reviews using sentiment analysis and ratings.
    Parameters

    ----------
    movies_with_their_reviews : Dict[str, List[Dict[str, Any]]]
        A dictionary where the keys are movie titles and the values are lists of dictionaries containing review data.
        Each review dictionary should have a "content" key with the review text and optionally a "rating" key with the review rating.

    Returns
    -------
    Dict[str, float]
        A dictionary where the keys are movie titles and the values are the calculated scores, sorted in descending order.
    """

    analyzer = SentimentIntensityAnalyzer()
    scores = {}

    for movie, reviews in movies_with_their_reviews.items():
        total_sentiment_score = 0
        total_keywords_sentiment = 0
        total_rating = 0

        for review_data in reviews:
            rating = review_data.get("rating", None)

            if rating is None:
                rating = 2.0

            sentiment_scores = analyzer.polarity_scores(review_data["content"])
            sentiment_score = sentiment_scores["compound"]

            # keyword_sentiment = analyze_sentiment_for_keywords(review, top_n=5)
            sentiment = analyze_sentiment_for_keywords(review_data["content"], top_n=5)
            print(f"Movie: {movie}, sentiment_keywords: {sentiment}")
            # print(sentiment)

            keyword_sentiment = sum(sentiment.values())
            total_keywords_sentiment += keyword_sentiment

            total_sentiment_score += sentiment_score
            total_rating += rating

        avg_sentiment = total_sentiment_score / len(reviews)
        avg_rating = total_rating / len(reviews)
        avg_keywords_sentiment = total_keywords_sentiment / len(reviews)

        score = (0.5 * avg_sentiment) + (0.3 * avg_rating) + (0.2 * avg_keywords_sentiment)

        scores[movie] = score
        print(
            f"Movie: {movie}, score: {score:.2f}, avg_sentiment: {avg_sentiment}, avg_rating: {avg_rating}, avg_keywords_sentiment: {avg_keywords_sentiment}"
        )

    ranked_movies = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    # print(ranked_movies)

    for rank, (movie, score) in enumerate(ranked_movies.items(), start=1):
        print(f"Rank: {rank}, movie: {movie}, score: {score:.2f}")

    return ranked_movies


if __name__ == "__main__":

    genai.configure(api_key=os.environ["genai"])

    system_instruction = """
        You are a text analysis expert. Your task is to analyze user reviews of movies and evaluate their sentiment. For each review, follow these steps:
        1. Identify the overall sentiment of the review as positive, neutral, or negative.
        2. Assign a sentiment score on a scale from -1 to 1:
        - Positive sentiment: 0.1 to 1.0
        - Neutral sentiment: -0.1 to 0.1
        - Negative sentiment: -1.0 to -0.1
        3. Highlight key reasons or phrases that influenced the sentiment classification.

        Always return your response in the following JSON format:
        {
        "sentiment": "positive/neutral/negative",
        "sentiment_score": float (between -1 and 1),
        "highlights": ["key phrase 1", "key phrase 2", ...]
        }
    """

    ranked_movies = rank_movies(filtered_reviews)

    # llm = set_up_llm("gemini-1.5-flash", system_instruction)

    # movies_with_their_responses = generate_ai_response(filtered_reviews, llm, delay=1)

    # print(movies_with_their_responses)

    # print_in_clean_format(movies_with_their_responses)

    # sorted_ranking_movies = create_movie_ranking(movies_with_their_responses)

    # for index, key in enumerate(sorted_ranking_movies, start=1):
    #     print(f"Rank: {index}, Key: {key}, Final score: {sorted_ranking_movies[key]["final_score"]:.2f}")

    # print("-" * 40)
    # print(sorted_ranking_movies)
