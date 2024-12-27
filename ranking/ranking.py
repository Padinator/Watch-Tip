import json
import re
import openai
import os
import sys

import google.generativeai as genai
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from typing import Any, Dict

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[1]
sys.path.append(str(project_dir))

from database.movie import Movies
from database.user import Users
from helper.file_system_interaction import save_object_in_file, load_object_from_file
from helper.api_requester import request_url
from helper import variables as vars

all_users_table = Users()
all_movies_table = Movies()

# all_movies = all_movies_table.get_all()
# all_users = all_users_table.get_all()

# save_object_in_file(Path("ranking/data/all_movies.json"), all_movies)
# save_object_in_file(Path("ranking/data/all_users.json"), all_users)

# all_movies = load_object_from_file(Path("ranking/data/all_movies.json"))
# all_users = load_object_from_file(Path("ranking/data/all_users.json"))

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
                results[movie_name] = review_data

    return results


movie_ids = list(movies.keys())

filtered_reviews = get_reviews_by_movie_ids(all_users, movies)

# for movie, review in filtered_reviews.items():
#     print(f"Movie: {movie}")
#     print(f"Review: {review['content']}")
#     print("-" * 100)

# Until here the code above is only a workaround.
# Normally you would get instantly all 10 best matching movies, with all information,
# including the user reviews

# OpenAI variant
# openai.api_key = os.environ["openai"]


# def analyze_user_review(review):
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a Film Critic Analyst, specializing in the evaluation of films based on user feedback. You analyze texts to evaluate the film objectively by identifying positive and negative aspects and comparing their frequency. Your goal is to create a single overall rating and indicate the ratio of positive to negative points.",
#             },
#             {"role": "user", "content": "Analyze the following review {review}"},
#         ],
#     )
#     return response["choices"][0]["message"]["content"]


# for movie, review in filtered_reviews.items():
#     print(f"Rezension: {review}")
#     print(f"Analyse: {analyze_user_review(review)}")


def set_up_llm(model_name: str, system_instruction: str) -> genai.GenerativeModel:
    # TODO: Currently only working with the google 'gemini' model
    # Setup and test this also with the openai 'gpt' model
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


def generate_ai_response(filtered_reviews: Dict[str, str], model: genai.GenerativeModel) -> Dict[str, Dict[str, Any]]:
    """
    Generate AI responses for a set of best matching movies on the user preferences

    Parameters
    ----------
    filtered_reviews : dict of str
        A dictionary where the keys are movie titles and the values are their corresponding reviews.

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
    for movie, review in filtered_reviews.items():
        response = model.generate_content(f"Analyze the following review: {review}")

        cleaned_response = re.search(r"\{.*?\}", response.text, re.DOTALL)
        if cleaned_response:
            try:
                response_data = json.loads(cleaned_response.group())
                movies_with_their_responses[movie] = response_data
            except json.JSONDecodeError as e:
                print(f"Error parsing cleaned JSON for {movie}: {e}")
        else:
            print(f"No valid JSON found in response for {movie}")
            movies_with_their_responses[movie] = None

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
    ranked_movies = []

    for movie, response in movies_with_responses.items():
        score = response["score"]
        num_highlights = len(response.get("highlights", []))
        tfidf_score, hightlight_diversity = calculate_tfidf_and_diversity(response["highlights"])
        final_score = score * 0.7
        scores = np.array([response["score"], tfidf_score, hightlight_diversity])
        ranked_movies.append((movie, final_score, score, num_highlights, tfidf_score, hightlight_diversity, scores))

    ranked_movies.sort(key=lambda x: -x[1])

    ranking = {}

    for idx, (movie, final_score, score, num_highlights, tfidf_score, hightlight_diversity, scores) in enumerate(
        ranked_movies, start=1
    ):
        ranking[movie] = {
            "rank": idx,
            "final_score": final_score,
            "score": score,
            "num_highlights": num_highlights,
            "tfidf score": tfidf_score,
            "highlight diversity": hightlight_diversity,
            "scores": scores,
        }

    return ranking


def calculate_tfidf_and_diversity(highlights):
    if not highlights:
        return 0, 0

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(highlights)
    tfidf_scores = np.sum(tfidf_matrix.toarray(), axis=1)
    tfidf_score = np.mean(tfidf_scores)

    unique_words = set()
    for sentence in highlights:
        unique_words.update(sentence.lower().split())
    highlight_diversity = len(unique_words)

    return tfidf_score, highlight_diversity


def scale_scores(movies_with_their_responses):

    weights = {"sentiment_score": 0.5, "tfidf_score": 0.3, "highlight_diversity": 0.2}
    scaler = MinMaxScaler()

    for _, response in movies_with_their_responses.items():
        normalised_scores = scaler.fit_transform(response["scores"])
        final_score = np.dot(
            normalised_scores, np.array([weights["sentiment_score"], weights["tfidf_score"], weights["sity"]])
        )
        response["scores"] = final_score


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
        "score": float (between -1 and 1),
        "highlights": ["key phrase 1", "key phrase 2", ...]
        }
    """

    llm = set_up_llm("gemini-1.5-flash", system_instruction)

    movies_with_their_responses = generate_ai_response(filtered_reviews, llm)

    # print_in_clean_format(movies_with_their_responses)

    ranking = create_movie_ranking(movies_with_their_responses)

    scale_scores(ranking)

    for movie, details in ranking.items():
        print(
            f"{details['rank']}: {movie} (Final Score: {details['final_score']}, Highlights: {details['num_highlights']}, TF-IDF Score: {details["tfidf score"]}, Highlight diversity: {details["highlight diversity"]}, Scores: {details["scores"]})"
        )
