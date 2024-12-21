import json
import re
import openai
import os
import sys

import google.generativeai as genai

from pathlib import Path

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[1]
sys.path.append(str(project_dir))

from database.movie import Movies
from database.user import Users
from helper.file_system_interaction import save_object_in_file, load_object_from_file

all_users_table = Users()
all_movies_table = Movies()

# all_movies = all_movies_table.get_all()
# all_users = all_users_table.get_all()

# save_object_in_file(Path("ranking/data/all_movies.json"), all_movies)
# save_object_in_file(Path("ranking/data/all_users.json"), all_users)

# all_movies = load_object_from_file(Path("ranking/data/all_movies.json"))
# all_users = load_object_from_file(Path("ranking/data/all_users.json"))

# This is a list of different movies, that matches the best on the user preferences
# FIXME: Problem - Some movies in the all_users file has the same movie_id, see 138
movies = {
    1207898: "Forrest Gump",
    1027073: "The Lord of the Rings: The Fellowship of the Ring",
    50008: "The Lost World: Jurassic Park",
    471506: "47 Meters Down: Uncaged",
    138: "Apollo 13",
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

# Gemini variant and their limits
#   - 15 requests per minute
#   - 1 million tokens per minute
#   - 1500 requests per day
genai_api_key = os.environ["genai"]
genai.configure(api_key=genai_api_key)
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

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=system_instruction,
)

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

for movie, response in movies_with_their_responses.items():
    print(f"Movie: {movie}\nResponse: {response}")
    print("-" * 100)
