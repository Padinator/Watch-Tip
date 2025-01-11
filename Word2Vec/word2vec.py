import sys

import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from pathlib import Path

# ---------- Import own python modules ----------
project_dir = Path(__file__).parents[1]
sys.path.append(str(project_dir))

from helper.file_system_interaction import load_object_from_file

users = load_object_from_file("tmp_small.pickle")

# print(f"Type of users: {type(users)}\nLength of Users: {len(users)}\nFirst user of users: {list(users)[0]}")
# print(users[1025579])

user_id_with_movie_history = {}
for user_id, movie_list in users.items():
    movie_history_of_user = []
    for movie in movie_list:
        movie_history_of_user.append(movie["movie_id"])

    user_id_with_movie_history[user_id] = movie_history_of_user

# print(user_id_with_movie_history)

sentences = [[str(movie) for movie in movies] for movies in user_id_with_movie_history.values()]
print(f"lenght_of_sentences: {len(sentences)}")

model = Word2Vec(sentences, window=2, min_count=1, sg=1)

movie_vector = model.wv["100"]
# print(f"Vector for the movie 100: {movie_vector}")
# print(f"Length of vetor: {len(movie_vector)}")

similarity = model.wv.similarity("2292", "100")
print(f"Similarity between movie 100 and 2292: {similarity}")

# exit(1)
movie_ids = model.wv.index_to_key
# print(f"length_of_movie_ids: {len(movie_ids)}")

# print(f"movie_ids: {movie_ids}, length of movie_ids: {len(movie_ids)}")
movie_vectors = [model.wv[movie_id] for movie_id in movie_ids]

movie_vectors_array = np.array(movie_vectors)
tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(movie_vectors_array)
# print(f"length_of_recuded_vectors: {len(reduced_vectors)}")

plt.figure(figsize=(10, 10))
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])

for i, movie_id in enumerate(movie_ids):
    plt.annotate(movie_id, (reduced_vectors[i, 0], reduced_vectors[i, 1]))

plt.title(f"T-SNE visualization of user {list(users)[0]}")
plt.show()
