import matplotlib.pyplot as plt
import numpy as np


# Define variables
genres = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"]
random_user = np.array([0.6, 0.23, 0.5, 0.1, 0.05, 0.01, 0.2, 0, 0.1, 0.01, 0.1, 0, 0.05, 0.05, 0.5, 0.03, 0.1, 0.05, 0.01])

# Set size of plot
plt.rcParams.update({'font.size': 30})

'''
# Plot a user profile
figure = plt.figure(figsize=(6.4, 4.8))
figure.subplots_adjust(left=0.095, bottom=0.315, right=0.94, top=0.886)
plt.bar(np.arange(random_user.shape[0]), random_user)
plt.title("Wie sehr möchte ein spezifischer Nutzer ein Gerne in einem Film haben?\n")
plt.xticks(np.arange(random_user.shape[0]), genres, rotation=45, ha="right")
plt.xlabel("User-Profil")
plt.ylabel("Genre-Rating\n")

# Show results
wm = plt.get_current_fig_manager()
wm.window.state('zoomed')
plt.show()
'''

# Plot a movie profile
figure = plt.figure(figsize=(6.4, 4.8))
figure.subplots_adjust(left=0.095, bottom=0.32, right=0.98, top=0.886)
random_movie = np.array([0.7, 0.1, 0.01, 0.04, 0, 0, 0.46, 0, 0.1, 0, 0.1, 0, 0.4, 0.02, 0.7, 0, 0.35, 0.3, 0])
plt.bar(np.arange(random_user.shape[0]), random_user)
plt.title("Real genres of a movie?\n")
plt.xticks(np.arange(random_user.shape[0]), genres, rotation=45, ha="right")
plt.xlabel("Movie profile")
plt.ylabel("Genre proportion\n")

# Show results
wm = plt.get_current_fig_manager()
wm.window.state('zoomed')
plt.show()
