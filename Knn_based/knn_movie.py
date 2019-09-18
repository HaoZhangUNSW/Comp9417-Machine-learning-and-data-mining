# Recommenders system application for MovieLens project (https://en.wikipedia.org/wiki/Recommender_system)
# Definition: A recommender system or a recommendation system is a subclass of information filtering system
# that seeks to predict the "rating" or "preference" a user would give to an item.
# Main methods: Collaborative Filtering(user- and item-based), Content-based Filtering, Hybrid Recommender systems
# Problems: data sparse,scalability,cold start
# Solution:
# we will use knn-based technique and focus on item-based collaboration filtering to train and predict model in this section.

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as scs
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
movies_file = "movies.csv"
rating_file = "ratings.csv"

# load movie and rating useful data, we just need movieId, title, userId, rating.
movie_inf = pd.read_csv(movies_file, usecols=['movieId', 'title'])
rating_inf = pd.read_csv(rating_file, usecols=['userId', 'movieId', 'rating'])

# print(movie_inf.head())
# print(rating_inf.info())

# data analysis
# we can see the total number of different rating for different movies by different users
movie_of_rating = pd.DataFrame(rating_inf.groupby('movieId').size(), columns=['count'])
# rating = pd.DataFrame(rating_inf.groupby('rating').size(), columns=['count'])
user_of_rating = pd.DataFrame(rating_inf.groupby('userId').size(), columns=['count'])
# print(movie_of_rating)
num_user = len(rating_inf.userId.unique())
num_movie = len(rating_inf.movieId.unique())
rating_zero = num_movie * num_user - rating_inf.shape[0]
# print(rating_zero)

# data processing
popular_movie_index = list(set(movie_of_rating.index))
popular_movie = rating_inf.movieId.isin(popular_movie_index).values
active_user_index = list(set(user_of_rating.index))
active_user = rating_inf.userId.isin(active_user_index).values
useful_rating = rating_inf[popular_movie & active_user]
movie_inf1 = useful_rating.pivot(index='movieId', columns='userId', values='rating').fillna(0)
new_movie_inf_name = {movie_name: ix for ix, movie_name in enumerate(list(movie_inf.set_index('movieId').loc[movie_inf1.index].title))}
movie_inf1_to_matrix = scs.csr_matrix(movie_inf1)
# print(movie_inf1_to_matrix)

# this part for plot Counts for Each Rating Score
# ratings_dataF = pd.DataFrame(rating_inf.groupby('rating').size(), columns=['count'])
# total_cnt = num_user * num_movie
# rating_zero_cnt = total_cnt - rating_inf.shape[0]
# # append counts of zero rating to df_ratings_cnt
# ratings_df = ratings_dataF.append(
#     pd.DataFrame({'count': rating_zero_cnt}, index=[0.0]),
#     verify_integrity=True,
# ).sort_index()
# ratings_df['log_count'] = np.log(ratings_df['count'])
# ax = ratings_df[['count']].reset_index().rename(columns={'index': 'rating score'}).plot(
#     x='rating score',
#     y='count',
#     kind='bar',
#     figsize=(12, 8),
#     title='Counts for Each Rating Score',
#     logy=True,
#     fontsize=12,
# )
# plt.show()

# return new_movie_inf_name, movie_inf1_to_matrix

# we will use knn to calculate the relation between different movies by cosine similarity method and return k neighbors
def recommend_movie(input_movie, topN_rec_movies, new_movie_inf_name, movie_inf1_to_matrix):
    name_match = []
    global movie_idx
    for name, idx in new_movie_inf_name.items():
        ratio = fuzz.ratio(name.lower(), input_movie.lower())
        if ratio >= 50:
            name_match.append((name, idx, ratio))
        else:
            continue
    name_match = sorted(name_match, key=lambda x: x[2])[::-1]
    if len(name_match) == 0:
        print("no match can be found")
    else:
         movie_idx = name_match[0][1]
         print("According to your input movie, we will give simple recommendation: {0}\n".format([ix[0] for ix in name_match]))
    fit_data = NearestNeighbors(n_neighbors=20, algorithm='brute', metric='cosine')
    # k = 25
    # fit_data = NearestNeighbors(n_neighbors=20, algorithm='brute', metric='euclidean')
    fit_data.fit(movie_inf1_to_matrix)
    distances, value = fit_data.kneighbors(movie_inf1_to_matrix[movie_idx], n_neighbors=topN_rec_movies+1)
    rec = sorted(list(zip(value.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    # below for euclidean
    # rec = sorted(list(zip(value.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1], reverse=True)[:0:-1]
    movie_name = {idx: name for name, idx in new_movie_inf_name.items()}
    print('Now, we will recommend movies to you based on recommender system:\n')
    for ix, (idx, distance) in enumerate(rec):
        print('{0}. {1}.'.format(ix+1, movie_name[idx]))





if __name__ == '__main__':
    input_movie = input("Please input your favourite movie: ")
    topN_rec_movies = int(input("Please input topN movies for your recommendation: "))
    recommend_movie(input_movie, topN_rec_movies, new_movie_inf_name, movie_inf1_to_matrix)
