import csv
import numpy as np
import os


file = "rating_matrix.csv"
input_movie = input("Please input your favourite movie: ")
print(input_movie)
topN_rec_movies = int(input("Please input topN movies for your recommendation: "))
print(topN_rec_movies)
matrix = []

with open("rating_matrix.csv","r") as input:
    reader = csv.reader(input)
    for line in reader:
            matrix.append(line)

np_matrix = np.asarray(matrix)

movie_dic_1={}
movie_dic_2={}
with open("movies.csv","r") as input:
    reader = csv.reader(input)
    for line in reader:
        try:
            movie_dic_1[line[1]] = int(line[0])
            movie_dic_2[int(line[0])] = line[1]
        except ValueError:
            continue


output_list = []
exist = 1
try:
    movie_dic_1[input_movie]
except KeyError:
    exist = 0
if exist == 1:
    index = movie_dic_1[input_movie]

    for i in range(len(np_matrix)):
        if np_matrix[i][index-1] == '5':
            output_list = np.where(np_matrix[i] == '5')[0]
    for j in range(topN_rec_movies):
        print(movie_dic_2[int(output_list[j])+1])
else:
    for i in range(len(np_matrix)):
        # print(np.where(np_matrix[i] == '5')[0].size)
        output_list.append([np.where(np_matrix[:, i] == '5')[0].size, i])
        # output_list.append(i)

    output_list = sorted(output_list, key = lambda i : i[0], reverse=True)
    # print(output_list)
    # max_index = max(output_list)
    # max_index = 379
    # print(m)

    # output_list = np.where(np_matrix[max_index] == '5')[0]
    for j in range(topN_rec_movies):
        print(movie_dic_2[int(output_list[j][1])+1])
        # print(output_list[j][0])




