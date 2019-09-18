import os
import numpy as np
import pandas as pd
import csv


movie_file = "movies.csv"
rating_file = "ratings.csv"

# movie_data = pd.read_csv(movie_file, usecols=['movieId', 'title'])
# rating_data = pd.read_csv(rating_file, index_col=None, usecols=['userId', 'movieId', 'rating'])


matrix = []
max_uid = 0
max_mid = 0
with open("ratings.csv","r") as input:
    reader = csv.reader(input)
    for line in reader:
            max_uid = line[0]
with open("movies.csv","r") as input:
    reader = csv.reader(input)
    for line in reader:
            max_mid = line[0]

max_uid = int(max_uid)
max_mid = int(max_mid)
for i in range(int(max_uid)+1):
    matrix.append([-1]*int(max_mid))

with open("ratings.csv","r") as input:
    reader = csv.reader(input)
    for line in reader:
        try:
            matrix[int(line[0])-1][int(line[1])-1] = int(float(line[2]))
        except ValueError:
            continue

np_matrix = np.asarray(matrix)


with open("rating_matrix.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(np_matrix)):
        print(i)
        for j in range(len(np_matrix[i])):
            print(j)
            l = [0,0,0,0,0,0]
            tail = [1,1,1,1,1,1]
            if np_matrix[i][j] == -1:
                valid_num = np.where(np_matrix[:, j] != -1)[0].size
                invalid_num = np.where(np_matrix[:, j] == -1)[0].size
                # print(valid_num)
                # print(invalid_num)
                if (j == 0):
                    for k in range(len(np_matrix[i])):
                        tail[0] += ((np.where(np_matrix[:, k] == 0)[0].size +1) / (valid_num+6))
                        tail[1] += ((np.where(np_matrix[:, k] == 1)[0].size +1) / (valid_num+6))
                        tail[2] += ((np.where(np_matrix[:, k] == 2)[0].size +1) / (valid_num+6))
                        tail[3] += ((np.where(np_matrix[:, k] == 3)[0].size +1) / (valid_num+6))
                        tail[4] += ((np.where(np_matrix[:, k] == 4)[0].size +1) / (valid_num+6))
                        tail[5] += ((np.where(np_matrix[:, k] == 5)[0].size +1) / (valid_num+6))

                l[0] = ((np.where(np_matrix[:, j] == 0)[0].size + 1) / (valid_num + 6)) * (invalid_num * (1/max_mid)) * tail[0]
                l[1] = ((np.where(np_matrix[:, j] == 1)[0].size + 1) / (valid_num + 6)) * (invalid_num * (1/max_mid)) * tail[1]
                l[2] = ((np.where(np_matrix[:, j] == 2)[0].size + 1) / (valid_num + 6)) * (invalid_num * (1/max_mid)) * tail[2]
                l[3] = ((np.where(np_matrix[:, j] == 3)[0].size + 1) / (valid_num + 6)) * (invalid_num * (1/max_mid)) * tail[3]
                l[4] = ((np.where(np_matrix[:, j] == 4)[0].size + 1) / (valid_num + 6)) * (invalid_num * (1/max_mid)) * tail[4]
                l[5] = ((np.where(np_matrix[:, j] == 5)[0].size + 1) / (valid_num + 6)) * (invalid_num * (1/max_mid)) * tail[5]
                ma = max(l)
                index = 0
                for p in range(len(l)):
                    if ma == l[p]:
                        index = p
                np_matrix[i][j] = index
                # print(l)
                # print(np_matrix[i])
        print(np_matrix[i])
        writer.writerow(np_matrix[i])



