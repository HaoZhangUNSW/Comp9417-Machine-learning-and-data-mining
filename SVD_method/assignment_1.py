import os
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from math import sqrt
import random

# read file and generate a small file to code and test
def generate_sample():
    df = pd.read_csv('data/rating.csv')
    sample_df = df.sample(frac =0.005)
    sample_df.to_csv("data/sample_4.csv")


def generate_100K():
    with open(file) as f:
        matrix=[line.strip().split("\t")[:-1] for line in f]
    matrix = np.array(matrix)
    pd.DataFrame({'userId':matrix[:,0],'movieId': matrix[:,1],'rating':matrix[:,2]}).to_csv('./data/100k_data.csv')

def get_training_set(file):
    # read sample file
    sample_data = pd.read_csv( file )
    # get the 3 columns of userid, movieid and rating
    training_set = sample_data[['userId', 'movieId', 'rating']].drop_duplicates()
    return training_set



def get_pivot_matrix(training_set):
    # pivot the dataframe to get a table of userid as row, movieid as column and rating as value

    pivot_table = training_set.groupby( ['userId', 'movieId'] )['rating'].mean().unstack().fillna(0)
    # pivot_table = sample_data[['userId', 'movieId', 'rating']].pivot(index='userId',columns='movieId',values='rating').fillna(0)
    pivot_matrix = pivot_table.to_numpy()
    return pivot_matrix, pivot_table.index, pivot_table.columns

def get_variance_matrix(pivot_matrix):
    mean_userid = np.mean(pivot_matrix, axis=1).reshape(-1,1)
    return pivot_matrix - mean_userid, mean_userid

def singular_vector_decomposition(variance_matrix):
    left_matrix, middle_matrix, right_matrix = svds( variance_matrix, k=20 )
    middle_diag = np.diag( middle_matrix )
    return left_matrix, middle_diag, right_matrix

def get_predicted_rating(left_matrix, middle_diag, right_matrix, index, columns, mean_userid):
    combined_matrix = np.dot( np.dot( left_matrix, middle_diag ), right_matrix ) + mean_userid
    predicted_rating = pd.DataFrame(combined_matrix,index=index, columns=columns)
    return predicted_rating

def evalue_of_rmse(predicted_matrix,pivot_matrix):
    rated_movie = pivot_matrix[pivot_matrix.nonzero()].flatten()
    predicted_movie = predicted_matrix[pivot_matrix.nonzero()].flatten()
    return sqrt(mean_squared_error(rated_movie, predicted_movie))

def predict(userid, recommend_num, model, movie_df):
    recommended_movieid = model[model.index==userid].sort_values(by=userid,ascending=False, axis=1).columns[:recommend_num]

    result = movie_df[movie_df['movieId'].isin(recommended_movieid.values)][['title']].to_string(index=False)

    return result

def predict_rate(userid, movieid,model):
    return model[model.index==userid and model['movieId']==movieid]

def run(file):
    training_set = get_training_set( file )
    index = random.randint(0,100)
    userid = training_set['userId'][index]
    pivot_matrix, index, columns = get_pivot_matrix( training_set )
    variance_matrix, mean_userid = get_variance_matrix( pivot_matrix )
    left_matrix, middle_diag, right_matrix = singular_vector_decomposition( variance_matrix )
    predicted_rating = get_predicted_rating( left_matrix, middle_diag, right_matrix, index, columns, mean_userid )
    rmse = evalue_of_rmse( predicted_rating.values, pivot_matrix )
    #  testing
    print(rmse)
    # read movie.csv
    movie_df = pd.read_csv( './data/movie.csv' )
    print('the recommend_movies of number=5 for userid={} : '.format(userid))
    recommend_movie = predict( userid, 5, predicted_rating, movie_df )
    print( recommend_movie )



def plot_data(percent_data,rmses):
    import matplotlib.pyplot as plt
    plt.bar( percent_data, rmses, width = 0.05)
    plt.xlabel( 'sample_data' )
    plt.ylabel( 'RMSE' )
    plt.title('compare different sizes of movie data')
    plt.show()

if __name__=='__main__':
    #
    # select a specified file
    print('it\'s loading: ')

    for i in [0,1,2]:
        file = './data/sample_{}.csv'.format(i)
        # processing
        print( 'RMSE of {}% data:'.format((i+1)/10) )
        run( file )


    print('bar chart of RMSE (0.1, 0.2, 0.3 ): ')
    rmses = [3.551560927145061, 3.5476378896337515, 3.542448714317859]
    sizes = [0.1, 0.2, 0.3]
    plot_data(sizes,rmses)








