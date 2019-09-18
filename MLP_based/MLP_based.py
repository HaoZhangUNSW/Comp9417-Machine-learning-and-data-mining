import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.regularizers import l2
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K

movies_file = "movies.csv"
rating_file = "ratings.csv"

# load movie and rating useful data, we just need movieId, title, userId, rating.
movie_inf = pd.read_csv(movies_file, usecols=['movieId', 'title'])
# print(movie_inf)
rating_inf = pd.read_csv(rating_file, usecols=['userId', 'movieId', 'rating'])

num_users = len(rating_inf.userId.unique())
num_movies = len(rating_inf.movieId.unique())

# print(num_users)
# print(num_movies)
user_maxId = rating_inf.userId.max()
movieId_max = rating_inf.movieId.max()

# print(user_maxId)
# print(movieId_max)

# use data cleaning to reduce the dimension of movie vector to length of distinct movie number

rating_inf_new = rating_inf.pivot(index='userId', columns='movieId', values='rating')
# print(rating_inf_new)
movies = {movie_name: ix for ix, movie_name in enumerate(list(movie_inf.set_index('movieId').title))}
# print(movies)

new_movie_inf_name = {movie_name: ix for ix, movie_name in enumerate(list(movie_inf.set_index('movieId').loc[rating_inf_new.index].title))}

movie_name = {idx: name for name, idx in movies.items()}
# print(new_movie_inf_name)
# print(movie_name[399])
# print(new_movie_inf_name)
user_item = rating_inf_new.T.reset_index(drop=True).T
rating_new = user_item.reset_index('userId').melt(id_vars='userId', value_vars=user_item.columns, var_name='movieId', value_name='rating')
rating_new_1 = rating_new.dropna().sort_values(['userId', 'movieId']).reset_index(drop=True)
# rating_new_1.to_csv("rating_new.csv")
train_data, test_data = train_test_split(rating_new_1, test_size=0.2, shuffle=True, random_state=64)
# print(train_data.shape)

def MLp_model(num_users, num_movies, layers, regLayers):
    layer_num = len(layers)
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    movie_input = Input(shape=(1,), dtype='int32', name='movie_input')

    MLP_Embedding_User = Embedding(input_dim=num_users + 1, output_dim=layers[0] // 2, embeddings_initializer='uniform', name='user_embedding', embeddings_regularizer=l2(regLayers[0]), input_length=1)
    MLP_Embedding_Movie = Embedding(input_dim=num_movies + 1, output_dim=layers[0] // 2, embeddings_initializer='uniform', name='movie_embedding', embeddings_regularizer=l2(regLayers[0]), input_length=1)

    user_latent = Flatten()(MLP_Embedding_User(user_input))
    movie_latent = Flatten()(MLP_Embedding_Movie(movie_input))

    vector = Concatenate(axis=-1)([user_latent, movie_latent])

    # this part is for MLP layers
    for idx in range(1, layer_num):
        layer = Dense(units=layers[idx], activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=l2(regLayers[idx]), name='layer%d' % idx)
        vector = layer(vector)

    # prediction layer
    prediction = Dense(1, kernel_initializer='glorot_uniform', name='prediction')(vector)

    # input and output
    model = Model([user_input, movie_input], prediction)

    return model


def train_model(model, learner, batch_size, epochs, val_split, inputs, outputs, filename):

    # using rmse to test error
    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_true - y_pred)))

    # model compile
    model.compile(optimizer=learner.lower(), loss='mean_squared_error', metrics=['mean_squared_error', rmse])

    # call backs
    early_stopper = EarlyStopping(monitor='val_rmse', patience=10, verbose=1)
    model_saver = ModelCheckpoint(filepath=filename, monitor='val_rmse', save_best_only=True, save_weights_only=True)
    # train model
    policy = model.fit(inputs, outputs, batch_size=batch_size, epochs=epochs, validation_split=val_split, callbacks=[early_stopper, model_saver])
    return policy

def load_trained_model(model, weights_path):
    model.load_weights(weights_path)
    return model


if __name__ == '__main__':
    model = MLp_model(num_users, num_movies, [16, 64, 32, 16, 8], [0, 0, 0, 0, 0])
    BATCH_SIZE = 64
    EPOCHS = 40
    VAL_SPLIT = 0.25
    # policy = train_model(model, 'adam', BATCH_SIZE, EPOCHS, VAL_SPLIT,
    #                       inputs=[train_data.userId.values, train_data.movieId.values],
    #                       outputs=train_data.rating.values, filename='result.h5')
    # # train model
    load_trained_model(model, 'result.h5')
    predictions = model.predict([test_data.userId.values, test_data.movieId.values])

    rmse = lambda true, pred: np.sqrt(np.mean(np.square(np.squeeze(predictions) - np.squeeze(test_data.rating.values))))
    error = rmse(test_data.rating.values, predictions)
    # print(error)

    movie = model.get_layer('movie_embedding')

    movie_weights = movie.get_weights()[0]

    movie_lengths = np.linalg.norm(movie_weights, axis=1)

    normalized_movies = (movie_weights.T / movie_lengths).T
    # print(new_movie_inf_name["Toy Story (1995)"])
    name = input("Please input your favourite movie: ")
    name1 = """ + name + """
    print('Now, we will recommend movies to you based on recommender system:\n')
    dists = np.dot(normalized_movies, normalized_movies[movies[name]])
    closest = np.argsort(dists)[-11:-1]
    # print(closest)
    for c in reversed(closest):
        print(movie_name[c], dists[c])






