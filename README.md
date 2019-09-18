


---
# **Solutions**
- SVD
 <br>this is a simple version by using 20M data of movie:

1. how to think:
    <br>
    a. because the data size is so large, I only extract ramdomly a small
    <br>
    part of the 20M data, respectively sample_0.csv of 0.1%, sample_1.csv of 0.2% and sample_2.csv of 0.3%.
    <br>
    b. read the samples of data and and select movieid, userid and rating columns,
    then privot the dataframe to get a pivot table in which userid as index, movieid as columns and rating as value
    <br>
    c. use SVD method to get 3 low rank matrixs for latent user and movie feature
    <br>
    d. combine the 3 low matrixs to get the predicted matrix
    <br>
    e. test a userid and get the rating or recommended movie

2. how to run:
    <br>
    a. the samples of data must in data directory
    <br>
    b. run the coding, then automately show some information
    <br>
    c. prediction function is for testing a userid and return some numbers movie tiles, and I have tested random userid for each size of data
    <br>
    d. print each RMSE and relative movie recommendation of userid
    <br>
    e. finally there should be a bar chart showing RMSEs

3. have problem:
    <br>
    a. this method can not deal with 20M data once, because it will exceed memory
    to get the key part of pivot matrix
    <br>
    b. if we want to deal with 100k data, format of the 100k data file should be converted into csv.

4. reference:
    <br>
    all things above come from a reference, called reference directory.
    I feel how to deal with big data is problem for any algorithm.

- KNN (Items_based Collaborative filtering)(for small size dataset)

<br>
  we will use knn-based technique and focus on item-based collaboration filtering to train and predict model in this section. 
  <br>
  Firstly, load movie and rating useful data, we just need movieId, title, userId, rating. 
  
  Secondly, data analysis, data filtering and data processing.
  Thirdly, we will use knn to calculate the relation between different movies by cosine similarity method and return k neighbors. 
  
  But for this part, we can ,firstly, use fuzzywuzzy method to give recommendation by simply string matching (movie name), we can set threshold value, such as ratio no less than 50, or larger.
  
  And then, use knn NearestNeighbors to fit and predict, for k value, i set 20, we can change it, but be careful for overfitting. Finally, give topN movies from users' input movie name.
<br>
Attention:

  when we run this progamming, we usw small dataset, 100k movielens, and we should firstly input user's favorite movie name, and these movies' name should come from our MOVIE dataset, then you need to input top-k k value to get k movies recommended for users. 
  
  then we can get all the recommended movies.

- Naive Bayes
<br>this is a simple version by using 20M data of movie:
1. how to think:
    <br>
    a. the rating data set is spares and the number of observed ratings is small, we can use smoothing method such as Laplace to handle this problem.
    <br>
    b. We need to rebuild observed rating matrix, each row group by user Id and each column present movie Id
    <br>
    c. For predict unobserved rating we can use the Bayes expression which mentioned in out report
    <br>


  2. how to run:
    <br>
    a. unzip data.zip
    <br>
    b. make sure all data file store in the same directory with RS.py
    <br>
    c. rating_matrix.csv generate by NB.py, if you want to rebuilt rating matrix for new movie rating data, you run NB.py again, if not, we can only consider run RS.py
    <br>
    d. RS.py allows user input a favor movie name, and top-k output number
    <br>


  3. have problem:
    <br>
    a. creating rating matrix cost a lot of time, cause observe rating data set is spares, and we may need to consider reduce the dimensionality of feature.
    



Future Study:
- Deep learning (Neural Matrix Factorization vs General Matrix Factorization vs Multilayer Perceptron (MLP))
- Deep Learning (Autoencoder, CNNs based, LSTM...........)
