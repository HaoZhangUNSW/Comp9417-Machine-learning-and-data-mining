this is a simple version by using 20M data of movie:

1. how to think:
  a. because the data size is so large, I only extract ramdomly a small 
  part of the 20M data, respectively sample_0.csv of 0.1%, sample_1.csv of 0.2% and sample_2.csv of 0.3%.
  b. read the samples of data and and select movieid, userid and rating columns,
  then privot the dataframe to get a pivot table in which userid as index, movieid as columns and rating as value
  c. use svd method to get 3 low rank matrixs for latent user and movie feature
  d. combine the 3 low matrixs to get the predicted matrix 
  e. test a userid and get the rating or recommended movie 

2. how to run:
  a. the samples of data must in data directory
  b. run the coding, then automately show some information
  c. prediction function is for testing a userid and return some numbers movie tiles, and I have tested random userid for each size of data
  d. print each RMSE and relative movie recommendation of userid
  e. finally there should be a bar chart showing RMSEs
  
3. have problem:
  a. this method can not deal with 20M data once, because it will exceed memory 
  to get the key part of pivot matrix
  b. if we want to deal with 100k data, format of the 100k data file should be converted into csv.

4. reference:
  all things above come from a reference, called reference directory. 
  I feel how to deal with big data is problem for any algorithm.
  