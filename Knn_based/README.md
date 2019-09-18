we will use knn-based technique and focus on item-based collaboration filtering to train and predict model in this section.
Firstly, load movie and rating useful data, we just need movieId, title, userId, rating.
Secondly, data analysis, data filtering and data processing
Thirdly, we will use knn to calculate the relation between different movies by cosine similarity method and return k neighbors.
but for this part, we can ,firstly, use fuzzywuzzy method to give recommendation by simply string matching (movie name), we can set
threshold value, such as ratio no less than 50, or larger.
And then, use knn NearestNeighbors to fit and predict, for k value, i set 20, we can change it, but be careful for overfitting.
Finally, give topN movies from users' input movie name.


Attention: this method based on small dataset(100K movielens, ratings.csv, movies.csv)

when we run this progamming, we should firstly input user's favorite movie name, 
and these movies' name should come from our MOVIE dataset,
then you need to input top-k k value to get k movies recommended for users.
then we can get all the recommended movies.
