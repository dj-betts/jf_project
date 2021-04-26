import pandas as pd
import numpy as np

from src.ItemItemRecommender import ItemItemRecommender
from sklearn.model_selection import train_test_split

import src.ItemItemRecommender as helper

from scipy import sparse
from scipy.sparse import csr_matrix

#1. import data
#2. convert to score
#5. drop event_name
#3. merge video titles
#4. remove unlabeled city names
#5. groupby user, item and sum
#6. reset index to create list of reviews

#7. create pivot table to visualize sparse matrix df
#8. create user, movie encoder/id system
#9. encode reviews for scipy.sparse.csr_matrix. index MUST be int not str

#10. test, train, split

if __name__ == '__main__':
    df = helper.load_data_city(filename='/Users/djbetts/Desktop/jf_project/data/recommender_top50_geo__city.csv')

    df = helper.preprocess_df(df)

    reviews, cities, movies = helper.encoding_tool(df)
    print(cities.shape)
    print(movies.shape)

    train, test = train_test_split(reviews, random_state = 100)

    ratings_as_mat = csr_matrix((train.score, ((train.city_id), (train.movie_id))))
    print(ratings_as_mat.shape)

    rec = ItemItemRecommender(neighborhood_size=75)
    rec.fit(ratings_as_mat)

    rec.pred_all_users(report_run_time=True)
    #automate testing
