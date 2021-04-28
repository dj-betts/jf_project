import pandas as pd
import numpy as np

from src.ItemItemRecommender import ItemItemRecommender
from sklearn.model_selection import train_test_split

import src.ItemItemRecommender as helper

from scipy import sparse
from scipy.sparse import csr_matrix

#import necessary libraries
from sklearn.metrics import mean_squared_error
from math import sqrt



if __name__ == '__main__':
    #1. import data
    df = helper.load_data_city(filename='/Users/djbetts/Desktop/jf_project/data/recommender_top50_geo__city_noChina.csv')

    #2. convert to score
    #3. drop event_name

    #4. merge video titles
    #5. remove unlabeled city names
    df = helper.preprocess_df(df)

    #8. create user, movie encoder/id system
    #9. encode reviews for scipy.sparse.csr_matrix. index MUST be int not str
    reviews, cities, movies = helper.encoding_tool(df)
    print(f'Number of users: {cities.shape}')
    print(f'Number of movies: {movies.shape}')

    #10. test, train, split
    train, test = train_test_split(reviews, random_state = 100)
    print(f'Training Sample: {train.shape}')
    print(f'Test Sample: {test.shape}')

    ratings_as_mat = csr_matrix((train.score, ((train.city_id), (train.movie_id))))
    print(ratings_as_mat.toarray())

    rec = ItemItemRecommender(neighborhood_size=75)
    rec.fit(ratings_as_mat)

    rec.pred_all_users(report_run_time=True)
    #automate testing

    print(test.groupby('city_id').count().head(20))

    city = input(f'Which city_id would you like to score?')

    #helper.test_function(city_id=city)