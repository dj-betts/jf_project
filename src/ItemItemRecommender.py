import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from time import time


class ItemItemRecommender(object):

    def __init__(self, neighborhood_size):
        self.neighborhood_size = neighborhood_size

    def fit(self, ratings_mat):
        self.ratings_mat = ratings_mat
        self.n_users = ratings_mat.shape[0]
        self.n_items = ratings_mat.shape[1]
        self.item_sim_mat = cosine_similarity(self.ratings_mat.T)
        self._set_neighborhoods()

    def _set_neighborhoods(self):
        least_to_most_sim_indexes = np.argsort(self.item_sim_mat, 1)
        self.neighborhoods = least_to_most_sim_indexes[:, -self.neighborhood_size:]

    def pred_one_user(self, user_id, report_run_time=False):
        start_time = time()
        items_rated_by_this_user = self.ratings_mat[user_id].nonzero()[1]
        # Just initializing so we have somewhere to put rating preds
        out = np.zeros(self.n_items)
        for item_to_rate in range(self.n_items):
            relevant_items = np.intersect1d(self.neighborhoods[item_to_rate],
                                            items_rated_by_this_user,
                                            assume_unique=True)                                      
        # assume_unique speeds up intersection op
        # note: self.ratings_mat has data type `sparse_lil_matrix`, while
        # self.items_sim_mat is a numpy array. Luckily for us, multiplication
        # between these two classes is defined, and even more luckily,
        # it is defined to as the dot product. So the numerator
        # in the following expression is an array of a single float
        # (not an array of elementwise products as you would expect
        #  if both things were numpy arrays)
            out[item_to_rate] = self.ratings_mat[user_id, relevant_items] * \
                self.item_sim_mat[item_to_rate, relevant_items] / \
                self.item_sim_mat[item_to_rate, relevant_items].sum()
        if report_run_time:
            print("Execution time: %f seconds" % (time()-start_time))
        cleaned_out = np.nan_to_num(out)
        return cleaned_out

    def pred_all_users(self, report_run_time=False):
        start_time = time()
        all_ratings = [
            self.pred_one_user(user_id) for user_id in range(self.n_users)]
        if report_run_time:
            print("Execution time: %f seconds" % (time()-start_time))
        return np.array(all_ratings)

    def top_n_recs(self, user_id, n=10):
        pred_ratings = self.pred_one_user(user_id)
        item_index_sorted_by_pred_rating = list(np.argsort(pred_ratings))
        items_rated_by_this_user = self.ratings_mat[user_id].nonzero()[1]
        unrated_items_by_pred_rating = [item for item in item_index_sorted_by_pred_rating
                                        if item not in items_rated_by_this_user]
        return unrated_items_by_pred_rating[-n:]

def score(event_name):
    d ={
        'videostarts':1,
        'videoplay':1, 
        'a_media_progress10':(1*.1),
        'a_media_progress25':(1*.25),
        'a_media_progress50':(1*.5), 
        'a_media_progress75':(1*.75), 
        'a_media_progress90':(1*.9)
    }
    
    return(d[event_name])

def load_data_city(filename='/Users/djbetts/Desktop/jf_project/data/recommender_top50_geo__city_noChina.csv'):
    data = pd.read_csv(filename)
    df = data.copy()

    df = df[df['geo__city'] != '(not set)']

    return df
    
def preprocess_df(df):
    #merge two video title columsn into one column and drop the old columns
    df['video_title'] = df['event_params__videotitle'].fillna(df['event_params__video_title'])
    df.drop(['event_params__videotitle','event_params__video_title'], axis=1, inplace=True)
    df['score'] = df['event_name'].apply(score)
    df.drop(['event_name'], axis=1, inplace=True)

    return df

def encoding_tool(df, user='geo__city'):
    groupby = df.groupby([user, 'video_title'])

    #could change groupby function to sum here
    reviews = groupby.mean()
    reviews.reset_index(inplace=True)

    pivot_table = reviews.pivot_table(values='score', index=user, columns='video_title')
    
    #change cities to user groups
    cities = pd.Series(pivot_table.index, index=np.arange(1,(len(pivot_table.index)+1)))
    cities_code = {v: k for k, v in cities.items()}
    reviews['city_id'] = reviews.geo__city.replace(to_replace=cities_code)

    movies = pd.Series(pivot_table.columns, index=np.arange(1,(len(pivot_table.columns)+1)))
    movies_code = {v: k for k, v in movies.items()}
    reviews['movie_id'] = reviews.video_title.replace(to_replace=movies_code)

    reviews = reviews[['city_id', 'movie_id', 'score']]

    return reviews, cities, movies

def test_function(city_id):
    city_name = cities.loc[city_id]
    print(f'City selected: {city_name}')
    
    city_test_df = test[test.city_id == city_id]
    predictions = rec.pred_one_user(user_id=city_id)
    city_test_df['prediction'] = predictions[test[test.city_id == city_id].movie_id.values]
    
    print(city_test_df)
    
    mse = sqrt(mean_squared_error(city_test_df.score, city_test_df.prediction))
    recommend = movies.loc[rec.top_n_recs(user_id=city_id, n=10)]
    
    print(mse)
    print(recommend)