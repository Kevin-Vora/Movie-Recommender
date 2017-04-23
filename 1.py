import pandas as pd
import numpy as np
from collections import OrderedDict

'''
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
 encoding='latin-1')

i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')


ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
'''

def dot_product(vector_1, vector_2):  
    return sum([ i*j for i,j in zip(vector_1, vector_2)])

def get_movie_score(movie_features, user_preferences):  
    return dot_product(movie_features, user_preferences)


def get_movie_recommendations_c(user_preferences, n_recommendations):  
    #we add a column to the movies_df dataset with the calculated score for each movie for the given user
    movies_df['score'] = movies_df[movie_categories].apply(get_movie_score, 
                                                           args=([user_preferences.values()]), axis=1)
    return movies_df.sort_values(by=['score'], ascending=False)['movie_title'][:n_recommendations]



movies_df=pd.read_table('movies.dat', header=None, sep='::', names=['movie_id', 'movie_title', 'movie_genre'])

movies_df = pd.concat([movies_df, movies_df.movie_genre.str.get_dummies(sep='|')], axis=1)  

movie_categories = movies_df.columns[3:]
user_preferences = OrderedDict(zip(movie_categories, []))

a=list()
for _ in range(18):
    a+=[int(input())]

# HardCoded
user_preferences['Action'] = a[0]  
user_preferences['Adventure'] = a[1] 
user_preferences['Animation'] = a[2]  
user_preferences["Children's"] = a[3]  
user_preferences["Comedy"] = a[4]  
user_preferences['Crime'] = a[5]  
user_preferences['Documentary'] = a[6]  
user_preferences['Drama'] = a[7]  
user_preferences['Fantasy'] = a[8]  
user_preferences['Film-Noir'] = a[9]  
user_preferences['Horror'] = a[10]  
user_preferences['Musical'] = a[11]  
user_preferences['Mystery'] = a[12]  
user_preferences['Romance'] = a[13]  
user_preferences['Sci-Fi'] = a[14]  
user_preferences['War'] = a[15]  
user_preferences['Thriller'] = a[16]  
user_preferences['Western'] =a[17]



ratings_df = pd.read_table('ratings.dat', header=None, sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'])

del ratings_df['timestamp']

ratings_df = pd.merge(ratings_df, movies_df, on='movie_id')[['user_id', 'movie_title', 'movie_id','rating']]

ratings_mtx_df = ratings_df.pivot_table(values='rating', index='user_id', columns='movie_title')  
ratings_mtx_df.fillna(0, inplace=True)

movie_index = ratings_mtx_df.columns

corr_matrix = np.corrcoef(ratings_mtx_df)  

def get_movie_recommendations(user_id):
    user_index = ratings_mtx_df.T.columns
    movie_index = ratings_mtx_df.columns
    P = corr_matrix[user_id]
    h_users = list(user_index[(P>0.95) & (P<1.0)])
    rec=list()
    for movies in movie_index:
        avg=0
        for i in h_users:
            avg+=ratings_mtx_df.loc[i][movies]
        if(len(h_users)!=0):
            avg/=len(h_users)
        rec+=[avg]
    similarities_df = pd.DataFrame({
        'movie_title': movie_index,
        'sum_similarity': rec
        })
    
    user_movies = ratings_df[ratings_df.user_id==user_id].movie_title.tolist() 
    similarities_df = similarities_df[~(similarities_df.movie_title.isin(user_movies))]
    similarities_df = similarities_df.sort_values(by=['sum_similarity'], ascending=False)
    return similarities_df

recommendations = get_movie_recommendations_c(user_preferences,30)
print(recommendations)
