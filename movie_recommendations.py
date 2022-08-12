# Original file is located at
#  https://colab.research.google.com/drive/1f_rObzqW1J-0VqMyPwwi9gZ87EVg-5pX - ANUJ PATEL

#importing dependencies
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#data collection and preprocessing
df = pd.read_csv("/content/drive/MyDrive/movies.csv") #use custom path of CSV

#selecting the relevent features for recamendation
selected_features = ['genres','keywords','tagline','cast','director']

# replacing NA values in the selected features only
for features in selected_features :
  df[features] = df[features].fillna('')

# combining all the 5 selected features
combined_features = df['genres']+' '+df['keywords']+' '+df['tagline']+' '+df['cast']+' '+df['director']

# converting string into featured vectors
vctzer = TfidfVectorizer()
feature_vectors = vctzer.fit_transform(combined_features)

# getting similarity score values using cosine similarity
similarity = cosine_similarity(feature_vectors)
# MOVIE RECAMENDATION SYSTEM

print('Movies Recamendation System by Anuj Patel...  \n')
movie_name = input(' Enter favourite movie name : ')
list_all_movies = df['title'].tolist()
find_close_name = difflib.get_close_matches(movie_name,list_all_movies)
close_match = find_close_name[0]
index_movie = df[df.title == close_match]['index'].values[0]
similarity_score = list(enumerate(similarity[index_movie]))
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = df[df.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1

