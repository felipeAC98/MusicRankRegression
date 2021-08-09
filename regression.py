import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

#Obtendo dados
matchSpotify4Mula =("matchSpotify4Mula.csv")
features=['music_id', 'music_name', 'music_lang', 'art_id','art_name', 'art_rank', 'main_genre', 'related_genre','musicnn_tags','danceability','energy','key','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature','mus_rank']

X = pd.read_csv(matchSpotify4Mula, sep=',', names = features)

print(X.head())

# Fit regression model
n_neighbors = 5

knn = neighbors.KNeighborsRegressor(n_neighbors)
Y = X['mus_rank']
X=X.drop(columns=['mus_rank'])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

knnFit = knn.fit(x_train, y_train)

preResult=knnFit.predict(x_test)

acuracia = metrics.accuracy_score(y_test, preResult)

print(acuracia)