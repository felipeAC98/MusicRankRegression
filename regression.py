import numpy as np
import pandas as pd
import transform
import regressionModels
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import traceback

#Obtendo dados
nomeDB="matchSpotify4Mula-large"
matchSpotify4Mula =(nomeDB+".csv")
#features=['music_id', 'music_name', 'music_lang', 'art_id','art_name', 'art_rank', 'main_genre', 'related_genre','musicnn_tags','danceability','energy','key','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature','period','position','mus_rank']
#features=['music_id', 'music_name', 'music_lang', 'art_id','art_name', 'art_rank', 'main_genre', 'related_genre','musicnn_tags','danceability','energy','key','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature','release_date','period','position','mus_rank']
features=['music_id', 'music_name', 'music_lang', 'art_id','art_name', 'art_rank', 'main_genre', 'related_genre','musicnn_tags','danceability','energy','key','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature','period','position','mus_rank']

X = pd.read_csv(matchSpotify4Mula, sep=',', names = features)
print(X.head())

X.drop(columns=['music_id'],inplace=True)
X.drop(columns=['music_lang'],inplace=True)
X.drop(columns=['art_id'],inplace=True)
X.drop(columns=['art_name'],inplace=True)
X.drop(columns=['music_name'],inplace=True)
X.drop(columns=['related_genre'],inplace=True)

#X.drop(columns=['main_genre'],inplace=True)
X.drop(columns=['musicnn_tags'],inplace=True)
X=X.dropna()
print(X.head())
print(len(X.index))

X = X.drop(X[X.position > 30000].index) 

X.drop(columns=['mus_rank'],inplace=True)
X.drop(columns=['period'],inplace=True)
X.drop(columns=['art_rank'],inplace=True)

#Aplicando one hot enconding
X = transform.useOneHotEncoder(X, 'main_genre')
#X = transform.useOneHotEncoder(X, 'music_lang')
#X = transform.useOneHotEncoder(X, 'related_genre') #precisa corrigir, nao esta dividindo o array em diferentes subcategorias

print(X.head())

#X = X.dropna(axis='columns')
print(X[X.isnull().any(axis=1)])

print(X.head())
print(len(X.index))
Y = X['position']

X.drop(columns=['position'],inplace=True)

print(len(Y.index))
#Splitando a feature de musicnn_tags
try:
	X = transform.splitMusicnnTags(X)

	#o segundo parametro é o parametro a ser utilizado o oneHot e o terceiro é o index dos novos campos que vao ser criados
	X = transform.useOneHotEncoder(X, 'musicnn_tags1', 'musicnn_tags')

	X = transform.useOneHotEncoder(X, 'musicnn_tags2', 'musicnn_tags', merge=True)

	X = transform.useOneHotEncoder(X, 'musicnn_tags3', 'musicnn_tags', merge=True)

except:
	print(' musicnn tags nao encontrada: '+str(traceback.format_exc()))

#obtendo release time
try:
	X = transform.monthsAfterRelease(X,'release_date')

except:
	print(' release_time nao encontrada: '+str(traceback.format_exc()))

print(X.columns)
print(len(X.index))

#Split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#KNN Regression
nNeighbors = 10

score=regressionModels.kNeighborsRegressorScore(x_train, y_train, x_test, y_test,nNeighbors)

print(score)

#Gaussian regression

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(x_train, y_train)

print("gpr.score: " +str(gpr.score(x_test,y_test)))

#SVM regression

from sklearn import svm

regr = svm.SVR()
regr.fit(x_train, y_train)

print("SVM score: " +str(regr.score(x_test,y_test)))