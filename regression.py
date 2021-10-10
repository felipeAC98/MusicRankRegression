import numpy as np
import pandas as pd
import transform
import regressionModels
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

#Obtendo dados
nomeDB="matchSpotify4Mula-tiny"
matchSpotify4Mula =(nomeDB+".csv")
#features=['music_id', 'music_name', 'music_lang', 'art_id','art_name', 'art_rank', 'main_genre', 'related_genre','musicnn_tags','danceability','energy','key','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature','period','position','mus_rank']
features=['music_id', 'music_name', 'music_lang', 'art_id','art_name', 'art_rank', 'main_genre', 'related_genre','musicnn_tags','danceability','energy','key','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature','release_date','period','position','mus_rank']

X = pd.read_csv(matchSpotify4Mula, sep=',', names = features)

X.drop(columns=['music_id'],inplace=True)
X.drop(columns=['music_lang'],inplace=True)
X.drop(columns=['art_id'],inplace=True)
X.drop(columns=['art_name'],inplace=True)
X.drop(columns=['music_name'],inplace=True)
X.drop(columns=['related_genre'],inplace=True)

Y = X['position']
X=X.drop(columns=['position'])
X.drop(columns=['mus_rank'],inplace=True)
X.drop(columns=['period'],inplace=True)

print(X.head())

#Aplicando one hot enconding
X = transform.useOneHotEncoder(X, 'main_genre')
#X = transform.useOneHotEncoder(X, 'related_genre') #precisa corrigir, nao esta dividindo o array em diferentes subcategorias
print(X.head())

#Splitando a feature de musicnn_tags
try:
	X = transform.splitMusicnnTags(X)
	print(X.head())

	#o segundo parametro é o parametro a ser utilizado o oneHot e o terceiro é o index dos novos campos que vao ser criados
	X = transform.useOneHotEncoder(X, 'musicnn_tags1', 'musicnn_tags')
	print(X.head())

	X = transform.useOneHotEncoder(X, 'musicnn_tags2', 'musicnn_tags', merge=True)
	print(X.head())

	X = transform.useOneHotEncoder(X, 'musicnn_tags3', 'musicnn_tags', merge=True)
	print(X.head())

except:
	print(' musicnn tags nao encontrada: '+str(traceback.format_exc()))

#obtendo release time
try:
	X = transform.monthsAfterRelease(X,'release_time')

except:
	print(' release_time nao encontrada: '+str(traceback.format_exc()))


print(X.columns)

#Split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#KNN Regression
nNeighbors = 5

score=regressionModels.kNeighborsRegressorScore(x_train, y_train, x_test, y_test,nNeighbors)

print(score)

