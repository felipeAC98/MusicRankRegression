import numpy as np
import pandas as pd
import transform
import regressionModels
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import traceback

#Obtendo dados
nomeDB="matchSpotify4Mula-metadata"
matchSpotify4Mula =(nomeDB+".csv")
#features=['music_id', 'music_name', 'music_lang', 'art_id','art_name', 'art_rank', 'main_genre', 'related_genre','musicnn_tags','danceability','energy','key','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature','period','position','mus_rank']
#features=['music_id', 'music_name', 'music_lang', 'art_id','art_name', 'art_rank', 'main_genre', 'related_genre','musicnn_tags','danceability','energy','key','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature','release_date','period','position','mus_rank']
features=['music_id', 'music_name', 'music_lang', 'art_id','art_name', 'art_rank4Mula', 'main_genre', 'related_genre','musicnn_tags','danceability','energy','key','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature', 'release_date','period','position','mus_rank','art_rank']

X = pd.read_csv(matchSpotify4Mula, sep=',', names = features)
print(X.head())

X.drop(columns=['music_id'],inplace=True)
#X.drop(columns=['music_lang'],inplace=True)
X.drop(columns=['art_id'],inplace=True)
X.drop(columns=['art_name'],inplace=True)
X.drop(columns=['music_name'],inplace=True)
X.drop(columns=['related_genre'],inplace=True)

#X.drop(columns=['release_date'],inplace=True)
X.drop(columns=['musicnn_tags'],inplace=True)
print(X.head())
print(len(X.index))

X = X.drop(X[X.position > 40000].index) 

X.drop(columns=['mus_rank'],inplace=True)
X.drop(columns=['period'],inplace=True)
X.drop(columns=['art_rank4Mula'],inplace=True)
X=X.dropna()
#Aplicando one hot enconding
X = transform.useOneHotEncoder(X, 'main_genre')
X = transform.useOneHotEncoder(X, 'music_lang')
#X = transform.useOneHotEncoder(X, 'related_genre') #precisa corrigir, nao esta dividindo o array em diferentes subcategorias

print(X.head())

#X = X.dropna(axis='columns')

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
	X = transform.monthsAfterRelease(X,'release_date1')
	#print(X[X.isna().any(axis=1)].head())

except:
	print(' release_time nao encontrada: '+str(traceback.format_exc()))

print(X.columns)
print(len(X.index))

#Split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

xtest_np=x_test.to_numpy()
ytest_np=y_test.to_numpy()

#KNN Regression
from sklearn.neighbors import KNeighborsRegressor

nNeighbors = 10
KNNR = KNeighborsRegressor(n_neighbors=nNeighbors)
KNNR.fit(x_train, y_train)
print(KNNR.predict(xtest_np[:10]))
print("KNNR score: " +str(KNNR.score(x_test,y_test)))
#score=regressionModels.kNeighborsRegressorScore(x_train, y_train, x_test, y_test,nNeighbors)

#Gaussian regression

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(x_train, y_train)

print(gpr.predict(xtest_np[:10]))
print("gpr.score: " +str(gpr.score(x_test,y_test)))

#Gaussian + bang

from sklearn.ensemble import BaggingRegressor

breg = BaggingRegressor(base_estimator=GaussianProcessRegressor(),n_estimators=10, random_state=0).fit(x_train, y_train)
print(breg.predict(xtest_np[:10]))
print("breg: " +str(breg.score(x_test,y_test)))

#MultinomialNB
from sklearn.naive_bayes import MultinomialNB
MNBreg = MultinomialNB()
MNBreg.fit(x_train, y_train)

print(MNBreg.predict(xtest_np[:10]))
print("MultinomialNB score: " +str(MNBreg.score(x_test,y_test)))

#SVM regression

from sklearn import svm

regr = svm.SVR()
regr.fit(x_train, y_train)

print(regr.predict(xtest_np[:10]))
print("SVM score: " +str(regr.score(x_test,y_test)))

#SGDRegressor

from sklearn.linear_model import SGDRegressor

sgdr = SGDRegressor(max_iter=2000, tol=1e-3)
sgdr.fit(x_train, y_train)

print(sgdr.predict(xtest_np[:10]))
print("SGDRegressor score: " +str(sgdr.score(x_test,y_test)))

#MLPRegressor

from sklearn.neural_network import MLPRegressor

mlpReg = MLPRegressor(random_state=1, max_iter=800).fit(x_train, y_train)
print(mlpReg.predict(xtest_np[:10]))
print("MLPReg score: " +str(mlpReg.score(x_test,y_test)))

#Decision Tree Regression

from sklearn import tree

treeR = tree.DecisionTreeRegressor()
treeR.fit(x_train, y_train)

print(treeR.predict(xtest_np[:10]))
print("treeR score: " +str(treeR.score(x_test,y_test)))

print(ytest_np[:10])