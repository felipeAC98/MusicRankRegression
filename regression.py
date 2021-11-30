import numpy as np
import pandas as pd
import transform
import regressionModels
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import traceback
from musicData import musicData, getPrepMusData

X=getPrepMusData()

#Aplicando one hot enconding
X = transform.useOneHotEncoder(X, 'main_genre','genre-')
X = transform.useOneHotEncoder(X, 'music_lang')
#X = transform.useOneHotEncoder(X, 'art_name')

#Recontando as posicoes das musicas, para ficar um valor continuo
#X=transform.recountColumn(X,'position')
#X=transform.recountColumn(X,'popularity')

#obtendo release time - tempo em meses em que a musica foi lancada
try:
	X = transform.monthsAfterRelease(X,'release_date')

except:
	print(' release_time nao encontrada: '+str(traceback.format_exc()))

#========== Y sera o indice de popularidade do spotify

Y = X['popularity']

X.drop(columns=['position'],inplace=True)
X.drop(columns=['popularity'],inplace=True)

#for column in X.columns:
#	print(column)
print(len(X.index))

#==============================
#Aplicando feature selection
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
#lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, Y)
#logRe=LogisticRegression().fit(X, Y)
#selector = SelectFromModel(logRe, prefit=True)
#print(selector.get_support())
#X=selector.transform(X)

#Split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

xtest_np=x_test[:10]
ytest_np=y_test[:10]

#KNN Regression
from sklearn.neighbors import KNeighborsRegressor

nNeighbors = 15
KNNR = KNeighborsRegressor(n_neighbors=nNeighbors)
KNNR.fit(x_train, y_train)
print(KNNR.predict(xtest_np))
print("KNNR score: " +str(KNNR.score(x_test,y_test)))

#KNN Regression + bang
'''
from sklearn.ensemble import BaggingRegressor

breg = BaggingRegressor(base_estimator=KNeighborsRegressor(n_neighbors=nNeighbors),n_estimators=50, random_state=0).fit(x_train, y_train)
print(breg.predict(xtest_np))
print("KNN + bang: " +str(breg.score(x_test,y_test)))
'''
#score=regressionModels.kNeighborsRegressorScore(x_train, y_train, x_test, y_test,nNeighbors)

#Gaussian regression
'''
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(x_train, y_train)
#gpr = GaussianProcessRegressor(random_state=0).fit(x_train, y_train)
print(gpr.predict(xtest_np))
print("gpr.score: " +str(gpr.score(x_test,y_test)))
'''
#Gaussian + bang
'''
from sklearn.ensemble import BaggingRegressor

breg = BaggingRegressor(base_estimator=GaussianProcessRegressor(),n_estimators=10, random_state=0).fit(x_train, y_train)
print(breg.predict(xtest_np[:10]))
print("breg: " +str(breg.score(x_test,y_test)))
'''
#MultinomialNB
'''
from sklearn.naive_bayes import MultinomialNB
MNBreg = MultinomialNB()
MNBreg.fit(x_train, y_train)

print(MNBreg.predict(xtest_np[:10]))
print("MultinomialNB score: " +str(MNBreg.score(x_test,y_test)))
'''
#SVM regression
'''
from sklearn import svm

regr = svm.SVR(kernel='linear')
regr.fit(x_train, y_train)

print(regr.predict(xtest_np[:10]))
print("SVM score: " +str(regr.score(x_test,y_test)))
'''
#SGDRegressor

'''from sklearn.linear_model import SGDRegressor

sgdr = SGDRegressor(max_iter=2000, tol=1e-3)
sgdr.fit(x_train, y_train)

print(sgdr.predict(xtest_np[:10]))
print("SGDRegressor score: " +str(sgdr.score(x_test,y_test)))
'''
#MLPRegressor
'''
from sklearn.neural_network import MLPRegressor

mlpReg = MLPRegressor(random_state=1, max_iter=500,activation='relu',solver='adam',learning_rate_init=0.001).fit(x_train, y_train)
print(mlpReg.predict(xtest_np))
print("MLPReg score: " +str(mlpReg.score(x_test,y_test)))
'''
#Decision Tree Regression
'''
from sklearn import tree

treeR = tree.DecisionTreeRegressor(min_impurity_decrease=0.1)
treeR.fit(x_train, y_train)

print(treeR.predict(xtest_np))
print("treeR score: " +str(treeR.score(x_test,y_test)))
'''
#Random Forest

from sklearn.ensemble import RandomForestRegressor
randFr = RandomForestRegressor(min_impurity_decrease=0.0005, random_state=0)
randFr.fit(x_train, y_train)
print(randFr.predict(xtest_np))
print("RandomForestRegressor score: " +str(randFr.score(x_test,y_test)))

#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(randFr, X, Y, cv=5)
#print("RandomForestRegressor cross_val_score: " +str(scores))

#Extra Random Forest

from sklearn.ensemble import ExtraTreesRegressor 
extTree = ExtraTreesRegressor(min_impurity_decrease=0.0005, random_state=0)
extTree.fit(x_train, y_train)
print(extTree.predict(xtest_np))
print("ExtraTreesRegressor  score: " +str(extTree.score(x_test,y_test)))


print(ytest_np)