from classes.musicData import music_data_entity, get_prep_mus_data
import classes.regressor
import transform
import traceback
import pandas as pd
import numpy as np
import copy

musicData=get_prep_mus_data()

#Aplicando one hot enconding
musicData.df = transform.useOneHotEncoder(musicData.df, 'main_genre','genre-')
musicData.df = transform.useOneHotEncoder(musicData.df, 'music_lang')

#obtendo release time - tempo em meses em que a musica foi lancada
try:
	musicData.df = transform.monthsAfterRelease(musicData.df,'release_date')

except:
	print(' release_time nao encontrada: '+str(traceback.format_exc()))

musicData.train_test_split(targetFeatureName="popularity")

#======= KNN #=======

#regressor score

#GRID search 
'''
KNN_grid_params={
	'n_neighbors':[10,20,30,50,100,200],
	'weights':['uniform', 'distance']
}
knn_regressor=classes.regressor.knn_regressor(musicData)
print(knn_regressor.grid_search(KNN_grid_params))
print(knn_regressor.get_grid_best_score())
#'''

'''
#Tree score
tree_grid_params={
	'criterion':["squared_error","absolute_error","friedman_mse","poisson"],
	'min_impurity_decrease':[0.05,0.1,0.2,0.5,1],
	'max_features':['auto','none']
}

tree_regressor=classes.regressor.tree_regressor(musicData)
print(tree_regressor.grid_search(tree_grid_params))
print(tree_regressor.get_grid_best_score())
'''
#'''
#Random forest
rf_grid_params={
	'criterion':["squared_error"],
	'min_impurity_decrease':[0.0005,0.005,0.001,0.05],
	'n_estimators':[100,200,400]
}

randon_forest_regressor=classes.regressor.randon_forest_regressor(musicData)
print(randon_forest_regressor.grid_search(rf_grid_params))
print(randon_forest_regressor.get_grid_best_score())
#'''

#Xgboost score
'''
xgboost_params={
	'learning_rate':[0.001,0.01,0.1],
	'max_depth':[5,10,15],
	'subsample':[0.8,1]
}

xgboost_regressor=classes.regressor.xgboost_regressor(musicDataTemp)
print(xgboost_regressor.grid_search(xgboost_params))
print(xgboost_regressor.get_grid_best_score())
'''
