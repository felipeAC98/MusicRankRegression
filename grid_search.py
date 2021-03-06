from classes.musicData import music_data_entity, get_prep_mus_data
import classes.regressor
import traceback
import pandas as pd
import numpy as np
import copy

musicData=get_prep_mus_data()
musicData.df.drop(columns=['music_name'],inplace=True)

#Aplicando one hot enconding
musicData.useOneHotEncoder('main_genre','genre-')
musicData.useOneHotEncoder('music_lang')

musicData.df.drop(columns=['artPopularity'],inplace=True)

#obtendo release time - tempo em meses em que a musica foi lancada
try:
	musicData.monthsAfterRelease(musicData.df,'release_date')

except:
	print(' release_time nao encontrada: '+str(traceback.format_exc()))

musicData.normalize()
musicData.train_test_split(targetFeatureName="popularity")

#======= Linear Regressor #=======
'''
linear_grid_params={
	'fit_intercept':[False,True]
}
linear_regressor=classes.regressor.linear_regressor(musicData)
print(linear_regressor.grid_search(linear_grid_params))
print(linear_regressor.get_grid_best_score())
print(linear_regressor.algorithm)
#'''
#======= KNN #=======

#regressor score

#GRID search 
#'''
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
print("tree_regressor tree_grid_params: "+str(tree_regressor.grid_search(tree_grid_params)))
print("tree_regressor get_grid_best_estimator: "+str(tree_regressor.get_grid_best_estimator()))
print("tree_regressor get_grid_best_score: "+str(tree_regressor.get_grid_best_score()))

#'''

'''
#Random forest
rf_grid_params={
	'criterion':["squared_error"],
	#'min_impurity_decrease':[0.005,0.001,0.01,0.1],
	'min_impurity_decrease':[0.01,0.003,0.005,0.007],
	'n_estimators':[200]
	#'n_estimators':[100,200,400]
}

randon_forest_regressor=classes.regressor.randon_forest_regressor(musicData)
print("randon_forest_regressor rf_grid_params: "+str( randon_forest_regressor.grid_search(rf_grid_params)))
print("randon_forest_regressor get_grid_best_estimator: "+str(randon_forest_regressor.get_grid_best_estimator()))
print("randon_forest_regressor get_grid_best_score: "+str(randon_forest_regressor.get_grid_best_score()))
#'''

#Xgboost score
'''
xgboostPrams={
	'learning_rate':[0.01,0.3,0.5],
	'max_depth':[6,10,15],
	'subsample':[0.5,1],
	'gamma':[0,1,3],
	'n_jobs':[6]
}

xgboost_regressor=classes.regressor.grid_xgboost_regressor(musicData,xgboostPrams)
print(xgboost_regressor.get_best_param())
print(xgboost_regressor.get_best_estimator())
#'''