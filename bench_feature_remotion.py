from classes.musicData import music_data_entity, get_prep_mus_data
import classes.regressor
import transform
import traceback
import pandas as pd
import numpy as np
import copy
import time

musicData=get_prep_mus_data()

for column in musicData.df.columns:
	if column not in ['popularity','release_date','music_lang','totalFollowers','danceability','loudness','liveness']:
		musicData.df.drop(columns=[column],inplace=True)

#Aplicando one hot enconding
#musicData.df = transform.useOneHotEncoder(musicData.df, 'main_genre','genre-')
musicData.df = transform.useOneHotEncoder(musicData.df, 'music_lang')

#obtendo release time - tempo em meses em que a musica foi lancada
try:
	musicData.df = transform.monthsAfterRelease(musicData.df,'release_date')

except:
	print(' release_time nao encontrada: '+str(traceback.format_exc()))

musicData.normalize()

musicData.train_test_split(targetFeatureName="popularity")

inicio = time.time()
#======= Linear Regressor #=======
#'''
linear_regressor=classes.regressor.linear_regressor(musicData,fit_intercept=False)
linear_regressor.fit()
print("Linear score: "+str(linear_regressor.get_score()))
print("MSE score: "+str(linear_regressor.get_MSE_score()))	
print("MAE score: "+str(linear_regressor.get_MAE_score()))	
print("R2 ajustado: "+str(linear_regressor.get_r2_adjusted()))	
print("RMSE: "+str(linear_regressor.get_RMSE()))	
print("MAPE: "+str(linear_regressor.get_MAPE()))	

print("Linear score (train): "+str(linear_regressor.get_score(isTest=False)))
print("MSE score (train): "+str(linear_regressor.get_MSE_score(isTest=False)))	
print("MAE score (train): "+str(linear_regressor.get_MAE_score(isTest=False)))	
print("R2 ajustado (train): "+str(linear_regressor.get_r2_adjusted(isTest=False)))	
print("RMSE (train): "+str(linear_regressor.get_RMSE()))	
print("MAPE (train): "+str(linear_regressor.get_MAPE(isTest=False)))	
#'''

#======= KNN #=======

#'''
knn_regressor=classes.regressor.knn_regressor(musicData,n_neighbors=200)
knn_regressor.fit()
print("KNN score: "+str(knn_regressor.get_score()))
print("MSE score: "+str(knn_regressor.get_MSE_score()))	
print("MAE score: "+str(knn_regressor.get_MAE_score()))	
print("R2 ajustado: "+str(knn_regressor.get_r2_adjusted()))	
print("RMSE: "+str(knn_regressor.get_RMSE()))	
print("MAPE: "+str(knn_regressor.get_MAPE()))	

print("KNN score (train): "+str(knn_regressor.get_score(isTest=False)))
print("MSE score (train): "+str(knn_regressor.get_MSE_score(isTest=False)))	
print("MAE score (train): "+str(knn_regressor.get_MAE_score(isTest=False)))	
print("R2 ajustado (train): "+str(knn_regressor.get_r2_adjusted(isTest=False)))	
print("RMSE (train): "+str(knn_regressor.get_RMSE()))	
print("MAPE (train): "+str(knn_regressor.get_MAPE(isTest=False)))	

#'''

#'''
#Tree score
tree_regressor=classes.regressor.tree_regressor(musicData,min_impurity_decrease=0.2)
tree_regressor.fit()
print("tree_regressor score: "+str(tree_regressor.get_score()))	
print("MSE score: "+str(tree_regressor.get_MSE_score()))	
print("MAE score: "+str(tree_regressor.get_MAE_score()))	
print("R2 ajustado: "+str(tree_regressor.get_r2_adjusted()))	
print("RMSE: "+str(tree_regressor.get_RMSE()))	
print("MAPE: "+str(tree_regressor.get_MAPE()))	

print("tree_regressor score (train): "+str(tree_regressor.get_score(isTest=False)))	
print("MSE score (train): "+str(tree_regressor.get_MSE_score(isTest=False)))	
print("MAE score (train): "+str(tree_regressor.get_MAE_score(isTest=False)))	
print("R2 ajustado (train): "+str(tree_regressor.get_r2_adjusted(isTest=False)))	
print("RMSE (train): "+str(tree_regressor.get_RMSE()))	
print("MAPE (train): "+str(tree_regressor.get_MAPE(isTest=False)))

#'''
#'''
#Random forest score
randon_forest_regressor=classes.regressor.randon_forest_regressor(musicData,min_impurity_decrease=0.001,n_estimators=400)
randon_forest_regressor.fit()
print("Random Forest score: "+str(randon_forest_regressor.get_score()))	
print("MMSE score: "+str(randon_forest_regressor.get_MSE_score()))	
print("MAE score: "+str(randon_forest_regressor.get_MAE_score()))
print("R2 ajustado: "+str(randon_forest_regressor.get_r2_adjusted()))	
print("RMSE: "+str(randon_forest_regressor.get_RMSE()))	
print("MAPE: "+str(randon_forest_regressor.get_MAPE()))	

print("Random Forest score (train): "+str(randon_forest_regressor.get_score(isTest=False)))	
print("MMSE score (train): "+str(randon_forest_regressor.get_MSE_score(isTest=False)))	
print("MAE score (train): "+str(randon_forest_regressor.get_MAE_score(isTest=False)))
print("R2 ajustado (train): "+str(randon_forest_regressor.get_r2_adjusted(isTest=False)))	
print("RMSE (train): "+str(randon_forest_regressor.get_RMSE()))	
print("MAPE (train): "+str(randon_forest_regressor.get_MAPE(isTest=False)))	
#'''

fim = time.time()
print("Tempo execucao:"+str(fim - inicio))
