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

dropPercents=[0,0.1,0.2,0.5,0.7,0.9] #*10

for dropPercent in dropPercents:
	print("Percentual de drop: "+str(dropPercent))

	musicDataTemp=copy.copy(musicData)

	nIndexsToDrop=int(len(musicDataTemp.df.index)*dropPercent)

	indexsToDrop = np.random.choice(musicDataTemp.df.index, nIndexsToDrop, replace=False)
	musicDataTemp.df = musicDataTemp.df.drop(indexsToDrop)					#efetuando removao 
	
	print("Tamanho do df: "+str(len(musicDataTemp.df.index)))

	musicDataTemp.train_test_split(targetFeatureName="popularity")

	#KNN regressor score
	knn_regressor=classes.regressor.knn_regressor(musicDataTemp,n_neighbors=15)
	print("KNN score: "+str(knn_regressor.get_score()))
	print("knn_regressor MSE score: "+str(knn_regressor.get_MSE_score()))	
	print("knn_regressor MAE score: "+str(knn_regressor.get_MAEP_score()))	

	#Tree score
	tree_regressor=classes.regressor.tree_regressor(musicDataTemp,min_impurity_decrease=0.1)
	print("tree_regressor score: "+str(tree_regressor.get_score()))	
	print("tree_regressor MSE score: "+str(tree_regressor.get_MSE_score()))	
	print("tree_regressor MAE score: "+str(tree_regressor.get_MAEP_score()))	


	#Random forest score
	randon_forest_regressor=classes.regressor.randon_forest_regressor(musicDataTemp,min_impurity_decrease=0.001)
	print("Random Forest score: "+str(randon_forest_regressor.get_score()))	

	#MLP score
	mlp_regressor=classes.regressor.mlp_regressor(musicDataTemp,random_state=1, max_iter=500,activation='relu',solver='adam',hidden_layer_sizes=(50,50,50,50,))
	print("MLP score: "+str(mlp_regressor.get_score()))	

	#Keras score
	keras_sequential_regressor=classes.regressor.keras_sequential_regressor(musicDataTemp)
	print("Keras score: "+str(keras_sequential_regressor.get_score()))	

	#AdaBoost score
	adaboost_regressor=classes.regressor.adaboost_regressor(musicDataTemp,n_estimators=100)
	print("AdaBoost score: "+str(adaboost_regressor.get_score()))	
	print("AdaBoost MSE score: "+str(adaboost_regressor.get_MSE_score()))	
	print("AdaBoost MAE score: "+str(adaboost_regressor.get_MAEP_score()))		

	#Xgboost score
	xgboost_regressor=classes.regressor.xgboost_regressor(musicDataTemp,learning_rate=0.1,
			max_depth= 3,
			subsample=0.8,
			n_jobs=-1,
			random_state=42)
	print("XGboost score: "+str(xgboost_regressor.get_score()))	

	break