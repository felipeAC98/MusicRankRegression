from classes.musicData import music_data_entity, get_prep_mus_data, get_prep_mus_data_spotify_only
import classes.regressor
import traceback
import pandas as pd
import numpy as np
import copy
import time
import sys
import argparse
import parameters as ps

def main():

	#Obtencao dos parametros
	parser = argparse.ArgumentParser(description='args')
	parser.add_argument('--algorithm') 					#Algoritmo que sera utilizado para o teste, todos serao utilizados caso nao seja definido
	parser.add_argument('--spotifyFOnly')				#Define se a base de dados sera somente de dados do spotify
	parser.add_argument('--dataset')				 	#Define o dataset a ser utilizado
	parser.add_argument('--dropDataTest')				#Habilita ou nao o teste de drop dos dados do dataset para construcao doss modelos
	parser.add_argument('--set')						#Define se ira utilizar o conjunto de treino ou de testes para o teste do modelo
	parser.add_argument('--dropParams')					#Define se ira utilizar somente os parametros principais definidos manualmente 
	parser.add_argument('--dropFollowers')				#Define se ira remover a caracteristica de total de seguidores, por padrao ela nao sera removida
	parser.add_argument('--dropArtPopularity')			#Define se ira remover a caracteristica popularidade do artista, por padrao ela SERA removida

	args = parser.parse_args()

	#Verificando se o teste do modelo sera sobre os atributos de teste ou treino
	if str(args.set).lower() == "train":
		isTest=False
	else:
		isTest=True
		print("isTest: "+str(isTest))

	if str(args.spotifyFOnly).lower() == "true":
		dataset="data/spotifyDataset"
		if args.dataset != None:
			dataset=str(args.dataset)
		musicData=get_prep_mus_data_spotify_only(dataset=dataset)

		#Aplicando one hot enconding
		musicData.useOneHotEncoder('genres','genre-')

	else:
		musicData=get_prep_mus_data()
		musicData.df.drop(columns=['music_name'],inplace=True)

		if str(args.dropParams).lower() == "true":
			#Removendo parametros que nao estao entre estes desta lista
			for column in musicData.df.columns:
				if column not in ['popularity','release_date','music_lang','totalFollowers','danceability','loudness','liveness']:
					musicData.df.drop(columns=[column],inplace=True)

		#Verificando se o teste do modelo sera sobre os atributos de teste ou treino
		else:
			#Aplicando one hot enconding
			musicData.useOneHotEncoder('main_genre','genre-')
			print(" Parametros nao foram dropados")

			if str(args.dropArtPopularity).lower() != "false":
				musicData.df.drop(columns=['artPopularity'],inplace=True)
				print("Removendo artPopularity")

		musicData.useOneHotEncoder('music_lang')

	#obtendo release time - tempo em meses em que a musica foi lancada
	try:
		musicData.monthsAfterRelease('release_date')

	except:
		print(' release_time nao encontrada: '+str(traceback.format_exc()))

	if str(args.dropFollowers).lower() == "true":
		musicData.df.drop(columns=['totalFollowers'],inplace=True)
		print("Removendo totalFollowers")

	musicData.normalize()

	#A primeira iteracao possui um drop de 0% dos dados para construcao do modelo
	dropPercents=[0,0.1,0.2,0.5,0.7,0.9] #*10

	for dropPercent in dropPercents:
		print("Percentual de drop: "+str(dropPercent))

		musicDataTemp=copy.copy(musicData)

		nIndexsToDrop=int(len(musicDataTemp.df.index)*dropPercent)

		indexsToDrop = np.random.choice(musicDataTemp.df.index, nIndexsToDrop, replace=False)
		musicDataTemp.df = musicDataTemp.df.drop(indexsToDrop)					#efetuando removao 
		
		print("Tamanho do df: "+str(len(musicDataTemp.df.index)))

		musicDataTemp.train_test_split(targetFeatureName="popularity")

		inicio = time.time()

		#======= Linear Regressor #=======
		if str(args.algorithm).lower() == "linear" or  args.algorithm == None:
			linear_regressor=classes.regressor.linear_regressor(musicDataTemp,fit_intercept=ps.fit_intercept)
			linear_regressor.fit()
			linear_regressor.get_scores(isTest=isTest)

		#======= KNN #=======
		if str(args.algorithm).lower() == "knn" or  args.algorithm == None:
			knn_regressor=classes.regressor.knn_regressor(musicDataTemp,n_neighbors=ps.n_neighbors)
			knn_regressor.fit()
			knn_regressor.get_scores(isTest=isTest)

		#======= Tree score
		if str(args.algorithm).lower() == "tree" or  args.algorithm == None:
			tree_regressor=classes.regressor.tree_regressor(musicDataTemp,min_impurity_decrease=ps.tree_min_impurity_decrease)
			tree_regressor.fit()
			tree_regressor.get_scores(isTest=isTest)

		#======= #Random forest score
		if str(args.algorithm).lower() == "rf" or  args.algorithm == None:			
			randon_forest_regressor=classes.regressor.randon_forest_regressor(musicDataTemp,min_impurity_decrease=ps.rf_min_impurity_decrease,n_estimators=ps.n_estimators)
			randon_forest_regressor.fit()
			randon_forest_regressor.get_scores(isTest=isTest)
		#'''

		'''
		#MLP score
		mlp_regressor=classes.regressor.mlp_regressor(musicDataTemp,random_state=1, max_iter=500,activation='relu',solver='adam',hidden_layer_sizes=(50,50,50,50,))
		print("MLP score: "+str(mlp_regressor.get_score(isTest=isTest)))	'''

		#Keras score
		#keras_sequential_regressor=classes.regressor.keras_sequential_regressor(musicDataTemp)
		#print("Keras score: "+str(keras_sequential_regressor.get_score(isTest=isTest)))	

		'''
		#AdaBoost score
		adaboost_regressor=classes.regressor.adaboost_regressor(musicDataTemp,n_estimators=100)
		adaboost_regressor.get_scores(isTest=isTest)
		'''

		#======= Xgboost score
		if str(args.algorithm).lower() == "xgboost" or  args.algorithm == None:		
			xgboost_regressor=classes.regressor.xgboost_regressor(musicDataTemp,learning_rate=0.3,max_depth=10,subsample=1,n_jobs=4 )
			xgboost_regressor.fit()
			xgboost_regressor.get_scores(isTest=isTest)

		fim = time.time()
		print("Tempo execucao:"+str(fim - inicio))

		if str(args.dropDataTest).lower() != "True":
			break

		print("\n")

if __name__ == '__main__':
    sys.exit(main())