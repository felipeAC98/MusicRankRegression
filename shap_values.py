from classes.musicData import music_data_entity, get_prep_mus_data
import classes.regressor
from classes.analytics import shap_values
import transform
import numpy as np
import sys
import argparse
import copy

def get_music_ID_by_name(musicData,musicName="Bang",dropColumns=True):
	musicID = musicData.df.index[musicData.df['music_name'] == musicName].tolist()[0]

	#Removendo atributo dos dfs
	drop_colums(musicData)

	return musicID

def get_music_ID_by_popularity(musicData,popularity=50,dropColumns=True):

	musicID=0
	for key in musicData.yTest:
		if key>popularity:
			break

		musicID+=1

	musicaSelecionada=musicData.xTest.iloc[musicID]
	print("Musica selecionada: " +str(musicaSelecionada["music_name"]))
	drop_colums(musicData)
	return musicID, musicaSelecionada["music_name"]

def drop_colums(musicData,column='music_name'):
	#Removendo atributo dos dfs
	musicData.df.drop(columns=[column],inplace=True)
	musicData.yTest.drop(columns=[column],inplace=True)
	musicData.yTrain.drop(columns=[column],inplace=True)
	musicData.xTrain.drop(columns=[column],inplace=True)
	musicData.xTest.drop(columns=[column],inplace=True)

def main():

	#Obtencao dos parametros
	parser = argparse.ArgumentParser(description='args')
	parser.add_argument('--algorithm') 					#Algoritmo que sera utilizado para o teste, todos serao utilizados caso nao seja definido
	parser.add_argument('--dropFollowers')				#Define se ira remover a caracteristica de total de seguidores, por padrao ela nao sera removida
	parser.add_argument('--unDropArtPopularity')		#Define se ira remover a caracteristica popularidade do artista, por padrao ela SERA removida
	parser.add_argument('--dropMainGenre')				#Define se ira remover a caracteristica popularidade do artista, por padrao ela SERA removida
	parser.add_argument('--dropDataPercent')			#Divisor que ira os dados, ex: 2, ira remover metade das amostras

	parser.add_argument('--musicName')					#Define se a aplicacao sera realizada sobre somente uma musica - busca pelo nome
	parser.add_argument('--minMusicPop')					#Define se a aplicacao sera realizada sobre somente uma musica - busca por popularidade maior que a recebida
	parser.add_argument('--shapPlot')					#Define o tipo de plot que sera realizado pelo SHAP, quando nao definido todos serao plotados
	parser.add_argument('--namePS')						#Observacao a mais para colocar no nome do arquivo gerado

	args = parser.parse_args()

	musicData=get_prep_mus_data()

	#deixando todas musicas maiusculas
	musicData.df['music_name'] = musicData.df['music_name'].str.upper()

	if args.namePS != None:
		namePS=str(args.namePS)

	if str(args.dropFollowers).lower() == "true":
		musicData.df.drop(columns=['totalFollowers'],inplace=True)

	#Verificando se o teste do modelo sera sobre os atributos de teste ou treino
	if str(args.unDropArtPopularity).lower() != "true":
		musicData.df.drop(columns=['artPopularity'],inplace=True)

	if str(args.dropMainGenre).lower() == "true":
		musicData.df.drop(columns=['main_genre'],inplace=True)
	else:
		#Aplicando one hot enconding
		musicData.df = transform.useOneHotEncoder(musicData.df, 'main_genre','genre-')

	musicData.df = transform.useOneHotEncoder(musicData.df, 'music_lang')

	#obtendo release time - tempo em meses em que a musica foi lancada
	try:
		musicData.df = transform.monthsAfterRelease(musicData.df,'release_date')

	except:
		print(' release_time nao encontrada: '+str(traceback.format_exc()))

	#removendo algumas amostras para deixar o treinamento inicial mais rapido
	if args.dropDataPercent!= None and float(args.dropDataPercent) >= 0 and float(args.dropDataPercent) < 1:
		nIndexsToDrop=int(len(musicData.df.index)*float(args.dropDataPercent))
		indexsToDrop = np.random.choice(musicData.df.index, nIndexsToDrop, replace=False)
		musicData.df = musicData.df.drop(indexsToDrop)

		print("nIndexsToDrop: "+str(nIndexsToDrop))

	#dividindo a base
	musicData.train_test_split(targetFeatureName="popularity")
	musicDataBackup=copy.copy(musicData)

	musicID=None
	musicName=None

	if args.musicName != None:
		try:
			musicID=get_music_ID(musicData,musicName=str(args.musicName).upper())
			musicName=str(args.musicName)

		except:
			print("Erro ao obter musica")
			musicID=None

	elif args.minMusicPop != None:
		musicID, musicName=get_music_ID_by_popularity(musicData,popularity=int(args.minMusicPop))

	if str(args.algorithm).lower() in ["tree","rf","xgboost"]:

		#======= Tree
		if str(args.algorithm).lower() == "tree":

			algName="tree_regressor"
			treeRegressor=classes.regressor.tree_regressor(musicData,min_impurity_decrease=1)
			treeRegressor.fit()
			shap=shap_values(treeRegressor,preShapConfig=False)

		#======= #Random forest
		elif str(args.algorithm).lower() == "rf":

			algName="random_forest"
			randon_forest_regressor=classes.regressor.randon_forest_regressor(musicData,min_impurity_decrease=0.001,n_estimators=400)
			randon_forest_regressor.fit()
			shap=shap_values(randon_forest_regressor,preShapConfig=False)

		else:

			algName="xgboost"
			xgboost_regressor=classes.regressor.xgboost_regressor(musicData,learning_rate=0.3,max_depth=10,subsample=1,n_jobs=4)
			xgboost_regressor.fit()
			shap=shap_values(xgboost_regressor)

		if str(args.shapPlot).lower() == "tree_explainer" or  args.shapPlot == None:
			shap.tree_explainer(algName+"-"+namePS)

		if str(args.shapPlot).lower() == "explainer" or  args.shapPlot == None:
			shap.explainer(algName+"-"+namePS)

		if str(args.shapPlot).lower() == "explainer_default" or  args.shapPlot == None:
			shap.explainer_default(algName+"-"+namePS)

		if str(args.shapPlot).lower() == "decision_plot" or  args.shapPlot == None:
			shap.decision_plot(algName+"-"+namePS)

		if str(args.shapPlot).lower() == "bar_plot" or  args.shapPlot == None:
			shap.bar_plot(algName+"-"+namePS)

		if str(args.shapPlot).lower() == "music_decision_plot" or  args.shapPlot == None:
			if musicID==None:
				print("Parametro musicPop ou musicName sao necessario porem nao validos")
			else:
				shap.music_decision_plot(algName+"-"+musicName,musID=musicID)

if __name__ == '__main__':
    sys.exit(main())