from classes.musicData import music_data, get_prep_mus_data
from classes.regressor import regressor
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
	regressorObj=regressor(musicDataTemp)
	
	print("KNN score: "+str(regressorObj.knn_score()))
	print("Extra tree score: "+ str(regressorObj.extra_tree_score()))