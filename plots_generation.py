from classes.musicData import music_data, get_prep_mus_data
from classes.regressor import regressor
from classes.dataVisualization import data_visualization
import transform
import traceback
import pandas as pd
import numpy as np
import copy

musicData=get_prep_mus_data()

#Aplicando one hot enconding
#musicData.df = transform.useOneHotEncoder(musicData.df, 'main_genre','genre-')
#musicData.df = transform.useOneHotEncoder(musicData.df, 'music_lang')

#obtendo release time - tempo em meses em que a musica foi lancada
try:
	musicData.df = transform.monthsAfterRelease(musicData.df,'release_date')

except:
	print(' release_time nao encontrada: '+str(traceback.format_exc()))

dataVisualization=data_visualization(musicData)

dataVisualization.plot_distribution(targetFeature="popularity")

dataVisualization.plot_qtd_musics_by(targetFeature="main_genre")