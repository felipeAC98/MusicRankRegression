from classes.musicData import music_data_entity, get_prep_mus_vis_data, get_prep_mus_data_spotify_only
from classes.regressor import regressor
from classes.dataVisualization import data_visualization
import traceback
import pandas as pd
import numpy as np
import copy
import argparse

parser = argparse.ArgumentParser(description='args')
parser.add_argument('--spotifyFOnly')				#Define se a base de dados sera somente de dados do spotify
args = parser.parse_args()
dataset="data/spotifyDataset"

if str(args.spotifyFOnly) != None:
	dataset="data/spotifyDataset"
	musicData=get_prep_mus_data_spotify_only(dataset=dataset)
else:
	musicData=get_prep_mus_vis_data()

#obtendo release time - tempo em meses em que a musica foi lancada
try:
	musicData.monthsAfterRelease('release_time')

except:
	print(' release_time nao encontrada: '+str(traceback.format_exc()))

dataVisualization=data_visualization(musicData)

dataVisualization.plot_distribution(targetFeature="popularity")

#dataVisualization.plot_distribution(targetFeature="mus_rank",length=20, height=15)

#dataVisualization.plot_log_distribution(targetFeature="mus_rank",length=10, height=8)

#dataVisualization.plot_qtd_musics_by_genre(ident="main_genre",fontScale=5,hideXLabels=True, showValues=False)

#dataVisualization.plot_qtd_musics_by_genre(ident="main_genre_detailed",length=60, height=40,fontScale=None,hideXLabels=False,showValues=True)

#dataVisualization.plot_qtd_musics_by_release_time(fontScale=5)

#dataVisualization.plot_distribution(targetFeature="release_time")

#dataVisualization.plot_boxplot(targetFeature1="popularity",targetFeature2="main_genre",length=30, height=45)

#Precisa corrigir, mas eh um plot interessante
#dataVisualization.plot_boxplot(targetFeature1="release_date",targetFeature2="popularity")

#dataVisualization.plot_corr_matrix()

#dataVisualization.plot_boxplot(targetFeature2="popularity",targetFeature1="totalFollowers",length=30, height=55,yLabel="Popularidade",xLabel="Total seguidores")