import numpy as np
import pandas as pd
import transform
from sklearn.model_selection import train_test_split
import traceback

class music_data_entity():

	def __init__(self,nomeDB,features):
		self.nomeDB=nomeDB
		self.df=None
		self.features=features

	def read_csv(self, csvFile=None, sep=',', featuresList = None):
		if csvFile==None:
			csvFile=self.nomeDB+".csv"

		if featuresList==None:
			featuresList=self.features

		self.df=pd.read_csv(csvFile, sep=sep, names = featuresList)

	def drop_columns(self, columns):
		for column in columns:
			self.df.drop(columns=[column],inplace=True)

	def merge_music_data(self, musicData):
		self.df=self.df.join(musicData.df)

	def train_test_split(self,targetFeatureName="popularity", testSize=0.2):
		targetDF=self.df[targetFeatureName]
		self.df.drop(columns=[targetFeatureName],inplace=True)
		
		self.xTrain, self.xTest, self.yTrain,self.yTest = train_test_split(self.df, targetDF, test_size=0.2)

def get_prep_mus_data():

	_4mulaFeatureNames=['music_id', 'music_name', 'music_lang', 'art_id','art_name', 'art_rank4Mula', 'main_genre', 'related_genre','musicnn_tags']

	_spotifyBasicAudioFeature=['danceability','energy','key','mode','speechiness','loudness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature']
	_spotifyAudioAnalysisTrack=['num_samples','tempo_confidence','time_signature_confidence','key_confidence','mode_confidence']
	_spotifyAudioAnalysis=['bars','beats','sections','segments','tatums']

	features=_4mulaFeatureNames+['spotify_trackID']+_spotifyBasicAudioFeature+['spotifyAlbum_id']+['release_date'] +['popularity']+['spotifyArt_id']+_spotifyAudioAnalysisTrack+_spotifyAudioAnalysis+['period']+['position']+['mus_rank']+['art_rank']

	musData=music_data_entity("matchSpotify4Mula-metadata",features)
	musData.read_csv()
	
	#Criacao de vetor com as features para remocao
	featuresParaRemover=['position','music_id','art_id','art_name','music_name','related_genre','art_rank4Mula','spotify_trackID','spotifyAlbum_id','spotifyArt_id','art_rank','mus_rank','period','musicnn_tags']
	#Removendo features
	musData.drop_columns(featuresParaRemover)

	#Obtendo dadaos somente do spotify
	features=['spotify_trackID','spotify_artID','totalFollowers','artPopularity']
	spotifyData=music_data_entity("spotifyOnlyFeatures",features)
	spotifyData.read_csv()
	#Criacao de vetor com as features para remocao
	featuresParaRemover=['spotify_trackID','spotify_artID','artPopularity']
	#Removendo features
	spotifyData.drop_columns(featuresParaRemover)

	#Mergeando os dois dfs
	musData.merge_music_data(spotifyData)

	musData.df=musData.df.dropna()

	return musData