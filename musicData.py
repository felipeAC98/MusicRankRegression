import numpy as np
import pandas as pd
import transform

import traceback

class musicData():

	def __init__(self,nomeDB,features):
		self.nomeDB=nomeDB
		self.df=None
		self.features=features

	def readCSV(self, csvFile=None, sep=',', featuresList = None):
		if csvFile==None:
			csvFile=self.nomeDB+".csv"

		if featuresList==None:
			featuresList=self.features

		self.df=pd.read_csv(csvFile, sep=sep, names = featuresList)

	def dropColumns(self, columns):
		for column in columns:
			self.df.drop(columns=[column],inplace=True)

	def mergeMusicData(self, musicData):
		self.df=self.df.join(musicData.df)

def getPrepMusData():

	_4mulaFeatureNames=['music_id', 'music_name', 'music_lang', 'art_id','art_name', 'art_rank4Mula', 'main_genre', 'related_genre','musicnn_tags']

	_spotifyBasicAudioFeature=['danceability','energy','key','mode','speechiness','loudness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature']
	_spotifyAudioAnalysisTrack=['num_samples','tempo_confidence','time_signature_confidence','key_confidence','mode_confidence']
	_spotifyAudioAnalysis=['bars','beats','sections','segments','tatums']

	features=_4mulaFeatureNames+['spotify_trackID']+_spotifyBasicAudioFeature+['spotifyAlbum_id']+['release_date'] +['popularity']+['spotifyArt_id']+_spotifyAudioAnalysisTrack+_spotifyAudioAnalysis+['period']+['position']+['mus_rank']+['art_rank']

	musData=musicData("matchSpotify4Mula-metadata",features)
	musData.readCSV()
	#Criacao de vetor com as features para remocao
	featuresParaRemover=['music_id','art_id','art_name','music_name','related_genre','art_rank4Mula','spotify_trackID','spotifyAlbum_id','spotifyArt_id','art_rank','mus_rank','period','musicnn_tags']
	#Removendo features
	musData.dropColumns(featuresParaRemover)

	#Obtendo dadaos somente do spotify
	features=['spotify_trackID','spotify_artID','totalFollowers','artPopularity']
	spotifyData=musicData("spotifyOnlyFeatures",features)
	spotifyData.readCSV()
	#Criacao de vetor com as features para remocao
	featuresParaRemover=['spotify_trackID','spotify_artID','artPopularity']
	#Removendo features
	spotifyData.dropColumns(featuresParaRemover)

	#Mergeando os dois dfs
	musData.mergeMusicData(spotifyData)

	musData.df=musData.df.dropna()

	return musData.df