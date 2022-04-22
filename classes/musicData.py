import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import traceback
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder

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

	def normalize(self,targetFeatureName="popularity"):
		#Normalizando dados
		for column in self.df.columns:
			scaler = MinMaxScaler()
			if str(column) != (targetFeatureName):
				self.df[[column]] = scaler.fit_transform(self.df[[column]])

	def useGetDumies(self, feature, featureNamePattern=None, merge=False):

		enc_df = pd.get_dummies(self.df[feature])

		#removendo a feature que esta sofrendo o one hot encoder
		self.df.drop(columns=[feature],inplace=True)

		#inserindo nomes mais compreensiveis nas features do df criado
		categories=[]
		for categori in enc_df.columns:

			if featureNamePattern==None:
				featureName=feature+'-'+str(categori)
			else:
				featureName=featureNamePattern+'-'+str(categori)
			featureName=featureName.replace('x0_','')
			categories.append(featureName)

		enc_df.columns =categories

		if merge==False:
			#retornando df com o one hot encode ja incluido
			return self.df.join(enc_df)

		else:
			#para casos onde as features do novo e antigo df forem as mesmas
			return mergeOneHotEncoder(self.df, enc_df)

	def useOneHotEncoder(self, feature, featureNamePattern=None, merge=False):

		enc = OneHotEncoder(handle_unknown='ignore')

		encArray=enc.fit_transform(self.df[[feature]]).toarray()

		#fazendo o encode e ja passando para um df temporario
		enc_df = pd.DataFrame(encArray,columns=enc.get_feature_names_out())

		#removendo a feature que esta sofrendo o one hot encoder
		self.df.drop(columns=[feature],inplace=True)

		if merge==False:
			#retornando df com o one hot encode ja incluido
			self.df=self.df.join(enc_df)

		else:
			#para casos onde as features do novo e antigo df forem as mesmas
			self.df= mergeOneHotEncoder(self.df, enc_df)

	def mergeOneHotEncoder(self, newEncData):
		print("mergeOneHotEncoder")

		for index, row in newEncData.iterrows():
			for rowIndex in row.index:
				#quando for 1, o valor sera substituido 
				if str(row[rowIndex])=='1.0':
					self.df.loc[index,rowIndex]='1.0'

		return self.df.replace(np.nan, '0.0')

	def splitMusicnnTags(self):

		#criando um novo df que sera mergiado posteriormente
		musicnnTagsDF= pd.DataFrame({'musicnn_tags1':[],'musicnn_tags2':[],'musicnn_tags3':[]})  

		for index, row in self.df.iterrows():

			musicnnTags=row['musicnn_tags']
			newRow=[]

			#essa feature esta sendo tratada como uma string ao inves de um vetor, vamos passala para 3 features distintas
			musicnnTagsSplitted = musicnnTags.split("'")
			newRow={'musicnn_tags1': musicnnTagsSplitted[1], 'musicnn_tags2':musicnnTagsSplitted[3], 'musicnn_tags3': musicnnTagsSplitted[5]}
			musicnnTagsDF = musicnnTagsDF.append(newRow, ignore_index=True)

		#removendo a musicnn_tags 
		self.df.drop(columns=['musicnn_tags'],inplace=True)

		return self.df.join(musicnnTagsDF)

	#funcao para obter quantidade de meses que se passaram desde o lancamento da musica
	def monthsAfterRelease(self, newFeatureName='release_time'):

		from datetime import datetime

		#criando um novo df que sera mergiado posteriormente
		mesesPosReleaseDF= []  

		hoje = datetime.today()

		mesAtual=hoje.month

		anoAtual=hoje.year

		for index, row in self.df.iterrows():

			release_date=row['release_date']

			try:
				release_date_obj = datetime.strptime(release_date, '%Y-%m-%d')
			except:
				try:
					release_date_obj = datetime.strptime(release_date, '%Y')
				except:
					release_date_obj = datetime.strptime(release_date, '%Y-%m')

			anoLancamento=release_date_obj.year

			mesLancamento=release_date_obj.month

			mesesPassadosPosLancamento=(anoAtual-anoLancamento)*12

			if(anoAtual!=anoLancamento):
				mesesPassadosPosLancamento=mesesPassadosPosLancamento+mesAtual-mesLancamento

			else:
				mesesPassadosPosLancamento=mesesPassadosPosLancamento+mesAtual-mesLancamento
			mesesPosReleaseDF.append(mesesPassadosPosLancamento)

		#removendo a musicnn_tags 
		self.df.drop(columns=['release_date'],inplace=True)  
		self.df.insert(2,newFeatureName,mesesPosReleaseDF)
		
		return self.df

	def recountColumn(self, column):
		#ordernando os valores
		self.df=self.df.sort_values(by=column, ascending=True)

		newValues=[]
		i=0

		for index, row in self.df.iterrows():

			newValues.append(i)
			i+=1

		self.df.drop(columns=column,inplace=True)  
		self.df.insert(0,column,newValues)
		print(self.df.head())
		return self.df

def get_prep_mus_data():

	_4mulaFeatureNames=['music_id', 'music_name', 'music_lang', 'art_id','art_name', 'art_rank4Mula', 'main_genre', 'related_genre','musicnn_tags']

	_spotifyBasicAudioFeature=['danceability','energy','key','mode','speechiness','loudness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature']
	_spotifyAudioAnalysisTrack=['num_samples','tempo_confidence','time_signature_confidence','key_confidence','mode_confidence']
	_spotifyAudioAnalysis=['bars','beats','sections','segments','tatums']

	features=_4mulaFeatureNames+['spotify_trackID']+_spotifyBasicAudioFeature+['spotifyAlbum_id']+['release_date'] +['popularity']+['spotifyArt_id']+_spotifyAudioAnalysisTrack+_spotifyAudioAnalysis+['period']+['position']+['mus_rank']+['art_rank']

	musData=music_data_entity("data/matchSpotify4Mula-metadata",features)
	musData.read_csv()
	
	#Criacao de vetor com as features para remocao
	featuresParaRemover=['position','music_id','art_id','art_name','related_genre','art_rank4Mula','spotify_trackID','spotifyAlbum_id','spotifyArt_id','art_rank','mus_rank','period','musicnn_tags']
	#Removendo features
	musData.drop_columns(featuresParaRemover)

	#Obtendo dadaos somente do spotify
	features=['spotify_trackID','spotify_artID','totalFollowers','artPopularity']
	spotifyData=music_data_entity("data/spotifyOnlyFeatures",features)
	spotifyData.read_csv()
	#Criacao de vetor com as features para remocao
	featuresParaRemover=['spotify_trackID','spotify_artID']
	#Removendo features
	spotifyData.drop_columns(featuresParaRemover)

	#Mergeando os dois dfs
	musData.merge_music_data(spotifyData)

	musData.df=musData.df.dropna()

	return musData

def get_prep_mus_vis_data():

	_4mulaFeatureNames=['music_id', 'music_name', 'music_lang', 'art_id','art_name', 'art_rank4Mula', 'main_genre', 'related_genre','musicnn_tags']

	_spotifyBasicAudioFeature=['danceability','energy','key','mode','speechiness','loudness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature']
	_spotifyAudioAnalysisTrack=['num_samples','tempo_confidence','time_signature_confidence','key_confidence','mode_confidence']
	_spotifyAudioAnalysis=['bars','beats','sections','segments','tatums']

	features=_4mulaFeatureNames+['spotify_trackID']+_spotifyBasicAudioFeature+['spotifyAlbum_id']+['release_date'] +['popularity']+['spotifyArt_id']+_spotifyAudioAnalysisTrack+_spotifyAudioAnalysis+['period']+['position']+['mus_rank']+['art_rank']

	musData=music_data_entity("matchSpotify4Mula-metadata",features)
	musData.read_csv()
	
	#Criacao de vetor com as features para remocao
	featuresParaRemover=['position','music_id','art_id','art_name','music_name','related_genre','art_rank4Mula','spotify_trackID','spotifyAlbum_id','spotifyArt_id','art_rank','period','musicnn_tags']
	#Removendo features
	musData.drop_columns(featuresParaRemover)

	#Obtendo dadaos somente do spotify
	features=['spotify_trackID','spotify_artID','totalFollowers','artPopularity']
	spotifyData=music_data_entity("spotifyOnlyFeatures",features)
	spotifyData.read_csv()
	#Criacao de vetor com as features para remocao
	featuresParaRemover=['spotify_trackID','spotify_artID']
	#Removendo features
	spotifyData.drop_columns(featuresParaRemover)

	#Mergeando os dois dfs
	musData.merge_music_data(spotifyData)

	musData.df=musData.df.dropna()

	return musData