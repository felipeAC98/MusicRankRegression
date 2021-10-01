from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

def useOneHotEncoder(data, feature, featureNamePattern=None, merge=False):

	enc = OneHotEncoder(handle_unknown='ignore')

	#fazendo o encode e ja passando para um df temporario
	enc_df = pd.DataFrame(enc.fit_transform(data[[feature]]).toarray())

	#removendo a feature que esta sofrendo o one hot encoder
	data.drop(columns=[feature],inplace=True)

	#inserindo nomes mais compreensiveis nas features do df criado
	categories=[]
	for categori in enc.get_feature_names():

		if featureNamePattern==None:
			featureName=feature+'-'+str(categori)
		else:
			featureName=featureNamePattern+'-'+str(categori)
		featureName=featureName.replace('x0_','')
		categories.append(featureName)
	
	enc_df.columns =categories

	if merge==False:
		#retornando df com o one hot encode ja incluido
		return data.join(enc_df)

	else:
		#para casos onde as features do novo e antigo df forem as mesmas
		return mergeOneHotEncoder(data, enc_df)

def mergeOneHotEncoder(data, newEncData):
	print("mergeOneHotEncoder")

	for index, row in newEncData.iterrows():
		for rowIndex in row.index:
			#quando for 1, o valor sera substituido 
			if str(row[rowIndex])=='1.0':
				data.loc[index,rowIndex]='1.0'

	return data.replace(np.nan, '0.0')

def splitMusicnnTags(data):

	#criando um novo df que sera mergiado posteriormente
	musicnnTagsDF= pd.DataFrame({'musicnn_tags1':[],'musicnn_tags2':[],'musicnn_tags3':[]})  

	for index, row in data.iterrows():

		musicnnTags=row['musicnn_tags']
		newRow=[]

		#essa feature esta sendo tratada como uma string ao inves de um vetor, vamos passala para 3 features distintas
		musicnnTagsSplitted = musicnnTags.split("'")
		newRow={'musicnn_tags1': musicnnTagsSplitted[1], 'musicnn_tags2':musicnnTagsSplitted[3], 'musicnn_tags3': musicnnTagsSplitted[5]}
		musicnnTagsDF = musicnnTagsDF.append(newRow, ignore_index=True)

	#removendo a musicnn_tags 
	data.drop(columns=['musicnn_tags'],inplace=True)

	return data.join(musicnnTagsDF)

#funcao para obter quantidade de meses que se passaram desde o lancamento da musica
def monthsAfterRelease(data):

	from datetime import datetime

	#criando um novo df que sera mergiado posteriormente
	mesesPosReleaseDF= pd.DataFrame({'release_time':[]})  

	hoje = datetime.date.today()

	mesAtual=hoje.month

	anoAtual=hoje.year

	for index, row in data.iterrows():

		release_date=row['release_date']

		release_date_obj = datetime.strptime(release_date, '%y/%d/%m')

		anoLancamento=release_date_obj.year

		mesLancamento=release_date_obj.month

		mesesPassadosPosLancamento=(anoAtual-anoLancamento)*12

		if(anoAtual!=anoLancamento):
			mesesPassadosPosLancamento=mesAtual-mesLancamento

		else:
			mesesPassadosPosLancamento=mesAtual-mesLancamento

		newRow={'release_time': mesesPassadosPosLancamento}
		mesesPosReleaseDF = mesesPosReleaseDF.append(newRow, ignore_index=True)

	#removendo a musicnn_tags 
	data.drop(columns=['release_date'],inplace=True)

	return data.join(mesesPosReleaseDF)