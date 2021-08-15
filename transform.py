from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def useOneHotEncoder(data, feature):

	enc = OneHotEncoder(handle_unknown='ignore')

	#fazendo o encode e ja passando para um df temporario
	enc_df = pd.DataFrame(enc.fit_transform(data[[feature]]).toarray())

	#removendo a feature que esta sofrendo o one hot encoder
	data.drop(columns=[feature],inplace=True)

	#inserindo nomes mais compreensiveis nas features do df criado
	categories=[]
	for categori in enc.get_feature_names():

		featureName=feature+'-'+str(categori)
		featureName=featureName.replace('x0_','')
		categories.append(featureName)
	
	enc_df.columns =categories

	#retornando df com o one hot encode ja incluido
	return data.join(enc_df)