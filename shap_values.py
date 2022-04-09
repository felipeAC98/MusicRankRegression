from classes.musicData import music_data_entity, get_prep_mus_data
import classes.regressor
from classes.analytics import shap_values
import transform
import numpy as np

def get_music_ID_by_name(musicData,musicName="Bang"):
	musicID = musicData.df.index[musicData.df['music_name'] == musicName].tolist()[0]

	#Removendo atributo dos dfs
	drop_colums(musicData)

	return musicID

def get_music_ID_by_popularity(musicData,popularity=50):

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



musicData=get_prep_mus_data()

print(musicData.df.head())
#Aplicando one hot enconding
#musicData.df = transform.useOneHotEncoder(musicData.df, 'main_genre','genre-')
musicData.df.drop(columns=['main_genre'],inplace=True)

musicData.df = transform.useOneHotEncoder(musicData.df, 'music_lang')
#musicData.drop_columns(['music_lang-pt-br'])
#musicData.drop_columns(['totalFollowers'])
print(musicData.df.head())
musicData.df.drop(columns=['artPopularity'],inplace=True)

#obtendo release time - tempo em meses em que a musica foi lancada
try:
	musicData.df = transform.monthsAfterRelease(musicData.df,'release_date')

except:
	print(' release_time nao encontrada: '+str(traceback.format_exc()))

#removendo algumas amostras para deixar o treinamento inicial mais rapido
#nIndexsToDrop=int(len(musicData.df.index)*0.5)

#indexsToDrop = np.random.choice(musicData.df.index, nIndexsToDrop, replace=False)
#musicData.df = musicData.df.drop(indexsToDrop)	

musicData.train_test_split(targetFeatureName="popularity")
#musicID=get_music_ID(musicData,musicName="Bang")
musicID, nomeMusica=get_music_ID_by_popularity(musicData,popularity=50)

#Tree score
'''
tree_regressor=classes.regressor.tree_regressor(musicData,min_impurity_decrease=1)
tree_regressor.fit()
shap=shap_values(tree_regressor,preShapConfig=False)
#shap.tree_explainer("tree_regressor-noTotFolllower")
#shap.explainer("tree_regressor-example")
#shap.explainer_default("tree_regressor-noTotFolllower")
#shap.decision_plot("tree_regressor-noTotFolllower")
#shap.bar_plot("tree_regressor-testBar")
shap.music_decision_plot("tree_regressor-testMusDec",musID=musicID)

#'''

#Random forest score
#'''
randon_forest_regressor=classes.regressor.randon_forest_regressor(musicData,min_impurity_decrease=0.001,n_estimators=400)
randon_forest_regressor.fit()
shap=shap_values(randon_forest_regressor,preShapConfig=False)
#shap.explainer_default("randon_forest-noTotFolllower")
#shap.decision_plot("randon_forest-100",nSamples=100)
#shap.tree_explainer("randon_forest-noTotFolllower")
#shap.bar_plot("randon_forest-testBar")
shap.music_decision_plot("randon_forest-"+str(nomeMusica),musID=musicID)
#'''

#XGBoost
'''

xgboost_regressor=classes.regressor.xgboost_regressor(musicData,learning_rate=0.3,max_depth=10,subsample=1,n_jobs=4)
xgboost_regressor.fit()
shap=shap_values(xgboost_regressor)
shap.explainer_default("xgboost_regressor-noTotFolllower")
#shap.decision_plot("xgboost_regressor-noTotFolllower")
shap.tree_explainer("xgboost_regressor-noTotFolllower")
#'''

'''
#Force plot
tree_regressor=classes.regressor.tree_regressor(musicData,min_impurity_decrease=1)
tree_regressor.fit()
shap=shap_values(tree_regressor,preShapConfig=False)
shap.force_plot("tree_regressor-test",2)
#'''