from classes.musicData import music_data_entity, get_prep_mus_data
import classes.regressor
from classes.analytics import shap_values
import transform
import numpy as np

musicData=get_prep_mus_data()

print(musicData.df.head())
#Aplicando one hot enconding
musicData.df = transform.useOneHotEncoder(musicData.df, 'main_genre','genre-')
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

#Tree score
'''
tree_regressor=classes.regressor.tree_regressor(musicData,min_impurity_decrease=1)
tree_regressor.fit()
shap=shap_values(tree_regressor)
shap.tree_explainer("tree_regressor-noTotFolllower")
#shap.explainer("tree_regressor-example")
shap.explainer_default("tree_regressor-noTotFolllower")
shap.decision_plot("tree_regressor-noTotFolllower")
#'''

#Random forest score
#'''
randon_forest_regressor=classes.regressor.randon_forest_regressor(musicData,min_impurity_decrease=0.001,n_estimators=400)
randon_forest_regressor.fit()
shap=shap_values(randon_forest_regressor)
#shap.explainer_default("randon_forest-noTotFolllower")
shap.decision_plot("randon_forest-100",nSamples=100)
#shap.tree_explainer("randon_forest-noTotFolllower")
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