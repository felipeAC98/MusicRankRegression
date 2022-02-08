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

#obtendo release time - tempo em meses em que a musica foi lancada
try:
	musicData.df = transform.monthsAfterRelease(musicData.df,'release_date')

except:
	print(' release_time nao encontrada: '+str(traceback.format_exc()))

#removendo algumas amostras para deixar o treinamento inicial mais rapido
#nIndexsToDrop=int(len(musicData.df.index)*0.9)

#indexsToDrop = np.random.choice(musicData.df.index, nIndexsToDrop, replace=False)
#musicData.df = musicData.df.drop(indexsToDrop)	

musicData.train_test_split(targetFeatureName="popularity")

#Tree score
tree_regressor=classes.regressor.tree_regressor(musicData,min_impurity_decrease=0.2)
tree_regressor.fit()
shapValues=shap_values(tree_regressor)
shapValues.tree_explainer("tree_regressor")
shapValues.explainer("tree_regressor")
shapValues.explainer_default("tree_regressor")

#Random forest score
#randon_forest_regressor=classes.regressor.randon_forest_regressor(musicData,min_impurity_decrease=0.001,n_estimators=400)
#randon_forest_regressor.fit()
#shapValues=shap_values(knn_regressor)
#shapValues.tree_explainer()
#shapValues.explainer()
#shapValues.explainer("knn_regressor")