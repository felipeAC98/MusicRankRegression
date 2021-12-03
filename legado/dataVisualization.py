import numpy as np
import pandas as pd
import transform
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import traceback

#Obtendo dados
nomeDB="matchSpotify4Mula-metadata"
matchSpotify4Mula =(nomeDB+".csv")

_4mulaFeatureNames=['music_id', 'music_name', 'music_lang', 'art_id','art_name', 'art_rank4Mula', 'main_genre', 'related_genre','musicnn_tags']

_spotifyBasicAudioFeature=['danceability','energy','key','mode','speechiness','loudness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature']
_spotifyAudioAnalysisTrack=['num_samples','tempo_confidence','time_signature_confidence','key_confidence','mode_confidence']
_spotifyAudioAnalysis=['bars','beats','sections','segments','tatums']

features=_4mulaFeatureNames+['spotify_trackID']+_spotifyBasicAudioFeature+['spotifyAlbum_id']+['release_date'] +['popularity']+['spotifyArt_id']+_spotifyAudioAnalysisTrack+_spotifyAudioAnalysis+['period']+['position']+['mus_rank']+['art_rank']


X = pd.read_csv(matchSpotify4Mula, sep=',', names = features)
print(X.head())

#Diminuindo dataset
#X = X.drop(X[X.position >30000].index) 

#Remocoes de features pertencentes ao _4mulaFeatureNames
X.drop(columns=['music_id'],inplace=True)
X.drop(columns=['art_id'],inplace=True)
#X.drop(columns=['art_name'],inplace=True)
X.drop(columns=['music_name'],inplace=True)
X.drop(columns=['related_genre'],inplace=True)
X.drop(columns=['art_rank4Mula'],inplace=True)

#Remocoes de features pertencentes ao _spotifyAudioAnalysisTrack
#for featureName in _spotifyAudioAnalysisTrack:
#	X.drop(columns=featureName,inplace=True)

#Remocoes de features gerais
X.drop(columns=['spotify_trackID'],inplace=True)
X.drop(columns=['spotifyAlbum_id'],inplace=True)
X.drop(columns=['spotifyArt_id'],inplace=True)

#Remocoes de features vagalume
X.drop(columns=['art_rank'],inplace=True)
X.drop(columns=['mus_rank'],inplace=True)
X.drop(columns=['period'],inplace=True)

X.drop(columns=['musicnn_tags'],inplace=True)
print("Total de amostras: "+str(len(X.index)))

X=X.dropna()

#Aplicando one hot enconding
X = transform.useOneHotEncoder(X, 'main_genre','genre-')
X = transform.useOneHotEncoder(X, 'music_lang')

#Recontando as posicoes das musicas, para ficar um valor continuo
#X=transform.recountColumn(X,'position')
#X=transform.recountColumn(X,'popularity')

#obtendo release time - tempo em meses em que a musica foi lancada
try:
	X = transform.monthsAfterRelease(X,'release_date')

except:
	print(' release_time nao encontrada: '+str(traceback.format_exc()))

principaisColunas=['popularity', 'danceability','energy','mode','speechiness','acousticness','instrumentalness','liveness']

df_principalFeatures=X[principaisColunas]

#Distribuicao das musicas pela popularity

plt.figure(figsize=(12, 8))
plt.title("Distribuicao das musicas por posicao")
plt.xlabel("Rank")
plt.ylabel("Quantidade de musicas")
sns.histplot(x = 'popularity', data =X, kde=True)
plt.savefig('plots/'+nomeDB+'-popularity-histplot.png')

#mesma distribuicao porem na scala logaritma

#sns.histplot(x = 'popularity', data =X, kde=True, log_scale=True)
#plt.savefig('plots/'+nomeDB+'-popularity-histplot_log.png')

#Relaçao entre features
sns.pairplot(df_principalFeatures, kind="scatter", hue="popularity", palette="rocket") 
plt.savefig('plots/'+nomeDB+'-popularity-cat_pairplot.png')

g=sns.pairplot(df_principalFeatures, diag_kind="kde") 
g.map_lower(sns.kdeplot, levels=4, color=".2")
plt.savefig('plots/'+nomeDB+'-popularity-level_pairplot.png')

sns.pairplot(df_principalFeatures, kind="kde") 
plt.savefig('plots/'+nomeDB+'-popularity-levelFull_pairplot.png')

cmap = sns.cubehelix_palette(as_cmap=True)
num_features = len(df_principalFeatures.columns)
fig,ax = plt.subplots(num_features, num_features, figsize=(50,50))
for axi, i in zip(ax, df_principalFeatures.columns):
    print(i)
    for axj, j in zip(axi, df_principalFeatures.columns):
            axj.scatter(x=df_principalFeatures[i],y=df_principalFeatures[j],c=df_principalFeatures['popularity'],s=50 , cmap=cmap)

plt.savefig('plots/'+nomeDB+'-popularity-cat_pairplot2.png')
