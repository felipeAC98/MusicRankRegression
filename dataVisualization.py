import numpy as np
import pandas as pd
import transform
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

#Obtendo dados
matchSpotify4Mula =("matchSpotify4Mula.csv")
features=['music_id', 'music_name', 'music_lang', 'art_id','art_name', 'art_rank', 'main_genre', 'related_genre','musicnn_tags','danceability','energy','key','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature','mus_rank']

X = pd.read_csv(matchSpotify4Mula, sep=',', names = features)

X.drop(columns=['music_id'],inplace=True)
X.drop(columns=['music_lang'],inplace=True)
X.drop(columns=['art_id'],inplace=True)
X.drop(columns=['art_name'],inplace=True)
X.drop(columns=['music_name'],inplace=True)
X.drop(columns=['related_genre'],inplace=True)

Y = X['mus_rank']
print(Y)
#X=X.drop(columns=['mus_rank'])

print(X.head())

#Aplicando one hot enconding
#X = transform.useOneHotEncoder(X, 'main_genre')
#X = transform.useOneHotEncoder(X, 'related_genre') #precisa corrigir, nao esta dividindo o array em diferentes subcategorias
print(X.head())

#Splitando a feature de musicnn_tags
X = transform.splitMusicnnTags(X)
print(X.head())

X = transform.useOneHotEncoder(X, 'musicnn_tags1', 'musicnn_tags')
print(X.head())

X = transform.useOneHotEncoder(X, 'musicnn_tags2', 'musicnn_tags', merge=True)
print(X.head())

X = transform.useOneHotEncoder(X, 'musicnn_tags3', 'musicnn_tags', merge=True)
print(X.head())

print(X.columns)

#Distribuicao das musicas pelo rank

plt.figure(figsize=(12, 8))
plt.title("Distribuicao das musicas por rank")
plt.xlabel("Rank")
plt.ylabel("Quantidade de musicas")
sns.histplot(x = 'mus_rank', data =X, kde=True)
plt.savefig('plots/histplot.png')

#mesma distribuicao porem na scala logaritma

sns.histplot(x = 'mus_rank', data =X, kde=True, log_scale=True)
plt.savefig('plots/histplot_log.png')

#Relaçao entre features

df_principalFeatures=X[['mus_rank', 'main_genre', 'danceability','energy','mode','speechiness','acousticness','instrumentalness','liveness']]

sns.pairplot(df_principalFeatures, kind="scatter", hue="mus_rank") 
plt.savefig('plots/pairplot.png')
