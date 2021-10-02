import numpy as np
import pandas as pd
import transform
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import traceback

nomeDB="matchSpotify4Mula-tiny"

#Obtendo dados
matchSpotify4Mula =(nomeDB+".csv")
#features=['music_id', 'music_name', 'music_lang', 'art_id','art_name', 'art_rank', 'main_genre', 'related_genre','musicnn_tags','danceability','energy','key','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature','period','position','mus_rank']
features=['music_id', 'music_name', 'music_lang', 'art_id','art_name', 'art_rank', 'main_genre', 'related_genre','musicnn_tags','danceability','energy','key','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature','release_date','period','position','mus_rank']

X = pd.read_csv(matchSpotify4Mula, sep=',', names = features)

X.drop(columns=['music_id'],inplace=True)
X.drop(columns=['music_lang'],inplace=True)
X.drop(columns=['art_id'],inplace=True)
X.drop(columns=['art_name'],inplace=True)
X.drop(columns=['music_name'],inplace=True)
X.drop(columns=['related_genre'],inplace=True)
X.drop(columns=['period'],inplace=True)

#X.drop(columns=['musicnn_tags'],inplace=True)

Y = X['mus_rank']
print(Y)
#X=X.drop(columns=['mus_rank'])

print(X.head())

#Aplicando one hot enconding
#X = transform.useOneHotEncoder(X, 'related_genre') #precisa corrigir, nao esta dividindo o array em diferentes subcategorias
print(X.head())

try:
	#Splitando a feature de musicnn_tags
	X = transform.splitMusicnnTags(X)
	print(X.head())

	#o segundo parametro é o parametro a ser utilizado o oneHot e o terceiro é o index dos novos campos que vao ser criados
	X = transform.useOneHotEncoder(X, 'musicnn_tags1', 'musicnn_tags')
	print(X.head())

	X = transform.useOneHotEncoder(X, 'musicnn_tags2', 'musicnn_tags', merge=True)
	print(X.head())

	X = transform.useOneHotEncoder(X, 'musicnn_tags3', 'musicnn_tags', merge=True)
	print(X.head())

except:
	print(' musicnn tags nao encontrada: '+str(traceback.format_exc()))

#removendo musicas com posicoes muito altas
X = X.drop(X[X.position > 5000].index) 

print(X.head())

print(X.columns)

#Trabalhando na feature main genre, para a versao final o OneHotEncoder sera usado

X = transform.useOneHotEncoder(X, 'main_genre')
#Passando genero principal de categorias string para numerica
#X['main_genre']=pd.factorize(X['main_genre'])[0]

print(X.head())

print(X.columns)

principaisColunas=['position', 'danceability','energy','mode','speechiness','acousticness','instrumentalness','liveness']

#obtendo release time
try:
	X = transform.monthsAfterRelease(X,'release_time')
	print(X.head())

	print(X.columns)

	principaisColunas.append('release_time')

except:
	print(' release_time nao encontrada: '+str(traceback.format_exc()))

df_principalFeatures=X[principaisColunas]

#Distribuicao das musicas pelo rank

plt.figure(figsize=(12, 8))
plt.title("Distribuicao das musicas por rank")
plt.xlabel("Rank")
plt.ylabel("Quantidade de musicas")
sns.histplot(x = 'mus_rank', data =X, kde=True)
plt.savefig('plots/'+nomeDB+'-mus_rank-histplot.png')

#Distribuicao das musicas pela posicao

plt.figure(figsize=(12, 8))
plt.title("Distribuicao das musicas por posicao")
plt.xlabel("Rank")
plt.ylabel("Quantidade de musicas")
sns.histplot(x = 'position', data =X, kde=True)
plt.savefig('plots/'+nomeDB+'-position-histplot.png')

#mesma distribuicao porem na scala logaritma

sns.histplot(x = 'position', data =X, kde=True, log_scale=True)
plt.savefig('plots/'+nomeDB+'-position-histplot_log.png')


#mesma distribuicao porem na scala logaritma

sns.histplot(x = 'mus_rank', data =X, kde=True, log_scale=True)
plt.savefig('plots/'+nomeDB+'-mus_rank-histplot_log.png')

#Relaçao entre features

sns.pairplot(df_principalFeatures, kind="scatter", hue="position", palette="rocket") 
plt.savefig('plots/'+nomeDB+'-position-cat_pairplot.png')

g=sns.pairplot(df_principalFeatures, diag_kind="kde") 
g.map_lower(sns.kdeplot, levels=4, color=".2")
plt.savefig('plots/'+nomeDB+'-position-level_pairplot.png')

sns.pairplot(df_principalFeatures, kind="kde") 
plt.savefig('plots/'+nomeDB+'-position-levelFull_pairplot.png')