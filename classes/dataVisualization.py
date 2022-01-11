from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor 
import matplotlib.pyplot as plt
import seaborn as sns

class data_visualization():

	def __init__(self,musicData,path="newPlots/"):
		self.musicData=musicData
		self.plt=plt
		self.path=path
		self.set_figure_size()

	def set_figure_size(self,length=12, height=8):
		self.lenth=length
		self.height=height
		self.plt.figure(figsize=(length, height))

	def save_figure(self,figID,figureName):
		self.plt.savefig(str(self.path)+str(figID)+'-'+figureName)

	def plot_distribution(self,targetFeature="popularity",length=50, height=30):
		self.plt.title("Distribuicao das musicas por popularidade")
		self.plt.xlabel(targetFeature)
		self.plt.ylabel("Quantidade de musicas")

		sns.histplot(x = targetFeature, data =self.musicData.df, kde=True)
		self.save_figure(targetFeature,"plot_distribution")

	def plot_qtd_musics_by(self,targetFeature,length=50, height=30):
		self.plt.title("Quantidade de musicas por '"+str(targetFeature)+ "'")
		self.plt.xlabel(targetFeature)
		self.plt.ylabel("Quantidade de musicas")
		self.set_figure_size(length=length, height=height)

		sns.countplot(x = targetFeature, data =self.musicData.df)
		self.save_figure(targetFeature,"plot_qtd_musics_by")

	def plot_boxplot(self,targetFeature1,targetFeature2,length=20, height=30):

		#f, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=False)
		#sns.despine(left=True)
		self.set_figure_size(length=length, height=height)
		sns.boxplot(targetFeature1, targetFeature2, data = self.musicData.df)
		self.save_figure(str(targetFeature1)+'_by_'+str(targetFeature2),"plot_boxplot")

	def plot_corr_matrix(self):
		self.set_figure_size(length=20, height=20)
		corrMatrix = self.musicData.df.corr()
		sns.heatmap(corrMatrix, annot=True)
		#plt.show()
		self.save_figure("correlation","plot_corr_matrix")