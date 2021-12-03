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

	def plot_distribution(self,targetFeature="popularity",**histplot):
		self.plt.title("Distribuicao das musicas por posicao")
		self.plt.xlabel(targetFeature)
		self.plt.ylabel("Quantidade de musicas")

		sns.histplot(x = targetFeature, data =self.musicData.df, kde=True)
		self.save_figure(targetFeature,"plot_distribution")

	def plot_qtd_musics_by(self,targetFeature):
		self.plt.title("Quantidade de musicas por '"+str(targetFeature)+ "'")
		self.plt.xlabel(targetFeature)
		self.plt.ylabel("Quantidade de musicas")
		self.plt.figure(figsize=(50, 30))

		sns.countplot(x = targetFeature, data =self.musicData.df)
		self.save_figure(targetFeature,"plot_qtd_musics_by")
