from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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
		#self.plt.title("Distribuicao das musicas por popularidade")
		self.set_figure_size(length=length, height=height)
		sns.displot(x = targetFeature, data =self.musicData.df, kde=True)
		self.plt.xlabel(targetFeature)
		self.plt.ylabel("Quantidade de musicas")
		self.save_figure(targetFeature,"plot_distribution")

	def plot_log_distribution(self,targetFeature="popularity",length=50, height=30):
		#self.plt.title("Distribuicao das musicas por popularidade")
		self.plt.xlabel(targetFeature)
		self.plt.ylabel("Quantidade de musicas")

		sns.histplot(x =targetFeature, data =self.musicData.df, kde=True, log_scale=True)
		self.save_figure(targetFeature,"plot_log_distribution")

	def plot_qtd_musics_by(self,targetFeature,length=50, height=30):
		#self.plt.title("Quantidade de musicas por '"+str(targetFeature)+ "'")
		self.set_figure_size(length=length, height=height)
		sns.set(font_scale = 5)
		sns.countplot(x = targetFeature, data =self.musicData.df)

		#self.plt.xlabel(targetFeature)
		self.plt.ylabel("Quantidade de musicas")


		self.save_figure(targetFeature,"plot_qtd_musics_by")

	def plot_qtd_musics_by_genre(self,ident="main_genre",length=50, height=30,fontScale=None,hideXLabels=True,showValues=False):
		#self.plt.title("Quantidade de musicas por '"+str(targetFeature)+ "'")
		targetFeature="main_genre"
		self.set_figure_size(length=length, height=height)

		if fontScale!=None:
			sns.set(font_scale = fontScale)

		countplot=sns.countplot(x = targetFeature, data =self.musicData.df)

		if hideXLabels:
			countplot.set(xticklabels=[])

		#self.plt.xlabel(targetFeature)
		self.plt.ylabel("Quantidade de musicas")

		if showValues==True:
			for p in countplot.patches:
				x=p.get_bbox().get_points()[:,0]
				y=p.get_bbox().get_points()[1,1]
				countplot.annotate(str(y),(x.mean(), y)) # set the alignment of the text

		self.save_figure(ident,"plot_qtd_musics_by")

	def plot_qtd_musics_by_release_time(self,ident="release_time",length=50, height=30,fontScale=None,showValues=False):
		#self.plt.title("Quantidade de musicas por '"+str(targetFeature)+ "'")
		targetFeature="release_time"
		self.set_figure_size(length=length, height=height)

		if fontScale!=None:
			sns.set(font_scale = fontScale)

		countplot=sns.countplot(x = targetFeature, data =self.musicData.df)

		self.plt.xlabel("Meses passados desde o lançamento")
		self.plt.ylabel("Quantidade de musicas")

		myXticks = countplot.get_xticks()

		self.plt.xticks(
          [myXticks[0], myXticks[-1]], 
          [myXticks[0],myXticks[-1]], visible=True, rotation="horizontal")
		self.save_figure(ident,"plot_qtd_musics_by")

	def plot_polar_graph(self,targetFeature='main_genre'):
		fig = px.line_polar(self.musicData.df, r=targetFeature, theta=targetFeature, line_close=True)
		fig.write_image("newPlots/"+str(targetFeature)+"_plot_polar_graph.jpeg")

	def plot_boxplot(self,targetFeature1,targetFeature2,length=20, height=30):

		#f, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=False)
		#sns.despine(left=True)
		self.set_figure_size(length=length, height=height)
		sns.set(font_scale = 3)
		sns.boxplot(targetFeature1, targetFeature2, data = self.musicData.df)
		self.save_figure(str(targetFeature1)+'_by_'+str(targetFeature2),"plot_boxplot")

	def plot_corr_matrix(self):
		self.set_figure_size(length=23, height=21)
		corrMatrix = self.musicData.df.corr()
		sns.heatmap(corrMatrix, annot=True)
		#plt.show()
		self.save_figure("correlation","plot_corr_matrix")