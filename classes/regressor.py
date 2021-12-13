from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor 
from sklearn import tree
import graphviz 
import pylab

class regressor():

	def __init__(self,musicData):
		self.musicData=musicData
		self.model=None
		self.score=None

	def fit(self):
		self.model.fit(self.musicData.xTrain, self.musicData.yTrain)

	def get_score(self):
		self.score=self.model.score(self.musicData.xTest,self.musicData.yTest)
		return self.score
		
class knn_regressor(regressor):

	def __init__(self, musicData, **params):
		super().__init__(musicData)
		self.model=KNeighborsRegressor(**params) #**despactando o dict para mandar os parametros para a funcao interna
		self.fit()
		self.params=params

class tree_regressor(regressor):

	def __init__(self, musicData, **params):
		super().__init__(musicData)
		self.model=tree.DecisionTreeRegressor(**params)
		self.fit()
		self.params=params

	def plot_tree(self):
		dot_data = tree.export_graphviz(self.model, out_file=None, 
                    feature_names= self.musicData.df.columns,  
                    filled=True, rounded=True,  
                    special_characters=True)  

		graph = graphviz.Source(dot_data) 
		graph.render("decision_tree") 
		pylab.savefig('decisionTree.png')

class randon_forest_regressor(regressor):

    def __init__(self, musicData, **params):
        super().__init__(musicData)
        self.model=RandomForestRegressor(**params)
       	self.fit()
        self.params=params

class randon_extra_tree_regressor(regressor):

    def __init__(self, musicData, **params):
        super().__init__(musicData)
        self.model=ExtraTreesRegressor(**params)
       	self.fit()
        self.params=params