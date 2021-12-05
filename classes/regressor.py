from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor 

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