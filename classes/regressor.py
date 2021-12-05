from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor 

class regressor():

	def __init__(self,musicData):
		self.musicData=musicData

	def knn_score(self,nNeighbors=15):
		KNNR = KNeighborsRegressor(n_neighbors=nNeighbors)
		KNNR.fit(self.musicData.xTrain, self.musicData.yTrain)
		score=KNNR.score(self.musicData.xTest,self.musicData.yTest)
		return score

	def random_forest_score(self,min_impurity_decrease=0.0005, random_state=0):
		randFr = RandomForestRegressor(min_impurity_decrease=min_impurity_decrease, random_state=random_state)
		randFr.fit(self.musicData.xTrain, self.musicData.yTrain)
		score=randFr.score(self.musicData.xTest,self.musicData.yTest)
		return score

	def extra_tree_score(self,min_impurity_decrease=0.0005, random_state=0):
		extTree = ExtraTreesRegressor(min_impurity_decrease=min_impurity_decrease, random_state=random_state)
		extTree.fit(self.musicData.xTrain, self.musicData.yTrain)
		score=extTree.score(self.musicData.xTest,self.musicData.yTest)
		return score