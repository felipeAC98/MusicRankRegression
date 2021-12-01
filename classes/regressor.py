from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor 

class regressor():

	def __init__(self,musicData):
		self.musicData=musicData

	def train_test_split(self,targetFeatureName="popularity", testSize=0.2):

		targetDF=self.musicData.df[targetFeatureName]
		self.musicData.df.drop(columns=[targetFeatureName],inplace=True)
		
		self.xTrain, self.xTest, self.yTrain,self.yTest = train_test_split(self.musicData.df, targetDF, test_size=0.2)

	def knn_score(self,nNeighbors=15):
		KNNR = KNeighborsRegressor(n_neighbors=nNeighbors)
		KNNR.fit(self.xTrain, self.yTrain)
		score=KNNR.score(self.xTest,self.yTest)
		return score

	def random_forest_score(self,min_impurity_decrease=0.0005, random_state=0):
		randFr = RandomForestRegressor(min_impurity_decrease=min_impurity_decrease, random_state=random_state)
		randFr.fit(self.xTrain, self.yTrain)
		score=randFr.score(self.xTest,self.yTest)
		return score

	def extra_tree_score(self,min_impurity_decrease=0.0005, random_state=0):
		extTree = ExtraTreesRegressor(min_impurity_decrease=min_impurity_decrease, random_state=random_state)
		extTree.fit(self.xTrain, self.yTrain)
		score=extTree.score(self.xTest,self.yTest)
		return score