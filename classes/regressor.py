from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor 
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn import tree
from sklearn.neural_network import MLPRegressor
import graphviz 
import pylab
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense

class regressor():

	def __init__(self,musicData):
		self.musicData=musicData
		self.model=None
		self.score=None

	def fit(self):
		self.model.fit(self.musicData.xTrain, self.musicData.yTrain)

	def get_score(self):
		yPred = self.model.predict(self.musicData.xTest)
		self.score=r2_score(self.musicData.yTest, yPred)
		return self.score

	def get_MSE_score(self):
		yPred = self.model.predict(self.musicData.xTest)
		self.MSE=mean_squared_error(self.musicData.yTest, yPred)
		return self.MSE

	def get_MAE_score(self):
		yPred = self.model.predict(self.musicData.xTest)
		self.MAE=mean_absolute_error(self.musicData.yTest, yPred)
		return self.MAE

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

class xgboost_regressor(regressor):

	def __init__(self, musicData, **params):
		super().__init__(musicData)
		self.model= xgb.XGBRegressor(**params)
		self.fit()
		self.params=params

class mlp_regressor(regressor):

	def __init__(self, musicData, **params):
		super().__init__(musicData)
		self.model= MLPRegressor(**params)
		self.fit()
		self.params=params

class adaboost_regressor(regressor):

	def __init__(self, musicData, **params):
		super().__init__(musicData)
		self.model= AdaBoostRegressor(**params)
		self.fit()
		self.params=params

class keras_sequential_regressor(regressor):

	def __init__(self, musicData, **params):
		super().__init__(musicData)
		self.model= self.keras_sequential_model(**params)
		self.fit()
		self.params=params

	def keras_sequential_model(self,**params):
		inputDim=len(self.musicData.df.columns)
		firstLayer=inputDim

		model=Sequential()
		model.add(Dense(firstLayer, input_dim=inputDim, kernel_initializer='normal', activation='relu'))
		model.add(Dense(firstLayer*2,  kernel_initializer='normal', activation='relu'))
		model.add(Dense(firstLayer*4,  kernel_initializer='normal', activation='relu'))
		model.add(Dense(firstLayer, kernel_initializer='normal', activation='relu'))
		model.add(Dense(firstLayer, kernel_initializer='normal', activation='relu'))
		#model.add(Dense(firstLayer, kernel_initializer='normal', activation='relu'))
		model.add(Dense(int(firstLayer/2), kernel_initializer='normal', activation='relu'))
		model.add(Dense(int(firstLayer/8), kernel_initializer='normal', activation='relu'))
		model.add(Dense(1, kernel_initializer='normal'))
		model.compile(loss='mean_squared_error', optimizer='Adamax')

		return model

	def fit(self):

		#Epoch: One pass through all of the rows in the training dataset.
		#Batch: One or more samples considered by the model within an epoch before weights are updated.
		self.model.fit(self.musicData.xTrain, self.musicData.yTrain, epochs=600, batch_size=700)

	def get_accuracy(self):
		_, accuracy = self.model.evaluate(self.musicData.xTest,self.musicData.yTest) 
		return accuracy

	def get_F1score(self):
		yPred = self.model.predict(self.musicData.xTest)
		self.F1score=f1_score(self.musicData.yTest, yPred, average=None)
		return self.score