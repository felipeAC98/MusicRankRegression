from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor 
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn import tree
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
import graphviz 
import pylab
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.linear_model import LinearRegression

class regressor():

	def __init__(self,musicData):
		self.musicData=musicData
		self.model=None
		self.score=None
		self.MSE=None
		self.RMSE=None

	def fit(self):
		self.model.fit(self.musicData.xTrain, self.musicData.yTrain)

	def get_data(self,isTest=True):
		if isTest==True:
			yPred = self.model.predict(self.musicData.xTest)
			yTrue=self.musicData.yTest
		else:
			yPred = self.model.predict(self.musicData.xTrain)
			yTrue=self.musicData.yTrain

		return yTrue, yPred

	def get_score(self,isTest=True):
		yTrue, yPred=self.get_data(isTest)
		self.score=r2_score(yTrue, yPred)
		return self.score

	def get_MSE_score(self,isTest=True):
		yTrue, yPred=self.get_data(isTest)
		self.MSE=mean_squared_error(yTrue, yPred)
		return self.MSE

	def get_RMSE(self):
		if self.MSE==None:
			self.get_MSE_score()
		self.RMSE=self.MSE**0.5
		return self.RMSE

	def get_MAE_score(self,isTest=True):
		yTrue, yPred=self.get_data(isTest)
		self.MAE=mean_absolute_error(yTrue, yPred)
		return self.MAE

	def get_MAPE(self,isTest=True):
		yTrue, yPred=self.get_data(isTest)
		self.MAPE = np.mean(np.abs((yTrue - yPred) /yTrue)) * 100
		return self.MAPE

	def get_r2_adjusted(self,isTest=True):
		nAmostras=len(self.musicData.df.index)
		nFeatures=len(self.musicData.df.columns)
		r2=self.get_score(isTest)
		r2Adjusted=1-((1-r2)*(nAmostras-1))/(nAmostras-1-nFeatures)
		return r2Adjusted

	def grid_search(self, params,nSplits=3):
		timeSplit = TimeSeriesSplit(n_splits=nSplits)
		self.model= GridSearchCV(estimator = self.model,param_grid=params, n_jobs = -1, verbose = 3)
		self.fit()
		return self.model.best_params_

	def get_grid_best_params(self):
		return self.model.best_params_

	def get_grid_best_score(self):
		return self.model.best_score_

class knn_regressor(regressor):

	def __init__(self, musicData, **params):
		super().__init__(musicData)
		self.model=KNeighborsRegressor(**params) #**despactando o dict para mandar os parametros para a funcao interna
		self.params=params

class linear_regressor(regressor):

	def __init__(self, musicData, **params):
		super().__init__(musicData)
		self.model=LinearRegression(**params) #**despactando o dict para mandar os parametros para a funcao interna
		self.params=params

class tree_regressor(regressor):

	def __init__(self, musicData, **params):
		super().__init__(musicData)
		self.model=tree.DecisionTreeRegressor(**params)
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
		#self.fit()
		self.params=params

class grid_randon_forest_regressor(regressor):

	def __init__(self, musicData, **params):
		super().__init__(musicData)
		self.model=RandomForestRegressor()
		self.params=params
		self.model= GridSearchCV(estimator = self.model,param_grid=params, n_jobs = 5, verbose = 2)
		self.fit()


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
		self.params=params

class grid_xgboost_regressor(regressor):

	def __init__(self, musicData, params):
		super().__init__(musicData)
		self.model=xgb.XGBRegressor()
		self.params=params
		self.model= GridSearchCV(estimator = self.model,param_grid=params, verbose = 2)
		self.fit()

	def get_best_param(self):
		return self.model.best_params_

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