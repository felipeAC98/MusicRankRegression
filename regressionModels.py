def kNeighborsRegressorScore(xTrain, yTrain, xTest, yTest, nNeighbors):
	from sklearn import neighbors

	knn = neighbors.KNeighborsRegressor(nNeighbors)

	knn.fit(xTrain, yTrain)

	preResult=knn.predict(xTest)

	score = knn.score(xTest, yTest)

	return score

#def kNeighborsKneeCheck(xTrain, yTrain, xTest, yTest, initialValue, finalValue):