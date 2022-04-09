import shap
import matplotlib.pyplot as plt

class shap_values():

	def __init__(self,regressor,dataPercent=1,preShapConfig=True):
		self.regressor=regressor
		self.dataPercent=dataPercent
		self.preShapConfig=preShapConfig

		if self.preShapConfig==True:
			self.set_explainer_config(dataPercent=self.dataPercent)

	def set_explainer_config(self,dataPercent=1,musID=None):
		self.explainer=shap.Explainer(self.regressor.get_model())
		print("set_explainer_config")
		if musID==None:
			self.nSamples=int(len(self.regressor.musicData.df.index)*dataPercent)
			self.shapValues=self.explainer.shap_values(self.regressor.musicData.xTest)[:self.nSamples]
		else:
			self.shapValues=self.explainer.shap_values(self.regressor.musicData.xTest.iloc[musID])

	def tree_explainer(self,name):
		shap_values = shap.TreeExplainer(self.regressor.model).shap_values(self.regressor.musicData.xTest)[:self.nSamples]
		#shap.summary_plot(shap_values, self.regressor.musicData.xTrain, plot_type="bar")		
		f = plt.figure()
		shap.summary_plot(shap_values , self.regressor.musicData.xTest, plot_type="bar")
		f.savefig("newPlots/"+str(name)+"-tree_explainer.png", bbox_inches='tight', dpi=600)

	def explainer(self,name):
		if self.preShapConfig==False:
			self.set_explainer_config()

		f = plt.figure()
		shap.summary_plot(self.shapValues , self.regressor.musicData.xTest, plot_type="bar")
		f.savefig("newPlots/"+str(name)+"-explainer.png", bbox_inches='tight', dpi=600)

	def explainer_default(self,name):
		if self.preShapConfig==False:
			self.set_explainer_config()

		f = plt.figure()
		shap.summary_plot(self.shapValues , self.regressor.musicData.xTest)
		f.savefig("newPlots/"+str(name)+"-explainer_default.png", bbox_inches='tight', dpi=600)

	def decision_plot(self,name,nSamples=400):

		f = plt.figure()
		#features = musicData.df.iloc[select]
		#featuresDisplay = X_display.loc[nSamples]
		shap.decision_plot(self.explainer.expected_value[:nSamples], self.shapValues[:nSamples], self.regressor.musicData.xTest,ignore_warnings=True,title="")
		#shap.summary_plot(shapValues , self.regressor.musicData.xTrain)
		f.savefig("newPlots/"+str(name)+"-decision_plot.png", bbox_inches='tight', dpi=600)

	def force_plot(self,name,featureN,maxPlots=30):

		f = plt.figure()
		explainer = shap.TreeExplainer(self.regressor.get_model())
		shap_values = explainer.shap_values(self.regressor.musicData.xTest)
		print("Expected value: "+str(explainer.expected_value))

		for value in range(featureN,maxPlots):
			shap.force_plot(explainer.expected_value, shap_values[value,:], self.regressor.musicData.xTest.iloc[value,:], matplotlib=True, show=True)
			plt.savefig("newPlots/forcePlot/"+str(name)+"-force_plot"+str(value)+".png")

		plt.close()

	def bar_plot(self,name):
		f = plt.figure()
		explainer =shap.Explainer(self.regressor.get_model(),self.regressor.musicData.xTest)
		shap_values = explainer(self.regressor.musicData.xTest)
		shap.plots.bar(shap_values,show_data='true')
		#shap.summary_plot(self.shapValues , self.regressor.musicData.xTest)
		f.savefig("newPlots/"+str(name)+"-bar_plot.png", bbox_inches='tight', dpi=600)

	def music_decision_plot(self,name,musID=None):
		f = plt.figure()

		if self.preShapConfig==False or musID!=None:
			self.set_explainer_config(musID=musID)

		shap.decision_plot(self.explainer.expected_value, self.shapValues, self.regressor.musicData.xTest,ignore_warnings=True,title="")
		#shap.summary_plot(shapValues , self.regressor.musicData.xTrain)
		f.savefig("newPlots/"+str(name)+"-music_decision.png", bbox_inches='tight', dpi=600)
