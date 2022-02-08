import shap
import matplotlib.pyplot as plt

class shap_values():

	def __init__(self,regressor):
		self.regressor=regressor

	def tree_explainer(self,name):
		shap_values = shap.TreeExplainer(self.regressor.model).shap_values(self.regressor.musicData.xTrain)
		#shap.summary_plot(shap_values, self.regressor.musicData.xTrain, plot_type="bar")		
		f = plt.figure()
		shap.summary_plot(shap_values , self.regressor.musicData.xTrain, plot_type="bar")
		f.savefig("newPlots/"+str(name)+"-tree_explainer.png", bbox_inches='tight', dpi=600)

	def explainer(self,name):
		shap_values = shap.Explainer(self.regressor.get_model()).shap_values(self.regressor.musicData.xTrain)
		f = plt.figure()
		shap.summary_plot(shap_values , self.regressor.musicData.xTrain, plot_type="bar")
		f.savefig("newPlots/"+str(name)+"-explainer.png", bbox_inches='tight', dpi=600)

	def explainer_default(self,name):
		shap_values = shap.Explainer(self.regressor.get_model()).shap_values(self.regressor.musicData.xTrain)
		f = plt.figure()
		shap.summary_plot(shap_values , self.regressor.musicData.xTrain)
		f.savefig("newPlots/"+str(name)+"-explainer_default.png", bbox_inches='tight', dpi=600)

	def decision_plot(self,name,nSamples=100):
		explainer = shap.TreeExplainer(self.regressor.get_model())
		shapValues =explainer.shap_values(self.regressor.musicData.xTest)

		f = plt.figure()
		#features = musicData.df.iloc[select]
		#featuresDisplay = X_display.loc[nSamples]
		shap.decision_plot(explainer.expected_value[:nSamples], shapValues[:nSamples], self.regressor.musicData.xTest,ignore_warnings=True)
		#shap.summary_plot(shapValues , self.regressor.musicData.xTrain)
		f.savefig("newPlots/"+str(name)+"-decision_plot.png", bbox_inches='tight', dpi=600)