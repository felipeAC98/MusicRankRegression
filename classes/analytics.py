import shap
import matplotlib.pyplot as plt

class shap_values():

	def __init__(self,regressor):
		self.regressor=regressor

	def tree_explainer(self):
		shap_values = shap.TreeExplainer(self.regressor.model).shap_values(self.regressor.musicData.xTrain)
		#shap.summary_plot(shap_values, self.regressor.musicData.xTrain, plot_type="bar")		

		f = plt.figure()
		shap.summary_plot(shap_values , self.regressor.musicData.xTrain, plot_type="bar")
		f.savefig("summary_plot1.png", bbox_inches='tight', dpi=600)

	def explainer(self):
		shap_values = shap.Explainer(self.regressor.model).shap_values(self.regressor.musicData.xTrain)
		f = plt.figure()
		shap.summary_plot(shap_values , self.regressor.musicData.xTrain, plot_type="bar")
		f.savefig("summary_plot1.png", bbox_inches='tight', dpi=600)

	def explainer_default(self):
		shap_values = shap.Explainer(self.regressor.model).shap_values(self.regressor.musicData.xTrain)
		f = plt.figure()
		shap.summary_plot(shap_values , self.regressor.musicData.xTrain)
		f.savefig("summary_plot1.png", bbox_inches='tight', dpi=600)