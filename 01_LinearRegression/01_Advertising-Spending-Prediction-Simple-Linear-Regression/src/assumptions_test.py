import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm

class LinearRegressionAssumptions:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.fitted_values = model.fittedvalues
        self.residuals = model.resid
        self.normalized_residuals = model.get_influence().resid_studentized_internal

    def linearity(self):
        """
        Assumption 1: Linearity
        """
        sns.regplot(x=self.X, y=self.y)
        plt.show()

    def indipendence(self):
        """
        Assumption 2: Independence of errors
        """
        durbin_watson = sm.stats.stattools.durbin_watson(self.residuals)
        print("Durbin watson statistics: ", durbin_watson)
        if durbin_watson < 1.5 or durbin_watson > 2.5:
            print("Errors are not independent")
        else:
            print("Errors are independent")

    def normality(self):
        """
        Assumption 3: Normality of errors
        """
        sns.distplot(self.normalized_residuals)
        plt.show()

    def homoscedasticity_assumption(self):
        """
        Homoscedasticity: Assumes that the errors exhibits constant variable
        """
        sns.regplot(x=self.fitted_values, y=self.residuals)
        plt.title("Homoscedasticity")
        plt.show()

    def run_all(self):
        self.linearity()
        self.indipendence()
        self.normality()
        self.homoscedasticity_assumption()