import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class DataPreprocessing:
    def __init__(self, df):
        self.df = df

    def identify_outliers(self, data: pd.DataFrame):
        """
        Function to identify outliers in data using Boxplot visualization
        Args:
            data (pd.DataFrame): The data to be visualized 
        Returns:
            None
        """
        fig, ax = plt.subplots()
        ax.boxplot(data)
        ax.set_title("Box plot of data")
        ax.set_ylabel("Value")
        plt.show()

    def idetify_outliers_zscore(self, data: pd.Series, threshold: float=3):
        """
        Function to identify outliers in the data using z_score method.

        Args:
            data (pd.Series): The data to be analysed
            threshold (float, optional): The z_score threshold used to identify outliers.
                                Outliers are data points with a Z_score greater than this threshold.
                                Default value is 3.
        """
        mean = np.mean(data)
        std = np.std(data)
        z_score = (data - mean) / std
        outliers = data[np.abs(z_score) > threshold]
        return outliers