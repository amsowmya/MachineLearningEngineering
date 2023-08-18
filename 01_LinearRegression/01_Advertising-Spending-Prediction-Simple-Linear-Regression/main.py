from src.data_ingest import DataIngestion
from src.data_preprocess import DataPreprocessing
from src.build_model import SimpleLinearRegression
from src.assumptions_test import LinearRegressionAssumptions

if __name__ == "__main__":
    # Initialize the DataIngestion class with the file path
    data_ingest = DataIngestion("./data/advertising.csv")

    # Load the data and get features and target variables 
    X, y, df = data_ingest.get_x_y()
    df.to_csv("./data/simple_df.csv", index=False)
    print(df)
    # Print the data to check if it was loaded correctly

    # Initialize the DataPreprocessing class with the data
    data_process = DataPreprocessing(df)
    # data_process.identify_outliers(df["TV"])
    outliers = data_process.idetify_outliers_zscore(df["TV"])
    print(outliers)

    # Build the model
    lr_model = SimpleLinearRegression(X, y)
    model = lr_model.summary()

    #  Assumptions test
    assumptions_test = LinearRegressionAssumptions(model, X, y)
    assumptions_test.run_all()

    


    
