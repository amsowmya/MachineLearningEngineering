from sklearn.model_selection import train_test_split

def find_constant_columns(dataframe):
    """
    This function takes in a dataframe and return columns that contain single value.

    Args:
        dataframe (pandas.DataFrame): dataframe to be analysed

    Returns:
        list: A list of columns that contains single values
    """
    constant_columns = []
    for column in dataframe.columns:
        unique_values = dataframe[column].unique()
        if len(unique_values) == 1:
            constant_columns.append(column)
    return constant_columns

def delete_constant_column(dataframe, columns_to_delete):
    """
    This function takes in dataframe and a list of columns to delete and deletes the column which contain single value.

    Args:
        dataframe (pandas.DataFrame): The dataframe to be analysed
        columns_to_delete (list): A list of columns to delete

    Returns:
        pandas.DataFrame: A DataFrame with columns and that contains single value deleted
    """
    # Delete the specified column
    dataframe = dataframe.drop(columns_to_delete, axis=1)
    return dataframe

def find_columns_with_few_values(dataframe, threshold):
    """
    This function takes dataframe and threshold as input and returns the columns that have less than the threshold unique values.

    Args:
        dataframe (pandas.DataFrame): A dataframe to be analysed
        threshold (int): The minimum number of unique values required for the column

    Returns:
        list: A list of columns which have unique values less than the threshold
    """
    few_values_columns = []
    for column in dataframe.columns:
        # Get the number of unique values in the column
        unique_values_count = len(dataframe[column].unique())
        if unique_values_count < threshold:
            few_values_columns.append(column)
    return few_values_columns

def find_duplicate_rows(dataframe):
    """
    This function takes dataframe as input and returns the rows contain duplicate data.

    Args:
        dataframe (pandas.DataFrame): A dataframe to be analysed

    Returns:
        pandas.DataFrame: The dataframe containing duplicated rows
    """
    duplicated_rows = dataframe[dataframe.duplicated()]
    return duplicated_rows

def delete_duplicate_rows(dataframe):
    """
    This function takes dataframe as input and deletes the rows contain duplicate data

    Args:
        dataframe (pandas.DataFrame): A dataframe to be analyse

    Returns:
        pandas.DataFrame: A dataframe without duplicate rows
    """
    # Drop duplicate rows
    dataframe = dataframe.drop_duplicates(keep="first")
    return dataframe

def drop_and_fill(dataframe):
    """
    This function take the dataframe as input and identify the columns which is having more than 50% of the columns are null then delete that column.
    Then with the less than 50% of the null values in the columns values will be filled with the mean.

    Args:
        dataframe (pandas.DataFrame): A dataframe to be analysed.

    Returns:
        pandas.DataFrame: A dataframe without null values.
    """
    # Get the columns with more than 50% of missing values
    cols_to_del = dataframe.columns[dataframe.isnull().mean() > 0.5]
    # Drop the columns
    dataframe = dataframe.drop(cols_to_del, axis=1)
    # Fill the remaining values with mean of the column
    dataframe = dataframe.fillna(dataframe.mean())
    return dataframe

def split_data(dataframe, target_column):
    """
    This function takes the dataframe and target column as input. It splits the dataframe into feature dataframe and target dataframe.

    Args:
        dataframe (pnadas.DataFrame): A dataframe to be analysed.
        target_column (str): The name of the target column

    Returns:
        pandas.DataFrame: The dataframe containing the features
        pandas.DataFrame: The dataframe containing the target columns
    """
    X = dataframe.drop(target_column, axis=1)
    y = dataframe[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return (X_train, X_test, y_train, y_test)