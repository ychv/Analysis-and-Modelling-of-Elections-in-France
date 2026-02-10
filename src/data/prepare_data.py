import pandas as pd
import random
import numpy as np
from typing import Tuple


def data_loader(path: str, starting_year: int) -> pd.DataFrame:
    """
    Loads the dataset part that satsfies "annee">starting_year.
    The loader does not read whole dataset into memory, it leverages Predicate Pushdown techniques used in pyarrow.
    Inputs:
        - path, str: path for full parquet dataset.
        - starting_year, int: lowest yeart allowed in dataset.

    Returns pd.DataFrame.
    """
    df = pd.read_parquet(
        path, engine="pyarrow", filters=[("annee", ">", starting_year)]
    )

    return df


def prepare_data(df: pd.DataFrame, threshold: float, type: int) -> pd.DataFrame:
    """
    Filters the data, keeps columns with missing data rate < threshold.
    Keeps rows relevant to provided type.

    Inputs:
        - df, pd.DataFrame, expecting the data output of data_loader.
        - threhold, float, corresponds to the maximum missing rate allowed in the columns.
        - type, int, 0 or 1, corresponds to the type of elections to focus on.
                     0 for "présidentielles" and 1 for "législatives"
    Output:
        - filtered pd.Dataframe
    """

    df_output = df[df["type"] == type]
    mean_missing_cols = df_output.isna().mean()
    cols_to_keep = mean_missing_cols[mean_missing_cols <= threshold].index
    df_output_filtered = df_output[cols_to_keep]

    return df_output_filtered

def sample_df(df: pd.DataFrame, num_communes: int) -> pd.DataFrame:
    """
    Randomly samples a specified number of unique communes and returns their full history.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'codecommune' column.
        num_communes (int): The number of unique communes to sample.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only the sampled communes, 
            sorted by commune code and year.
    """
    sample=random.sample(list(df["codecommune"].unique()), num_communes)
    return df[df["codecommune"].isin(sample)].sort_values(by=["codecommune", "annee"])

def split_serie_temp(data: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Splits a time-series dataset into training and testing sets based on election years.

    Args:
        data (pd.DataFrame): The input DataFrame containing an 'annee' (year) column.
        horizon (int): The number of most recent election years to include in the test set.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: A tuple containing:
            - train (pd.DataFrame): Data from all years except the last 'horizon' years.
            - test (pd.DataFrame): Data from the last 'horizon' years.
            - years (np.ndarray): A sorted array of all unique years in the original data.

    Raises:
        ValueError: If the number of unique years in the data is less than or equal to the horizon.
    """
    years = np.sort(data["annee"].unique())
    if len(years) <= horizon:
        raise ValueError("Not enough elections in the selected period for horizon h.")
    train_years = years[:-horizon]
    test_years = years[-horizon:]
    train = data[data["annee"].isin(train_years)].copy()
    test = data[data["annee"].isin(test_years)].copy()
    return train, test, years