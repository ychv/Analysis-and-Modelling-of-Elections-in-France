import pandas as pd


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
        path, engine="pyarrow", filters=[("Annee", ">", starting_year)]
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
