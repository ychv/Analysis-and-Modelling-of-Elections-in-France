import pandas as pd


def data_loader(
    path: str,
    type_election: int,
    starting_year: int,
    last_year: int,
    features: list[str] | None,
) -> pd.DataFrame:
    """
    The loader does not read whole dataset into memory, it leverages Predicate Pushdown techniques used in pyarrow.

    Inputs:
    - path str, path for full parquet dataset.
    - type_election int, 1 for "legislative" or 0 for "presidentielle"
    - starting_year int, lowest year allowed in dataset.
    - last_year int, highest year allowed in dataset.
    - features list[str], features to keep

    Returns:
    - df pd.DataFrame.
    """
    df = pd.read_parquet(
        path,
        engine="pyarrow",
        filters=[
            ("type", "==", type_election),
            ("annee", ">=", starting_year),
            ("annee", "<=", last_year),
        ],
        columns=features,
    )

    return df


def prepare_data(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Filters the data, keeps columns with missing data rate < threshold.
    Keeps rows relevant to provided type.

    Inputs:
        - df, pd.DataFrame, expecting the data output of data_loader.
        - threhold, float, corresponds to the maximum missing rate allowed in the columns.
    Output:
        - filtered pd.Dataframe
    """

    df_output = df.copy()
    mean_missing_cols = df_output.isna().mean()
    cols_to_keep = mean_missing_cols[mean_missing_cols <= threshold].index
    df_output_filtered = df_output[cols_to_keep]

    return df_output_filtered
