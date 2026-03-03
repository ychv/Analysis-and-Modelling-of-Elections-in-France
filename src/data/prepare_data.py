import pandas as pd
import random
import numpy as np
from typing import Tuple

from src.data.data_utils import get_all_years


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

def sample_df(df: pd.DataFrame, num_communes: int, random_seed:int) -> pd.DataFrame:
    """
    Randomly samples a specified number of unique communes and returns their full history.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'codecommune' column.
        num_communes (int): The number of unique communes to sample.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only the sampled communes, 
            sorted by commune code and year.
    """
    random.seed(random_seed)
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
    num_years = len(years)
    
    if num_years <= horizon:
        raise ValueError("Not enough elections in the selected period for horizon h.")

    # 2. DROP COMMUNES WITH MISSING INFO
    # We count how many years each commune appears in. 
    # If a commune's count < total unique years, it has missing data.
    commune_counts = data.groupby("codecommune")["annee"].nunique()
    complete_communes = commune_counts[commune_counts == num_years].index
    data_clean = data[data["codecommune"].isin(complete_communes)].copy()
    
    dropped_count = data["codecommune"].nunique() - len(complete_communes)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} communes due to missing election years.")

    # 3. Split based on the cleaned data
    train_years = years[:-horizon]
    test_years = years[-horizon:]
    
    train = data_clean[data_clean["annee"].isin(train_years)].copy()
    test = data_clean[data_clean["annee"].isin(test_years)].copy()
    
    return train, test, years

def create_election_mapping(years_list: list[int]) -> Tuple[dict[int, int], dict[int, int]]:
    """
    Creates bidirectional mapping between irregular years and a continuous index.
    """
    sorted_years = sorted(list(set(years_list)))
    year_to_idx = {year: i for i, year in enumerate(sorted_years)}
    idx_to_year = {i: year for i, year in enumerate(sorted_years)}
    return year_to_idx, idx_to_year

def apply_time_mapping(df: pd.DataFrame, time_col: str, mapping: dict[int, int]) -> pd.DataFrame:
    """Applies the year-to-index mapping to a dataframe."""
    df_mapped = df.copy()
    df_mapped[time_col] = df_mapped[time_col].map(mapping)
    return df_mapped

def reverse_time_mapping(df: pd.DataFrame, time_col: str, inv_mapping: dict[int, int]) -> pd.DataFrame:
    """Restores original years from the integer index."""
    df_decoded = df.copy()
    df_decoded[time_col] = df_decoded[time_col].map(inv_mapping)
    return df_decoded

def calculate_election_duration(df: pd.DataFrame, time_col: str, id_col: str) -> pd.DataFrame:
    """
    Calculates the number of years since the last election for each entity.
    """
    df = df.sort_values([id_col, time_col])
    
    # Calculate the difference between current year and previous year per ID
    df['election_duration'] = df.groupby(id_col)[time_col].diff()
    
    # Fill the first election (NaN) with the most common duration (e.g., 5) 
    # or a specific value to avoid nulls in the model.
    df['election_duration'] = df['election_duration'].fillna(5)
    
    return df

def get_display_window_lagged(full_dataset_path:str, year_start:int, year_end:int, lags:list[int]) -> int:
    """
    Get and display the exact start year to be able to compute lags afterwards.
    """
    all_available_years = get_all_years(full_dataset_path)

    # Trouver l'index de l'année de départ
    start_idx = all_available_years.index(year_start)
    if lags!=[]:
        extended_start_year = all_available_years[max(0, start_idx - max(lags))]
    else:
        extended_start_year = year_start

    print(f"Fenêtre d'analyse : {year_start} à {year_end}")
    print(f"Chargement avec historique pour lags : {extended_start_year} à {year_end}")
    return extended_start_year