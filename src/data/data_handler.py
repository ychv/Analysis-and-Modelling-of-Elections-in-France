import pandas as pd
import numpy as np
from src.config import (
    random_seed,
    selected_num_features,
    full_dataset_path,
    vote_features,
    missing_data_thresh,
    cols_to_keep,
)
from src.data.prepare_data import data_loader, prepare_data


class DataHandler:
    def __init__(self, data_path=full_dataset_path, random_seed=random_seed):
        self.random_seed = random_seed
        self.data_path = data_path
        self.df = pd.DataFrame()
        self.features = selected_num_features
        self.vote_cols = vote_features
        self.cols_to_keep = cols_to_keep

    def load_data(
        self,
        starting_year,
        last_year,
        election_type,
        features=selected_num_features + vote_features + cols_to_keep,
        threshold=missing_data_thresh,
    ):
        """
        Loads dataset using provided filters. (cf. data_loader function)
        Filters out columns with high missing rate.
        """

        df = data_loader(
            self.data_path, election_type, starting_year, last_year, features
        )

        self.df = prepare_data(df, threshold)

    def split_X_y(self, predicted_col: str):
        X = self.df.drop(columns=self.vote_cols + self.cols_to_keep)
        y = self.df[predicted_col]

        return X, y

    def aggregate_dep(self) -> pd.DataFrame:
        """
        Aggregates dataframe on department level, weighted sum using population as ponderation.
        """

        df = self.df.copy()
        df["pop"] = self.df["agesexcommunes/popf"] + self.df["agesexcommunes/poph"]

        exclude_cols = ["codecommune", "pop"]
        target_cols = [c for c in self.df.columns if c not in exclude_cols]
        df["dep"] = df["codecommune"].astype(str).str.zfill(5).str.slice(0, 2)
        df.loc[df["dep"] == "97", "dep"] = df["codecommune"].str.slice(0, 3)

        weighted_df = df[target_cols].multiply(df["pop"], axis=0)
        weighted_df["dep"] = df["dep"]
        weighted_df["pop"] = df["pop"]

        agg = weighted_df.groupby("dep").sum()
        return agg[target_cols].div(agg["pop"], axis=0).reset_index()
