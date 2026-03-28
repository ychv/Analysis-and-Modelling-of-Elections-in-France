import pandas as pd
import numpy as np
from src.config import (
    random_seed,
    selected_num_features,
    full_dataset_path,
    vote_features,
    missing_data_thresh,
    cols_to_keep,
    test_size,
)
from src.data.prepare_data import data_loader, prepare_data
from sklearn.model_selection import train_test_split


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
        features=None,
        threshold=missing_data_thresh,
    ):
        """
        Loads dataset using provided filters. (cf. data_loader function)
        Filters out columns with high missing rate.
        """
        if features is None:
            combined = self.features + self.vote_cols + self.cols_to_keep
            features = [f if isinstance(f, str) else f[0] for f in combined]
        print(
            "---------------------------------------------\n",
            features,
            "\n---------------------------------------------",
        )
        df = data_loader(
            self.data_path, election_type, starting_year, last_year, features
        )

        self.df = prepare_data(df, threshold)
        self.s_year, self.l_year = starting_year, last_year

    def split_X_y(self, predicted_col: str):
        X = self.df.sort_values("annee").drop(
            columns=self.vote_cols + self.cols_to_keep
        )
        y = self.df.sort_values("annee")[predicted_col]

        return X, y

    def split_train_test(self, predicted_col: str):
        is_last_year = self.df["annee"] == self.l_year

        drop_cols = list(set(self.vote_cols + self.cols_to_keep))

        df_last = self.df[is_last_year]
        X_last = df_last.drop(columns=drop_cols)
        y_last = df_last[predicted_col]

        X_test_spatial, X_val_spatial, y_test_spatial, y_val_spatial = train_test_split(
            X_last, y_last, test_size=test_size, random_state=self.random_seed
        )

        df_hist = self.df[~is_last_year]
        X_hist = df_hist.drop(columns=drop_cols)
        y_hist = df_hist[predicted_col]

        X_train = pd.concat([X_hist, X_test_spatial], axis=0)
        y_train = pd.concat([y_hist, y_test_spatial], axis=0)

        return (
            X_train.astype(np.float32),
            X_val_spatial.astype(np.float32),
            y_train.astype(np.float32),
            y_val_spatial.astype(np.float32),
        )

    def aggregate_dep(self):
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
        self.agg_df = agg[target_cols].div(agg["pop"], axis=0).reset_index()

    def aggregate_dep_inscrits(self):
        """
        Aggregates dataframe on department level, weighted sum using registered voters as ponderation.
        """

        df = self.df.copy()
        df["pop"] = self.df["inscrits"]

        exclude_cols = ["codecommune", "pop"]
        target_cols = [c for c in self.df.columns if c not in exclude_cols]
        df["dep"] = df["codecommune"].astype(str).str.zfill(5).str.slice(0, 2)
        df.loc[df["dep"] == "97", "dep"] = df["codecommune"].str.slice(0, 3)

        weighted_df = df[target_cols].multiply(df["pop"], axis=0)
        weighted_df["dep"] = df["dep"]
        weighted_df["pop"] = df["pop"]

        agg = weighted_df.groupby("dep").sum()
        self.agg_df = agg[target_cols].div(agg["pop"], axis=0).reset_index()
