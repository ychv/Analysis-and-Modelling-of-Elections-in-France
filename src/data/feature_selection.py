import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from src.config import random_seed


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    A high-dimensional feature selector.

    This selector operates in two main phases to reduce a large feature set.
    It uses parallel processing and statistical sampling.

    Two filters:
        - Relevance Filter (Mutual Information):
        Calculates the Mutual Information (MI) between each feature and the target.
        MI captures any statistical dependency (linear or non-linear). Only the
        top `n_mi` features are retained.

        - Redundancy Filter (Recursive VIF):
        Computes the Variance Inflation Factor (VIF) for the remaining features.
        It iteratively removes the feature with the highest VIF until all
        remaining features fall below `vif_limit`. This ensures the final
        subset is not multicollinear.
    """

    def __init__(self, n_mi=50, vif_limit=10.0, n_jobs=-1):
        """
        Initializes the class.

        Inputs:
            - n_mi : int
            The number of features to retain after the Mutual Information phase.
            Acts as a pre-filter for the computationally expensive VIF phase.

            - vif_limit : float
            The maximum allowable VIF score. Values above 5-10 typically
            indicate significant multicollinearity in socio-economic data.

            - n_jobs : int
            The number of CPU cores to use for parallelizing MI and VIF calculations.
        """

        self.n_mi = n_mi
        self.vif_limit = vif_limit
        self.n_jobs = n_jobs
        self.selected_features = []

    def fit(self, X, y):
        """
        Fits class on provided data.
        Inputs:
            - X, pd.DataFrame
            - y, pd.Series

        Performs MI based selection first then VIF based selection on remaining features.
        """

        X_df = X.select_dtypes(include=[np.number])
        X_df = X_df.fillna(X_df.median())
        X_df = X_df.fillna(0)

        idx = X_df.sample(min(len(X_df), 50000), random_state=random_seed).index
        X_s = X_df.loc[idx]
        y_s = y.loc[idx]

        # Mutual Information
        print(f"Calcul MI sur {len(X_df.columns)} colonnes...")

        def get_mi(col_name):
            x_val = X_s[col_name].values.reshape(-1, 1)
            y_val = y_s.values.ravel()
            return (
                col_name,
                mutual_info_regression(x_val, y_val, random_state=random_seed)[0],
            )

        mi_results = Parallel(n_jobs=self.n_jobs)(
            delayed(get_mi)(c) for c in X_df.columns
        )
        mi_df = pd.DataFrame(mi_results, columns=["f", "score"]).sort_values(
            "score", ascending=False
        )
        current_cols = mi_df.head(self.n_mi)["f"].tolist()

        # VIF récursif
        print(f"Filtrage VIF sur {len(current_cols)} colonnes...")
        while len(current_cols) > 1:
            X_vif_vals = X_s[current_cols].values

            def compute_vif(i):
                target = X_vif_vals[:, i]
                features = np.delete(X_vif_vals, i, axis=1)
                r2 = LinearRegression().fit(features, target).score(features, target)
                return 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")

            vifs = Parallel(n_jobs=self.n_jobs)(
                delayed(compute_vif)(i) for i in range(len(current_cols))
            )

            max_vif = max(vifs)
            if max_vif <= self.vif_limit:
                break

            current_cols.pop(np.argmax(vifs))

        self.selected_features = current_cols
        return self

    def transform(self, X):
        return X[self.selected_features]
