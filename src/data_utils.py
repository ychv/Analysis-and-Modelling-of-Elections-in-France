import pandas as pd
import numpy as np

eps = 1e-4


def logit(y):
    y = np.clip(y, eps, 1 - eps)
    return np.log(y / (1 - y))


def expit(z):
    return 1 / (1 + np.exp(-z))


def augment_df(
    current_df: pd.DataFrame,
    next_df: pd.DataFrame,
    cols: list,
    only_delta: bool = True,
) -> "pd.DataFrame":
    delta_df = pd.merge(
        current_df[["codecommune"] + cols],
        next_df[["codecommune"] + cols],
        on="codecommune",
        suffixes=("_current", "_next"),
        how="inner",
    )
    new_cols = ["codecommune"]
    for col in cols:
        delta_df[f"delta_{col}"] = (
            delta_df[f"{col}_next"] - delta_df[f"{col}_current"]
        ) / (delta_df[f"{col}_current"] + 1e-12)
        new_cols.append(f"delta_{col}")
    return delta_df[new_cols] if only_delta else delta_df


def from_years_to_delta(
    years: list[int], df: pd.DataFrame, cols: list[str]
) -> pd.DataFrame:
    years.sort()
    n = len(years)
    delta_hist = []
    for i in range(n - 1):
        delta = augment_df(
            df[df["annee"] == years[i]], df[df["annee"] == years[i + 1]], cols
        )
        delta_hist.append(delta)
    new_df = delta_hist[0]
    for i, delta in enumerate(delta_hist[1:]):
        new_df = pd.merge(
            new_df, delta, on="codecommune", how="left", suffixes=(f"_{i}", f"_{i+1}")
        )

    return new_df
