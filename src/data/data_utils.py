import pandas as pd
import polars as pl
from typing import Optional,List

def filter_large_parquet(
    file_path: str,
    columns_to_keep: list[str],
    dropna_subset: Optional[list[str]] = None,
    filter: Optional[pl.Expr] = None
) -> pd.DataFrame:
    """
    Fastest and lightest method: Polars lazy scan of Parquet with
    predicate pushdown and projection pushdown
    (only reads the columns and row groups needed).
    """

    # Lazy scan (does NOT load data)
    lf = pl.scan_parquet(file_path)

    # Projection pushdown: only load needed columns
    needed = list(set(columns_to_keep + (dropna_subset or [])))
    lf = lf.select(needed)
    if dropna_subset:
        lf = lf.drop_nulls(subset=dropna_subset)
    if filter is not None:
        lf = lf.filter(filter)
    lf = lf.select(columns_to_keep)
    return lf.collect().to_pandas()

if __name__=="__main__":
    path = "data/data_merged_20250922.parquet" # path to Data file (in repo parent folder)
    df=filter_large_parquet(path, 
        ["codecommune", "annee"],
        dropna_subset=["codecommune", "annee"],
        filter=(
            # pl.col("codecommune") == "75010", 
            ~pl.col("annee").is_in([2022, 2017, 2012]) # ! on filtre (donc drop) les années qui sont là-dedans
            )
    )
    print(df)