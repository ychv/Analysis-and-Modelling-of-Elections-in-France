import pandas as pd

path = "../Data/data_merged_20250922.parquet" # path to Data file (in repo parent folder)

# data = pd.read_parquet(path,engine='fastparquet')
# print(data)

def importCommuneData(postcode,feature_list = None): # Create a datapoint for a specific commune by merging all the available data on this commune

    path = "../Data/data_merged_20250922.parquet" # path to Data file (in repo parent folder)
    col = None
    if feature_list is not None:
        col = ["codecommune"] + feature_list

    sel = [("codecommune","==",str(postcode))]
    data = pd.read_parquet(path,engine='pyarrow',columns = col,filters=sel)

    return data

    # path = "../Data/" # path to Data folder (in repo parent folder)
    # vect = pd.DataFrame({'code' : postcode},index=[0])

    # # Age
    # folder_path = path + 'Age_csv/'
    # data = pd.read_csv(folder_path + 'agesexcommunes.csv',index_col=False)
    # data = data.loc[data['codecommune'] == postcode]
    # vect = pd.concat([vect, data[[key for key in data.columns if key not in ['dep', 'nomdep', 'nomcommune']]]],axis=1)

    # data = pd.read_csv(folder_path + 'menagescommunes.csv',index_col=False)
    # data = data.loc[data['codecommune'] == postcode]
    # vect = pd.concat([vect, data[[key for key in data.columns if key not in ['dep', 'nomdep', 'nomcommune']]]],axis=1)

    # # Alphabetisation
    # file_path = path + 'Alphabetisation_csv/alphabetisationcommunes.csv'
    # data = pd.read_csv(file_path,index_col=False)
    # data = data.loc[data['codecommune'] == postcode]
    # vect = pd.concat([vect, data[[key for key in data.columns if key not in ['dep', 'nomdep', 'nomcommune']]]],axis=1)

    # # Capital Immobilier
    # folder_path = path + 'Capital_immobilier_csv/'
    # data = pd.read_csv(folder_path + 'basesfiscalescommunes.csv',index_col=False)
    # data = data.loc[data['codecommune'] == postcode]
    # vect = pd.concat([vect, data[[key for key in data.columns if key not in ['dep', 'nomdep', 'nomcommune']]]],axis=1)

    # data = pd.read_csv(folder_path + 'capitalimmobiliercommunes.csv',index_col=False)
    # data = data.loc[data['codecommune'] == postcode]
    # vect = pd.concat([vect, data[[key for key in data.columns if key not in ['dep', 'nomdep', 'nomcommune']]]],axis=1)
    
    # data = pd.read_csv(folder_path + 'isfcommunes.csv',index_col=False)
    # data = data.loc[data['codecommune'] == postcode]
    # vect = pd.concat([vect, data[[key for key in data.columns if key not in ['dep', 'nomdep', 'nomcommune']]]],axis=1)

    # data = pd.read_csv(folder_path + 'terrescommunes.csv',index_col=False)
    # data = data.loc[data['codecommune'] == postcode]
    # vect = pd.concat([vect, data[[key for key in data.columns if key not in ['dep', 'nomdep', 'nomcommune']]]],axis=1)

    # # Catégories socio professionnelles
    # folder_path = path + 'CSP_csv/'
    # data = pd.read_csv(folder_path + 'crimesdelitscommunes.csv',index_col=False)
    # data = data.loc[data['codecommune'] == postcode]
    # vect = pd.concat([vect, data[[key for key in data.columns if key not in ['dep', 'nomdep', 'nomcommune']]]],axis=1)
    
    # data = pd.read_csv(folder_path + 'cspcommunes.csv',index_col=False)
    # data = data.loc[data['codecommune'] == postcode]
    # vect = pd.concat([vect, data[[key for key in data.columns if key not in ['dep', 'nomdep', 'nomcommune']]]],axis=1)

    # data = pd.read_csv(folder_path + 'empfoncommunes.csv',index_col=False)
    # data = data.loc[data['codecommune'] == postcode]
    # vect = pd.concat([vect, data[[key for key in data.columns if key not in ['dep', 'nomdep', 'nomcommune']]]],axis=1)

    # data = pd.read_csv(folder_path + 'emploicommunes.csv',index_col=False)
    # data = data.loc[data['codecommune'] == postcode]
    # vect = pd.concat([vect, data[[key for key in data.columns if key not in ['dep', 'nomdep', 'nomcommune']]]],axis=1)

    # data = pd.read_csv(folder_path + 'rsacommunes.csv', index_col=False)
    # data = data.loc[data['codecommune'] == postcode]
    # vect = pd.concat([vect, data[[key for key in data.columns if key not in ['dep', 'nomdep', 'nomcommune']]]],axis=1)

    # # Diplomes
    # file_path = path + 'Diplomes_csv/diplomescommunes.csv'
    # data = pd.read_csv(file_path,index_col=False)
    # data = data.loc[data['codecommune'] == postcode]
    # vect = pd.concat([vect, data[[key for key in data.columns if key not in ['dep', 'nomdep', 'nomcommune']]]],axis=1)

    # # Enseignement privé et religion

    # # Législatives

    # # Nationalités
    # folder_path = path + 'Nationalites_csv/'
    # data = pd.read_csv(folder_path + 'etrangerscommunes.csv',index_col=False)
    # data = data.loc[data['codecommune'] == postcode]
    # vect = pd.concat([vect, data[[key for key in data.columns if key not in ['dep', 'nomdep', 'nomcommune']]]],axis=1)

    # data = pd.read_csv(folder_path + 'naticommunes.csv',index_col=False)
    # data = data.loc[data['codecommune'] == postcode]
    # vect = pd.concat([vect, data[[key for key in data.columns if key not in ['dep', 'nomdep', 'nomcommune']]]],axis=1)
    
    # # Présidentielles

    # # Propriétaires
    # file_path = path + 'Proprietaires_csv/proprietairescommunes.csv'
    # data = pd.read_csv(file_path,index_col=False)
    # data = data.loc[data['codecommune'] == postcode]
    # vect = pd.concat([vect, data[[key for key in data.columns if key not in ['dep', 'nomdep', 'nomcommune']]]],axis=1)

    # # Referundum

    # # Revenus
    # folder_path = path + 'Revenus_csv/'
    # data = pd.read_csv(folder_path + 'pibcommunes.csv',index_col=False)
    # data = data.loc[data['codecommune'] == postcode]
    # vect = pd.concat([vect, data[[key for key in data.columns if key not in ['dep', 'nomdep', 'nomcommune']]]],axis=1)
    
    # data = pd.read_csv(folder_path + 'revcommunes.csv',index_col=False)
    # data = data.loc[data['codecommune'] == postcode]
    # vect = pd.concat([vect, data[[key for key in data.columns if key not in ['dep', 'nomdep', 'nomcommune']]]],axis=1)

    # # Taille communes et agglomération


    # return vect

import dask.dataframe as dd
import pandas as pd
from typing import List, Optional

def filter_large_parquet(
    file_path: str,
    columns_to_keep: List[str],
    dropna_subset: Optional[List[str]] = None,
    npartitions: int = 4
) -> pd.DataFrame:
    """
    Reads a large Parquet file, selects necessary columns, and drops rows
    with missing values in specified columns, all without loading the
    entire file into RAM at once using Dask.

    Args:
        file_path: The path to the Parquet file.
        columns_to_keep: A list of column names that must be kept in the final DataFrame.
        dropna_subset: A list of column names to check for NaN/Na values. Rows
                       where any of these columns are NaN will be dropped.
                       If None, no rows will be dropped based on NaNs.
        npartitions: The number of Dask partitions to read the data into.
                     Adjust this based on your system's core count and data size.

    Returns:
        A pandas DataFrame containing the filtered and selected data.
        NOTE: The final output MUST fit into RAM, otherwise, this function
              should be modified to return the Dask DataFrame for further
              out-of-core processing.
    """
    print(f"Loading Parquet file from: {file_path} with {npartitions} partitions...")
    all_needed_cols = list(set(columns_to_keep + (dropna_subset if dropna_subset else [])))

    try:
        ddf = dd.read_parquet(
            file_path, 
            columns=all_needed_cols, 
            split_row_groups=True,
            npartitions=npartitions
        )
        print(f"Initial Dask DataFrame created with {len(ddf.columns)} columns.")
    except Exception as e:
        print(f"Error loading Parquet file: {e}")
        return pd.DataFrame()

    if dropna_subset:
        print(f"Dropping rows with NaNs in subset: {dropna_subset}")
        ddf = ddf.dropna(subset=dropna_subset)

    ddf = ddf[columns_to_keep]
    print(f"Final columns selected: {columns_to_keep}")
    
    print("Computing result and converting to a single Pandas DataFrame...")
    final_df = ddf.compute()
    
    print(f"Successfully loaded and filtered. Final DataFrame shape: {final_df.shape}")
    return final_df

if __name__=="__main__":
    df=filter_large_parquet(path, ["codecommune", "annee"], ["codecommune", "annee"])
    print(df)