#%%
from dataImport import filter_large_parquet

path = "data/data_merged_20250922.parquet"
feature_list = [
    "codecommune",
    'annee',
    'type',

    y:='pvoteppar',
    # y:='pvotepvoteG',
    # y:='pvotepvoteC',
    # y:='pvotepvoteD',

'popcommunes/peragglo', 'popcommunes/peragglo_delta', 'popcommunes/peragglo_pctchange', 'popcommunes/peragglo_rank', 'popcommunes/percommu', 'popcommunes/percommu_delta', 'popcommunes/percommu_pctchange', 'popcommunes/percommu_rank', 'popcommunes/pop', 'popcommunes/pop_delta', 'popcommunes/pop_pctchange', 'popcommunes/pop_rank', 'popcommunes/popagglo', 'popcommunes/popagglo_delta', 'popcommunes/popagglo_pctchange', 'popcommunes/popagglo_rank', 'popcommuneselecteurs/electeurs', 'popcommuneselecteurs/electeurs_delta', 'popcommuneselecteurs/electeurs_pctchange', 'popcommuneselecteurs/electeurs_rank', 

]
df = filter_large_parquet(path, feature_list)
df=df.sort_values(by=["codecommune","annee"])
df.count()
#%%
import re
import pandas as pd
import tqdm
from pathlib import Path

def get_all_features():
    with open("features.txt", "r") as f:
        text=f.read()
        text = re.sub(r'[^\w\s\.\,\-\'\"\/]', '', text)
        text = re.sub(r'[\n\t\r]', '', text)
        text = re.sub(r'\s+', '', text)
        text=re.sub(r"\'", '', text)
        feature_list= text.split(",")
    return feature_list

def generate_count_file(path_merged_df,features_list):
    count_df={"feature":[], "count":[]}
    for feature in tqdm.tqdm(features_list):
        features=["codecommune", "annee"] + [feature]
        df=filter_large_parquet(path_merged_df,features,None, npartitions=10)
        count_df["feature"].append(feature)
        count_df["count"].append(df[feature].count())
        pd.DataFrame(count_df).to_csv("features_count.csv")
    return pd.DataFrame(count_df)

def select_no_na_features(features_list:list[str], prop_na:float=0.15, path_merged_df="data/data_merged_20250922.parquet", display_count=False):
    if (path:=Path("features_count.csv")).is_file():
        df=pd.read_csv(path)
    else:
        df=generate_count_file(path_merged_df, features_list)
    total=int(df[df["feature"]=="type"]["count"])
    df["missing"]=100-df["count"]/total * 100
    features_df=df[df['missing']<prop_na*100].sort_values(by=["missing","feature"])
    if display_count:
        return features_df
    else:
        return features_df["feature"].tolist()

feature_list=get_all_features()
select_no_na_features(
    feature_list,
    # display_count=True
    )