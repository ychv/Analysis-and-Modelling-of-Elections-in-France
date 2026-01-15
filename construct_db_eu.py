#%%
from dataImport import filter_large_parquet
import polars as pl
import pandas as pd
#%%
# Lien des datasets + nuances : https://www.data.gouv.fr/datasets/donnees-des-elections-agregees
#%%
path = "data/europeennes/candidats_results.parquet" # path to Data file (in repo parent folder)
features= ["id_election","id_brut_miom", "Code du département", "Code de la commune", "Code du b.vote", "Libellé Abrégé Liste", "Libellé Etendu Liste", "Nom Tête de Liste", "Voix", "Nom", "Prénom", "Nuance", "Binôme", "Liste"]
filter=( ~pl.col("id_election").is_in(['2001_cant_t1', '2001_cant_t2', '2002_legi_t1', '2002_legi_t2',
       '2002_pres_t1', '2002_pres_t2', '2004_cant_t1', '2004_cant_t2',
       '2004_regi_t1', '2007_legi_t1', '2007_legi_t2', '2007_pres_t1',
       '2007_pres_t2', '2008_cant_t1', '2008_cant_t2', '2008_muni_t1',
       '2008_muni_t2', '2010_regi_t1', '2010_regi_t2', '2011_cant_t1',
       '2011_cant_t2', '2012_legi_t1', '2012_legi_t2', '2012_pres_t1',
       '2012_pres_t2', '2014_muni_t1', '2014_muni_t2', '2015_dpmt_t1',
       '2015_dpmt_t2', '2015_regi_t1', '2015_regi_t2', '2017_legi_t1',
       '2017_legi_t2', '2017_pres_t1', '2017_pres_t2', '2020_muni_t1',
       '2020_muni_t2', '2021_dpmt_t1', '2021_dpmt_t2', '2021_regi_t1',
       '2021_regi_t2', '2022_legi_t1', '2022_legi_t2', '2022_pres_t1',
       '2022_pres_t2', '2024_legi_t1', '2024_legi_t2']))

data:pd.DataFrame=filter_large_parquet(path, features, filter=filter)
data


#%% DATA EXPLORATION
data.columns
#%%
data=data["id_election"].unique()
data.sort()
#%%
data["Nuance"][pd.isna(data["Nuance"]) == False].unique()


#%% BASIC MODIFS
data["index_commune"]=data["Code du département"]+data["Code de la commune"]
data
#%%
data = data[['id_election',"id_brut_miom", 'index_commune', 'Nuance', 'Voix',
    'Libellé Etendu Liste', 'Libellé Abrégé Liste', 'Code du département', 'Code de la commune','Code du b.vote',
    'Nom Tête de Liste',  'Nom', 'Prénom',  'Binôme',
    'Liste']]
data
#%%
data["id_election"]=data["id_election"].apply(lambda x : int(x[:4]))
data
#%%
data=data.sort_values(by=["index_commune", "id_election"])
data=data.reset_index(drop=True)
data
#%%
data.to_parquet("data/europeennes/agg_elections_99-24_commune.parquet")
#%%
data=pd.read_parquet("data/europeennes/agg_elections_99-24_commune.parquet")
data


#%% MAPPINGS
#%% mapping 1 : en utilisant le nuancier fourni
mapping=pd.read_csv("data/europeennes/nuances.csv")
mapping
#%%
mapping_dict=mapping.set_index("Nuance")["Tendance"]
mapping_dict=mapping_dict.to_dict()
mapping_dict
#%%
data["Tendance"]=data["Nuance"].map(mapping_dict)
data
#%% mapping 2 : en labelisant ce qui manque à la main
pd.read_csv("data/europeennes/libelle_to_nuance.csv").sort_values(by="Tendance").to_csv("data/europeennes/tendances_sort.csv", index=False)
#%%
mapp=pd.read_csv("data/europeennes/libelle_to_nuance.csv")
mapp
#%%
mapp=mapp.set_index("Nuance")["Tendance"]
mapp=mapp.to_dict()
mapp
#%%
mask = data["Tendance"].isna()
data.loc[mask, "Tendance"] = data.loc[mask, "Nuance"].map(mapp)
data
#%%
data["Tendance"].value_counts(dropna=False)
#%%
data.to_parquet("data/europeennes/save_after_mappings.parquet")
#%%
data=pd.read_parquet("data/europeennes/save_after_mappings.parquet")
data


#%% Drop unused columns + reorder
data=data.drop(columns=["Libellé Etendu Liste", "Libellé Abrégé Liste"])
data=data[['id_election', 'index_commune', 'Voix', 'Nuance','Tendance', 'Code du département',
       'Code de la commune', 'Nom', 'Prénom','Nom Tête de Liste', 'Binôme',
       'Liste']]
data
#%%
data.to_parquet("data/europeennes/cleaned_df.parquet")
#%%
data=pd.read_parquet("data/europeennes/cleaned_df.parquet")
data


#%%
df_sum = (
    data
    .groupby(["id_election", "index_commune", "Tendance"], as_index=False)
    .agg({
        "Voix": "sum",
        'Code du département' : "first",
       'Code de la commune' : "first",
})
)
df_sum
#%%
df_sum.to_parquet("data/europeennes/df_sum.parquet")
#%%
df_sum=pd.read_parquet("data/europeennes/df_sum.parquet")
df_sum

#%%
df=df_sum.pivot_table(
       index=["index_commune", "id_election"],
       columns="Tendance",
       values="Voix",
       aggfunc="sum",
       fill_value=0
).reset_index()
df["Total"]=df[["Centre","Centre-droit", "Centre-gauche", "Gauche", "Droite"]].sum(axis=1)
df[["Centre","Centre-droit", "Centre-gauche", "Gauche", "Droite"]]=df.apply(lambda x : x[["Centre","Centre-droit", "Centre-gauche", "Gauche", "Droite"]]/x["Total"] if x["Total"]!=0 else 0, axis=1)
#%%
mapper={
    "Centre":'pvotepvoteC',
    "Centre-droit":'pvotepvoteCD',
    "Centre-gauche":'pvotepvoteCG',
    "Droite":'pvotepvoteD',
    "Gauche":"pvotepvoteG",
    "Total":"exprimes"
}
df=df.rename(columns=mapper)
df

#%%
df.to_parquet("data/europeennes/df_europeennes.parquet")
#%%
df=pd.read_parquet("data/europeennes/df_europeennes.parquet")
df

#%%
general_results=pd.read_parquet("data/europeennes/general_results.parquet")
general_results
#%% BASIC MODIFS
general_results["index_commune"]=general_results["Code du département"]+general_results["Code de la commune"]
general_results
#%%
general_results = general_results[["id_election", "index_commune", "Inscrits"]]
general_results
#%%
general_results["id_election"]=general_results["id_election"].apply(lambda x : int(x[:4]))
general_results
#%%
general_results=general_results.sort_values(by=["index_commune", "id_election"])
general_results=general_results.reset_index(drop=True)
general_results
#%%
general_results = (
    general_results
    .groupby(["id_election", "index_commune"], as_index=False)
    .agg({
        "Inscrits": "sum",
})
)
#%%
general_results.to_parquet("data/europeennes/general_results_clean.parquet")
#%%
general_results=pd.read_parquet("data/europeennes/general_results_clean.parquet")
general_results

#%%
df=df.merge(
       general_results, 
       on=["id_election", "index_commune"],
       how="left",
)
df
#%%
df["exprimes"]=df.apply(lambda x : x["exprimes"]/x["Inscrits"] if x["Inscrits"]!=0 else 0, axis=1)
df=df.rename(columns={"exprimes":"pvoteppar"})
df
#%%
df.to_parquet("data/europeennes/data_europeennes_final.parquet")
#%%
pd.read_parquet("data/europeennes/data_europeennes_final.parquet")