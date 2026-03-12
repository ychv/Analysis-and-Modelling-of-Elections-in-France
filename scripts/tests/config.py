from mlforecast.target_transforms import Differences
from src.config import type_legis, type_pres, random_seed
from sklearn.ensemble import RandomForestRegressor
# from sktree.ensemble import ObliqueRandomForestRegressor

id_col="codecommune" #"series_id"
time_col="annee" #"step"
horizon=1 # can be negative # Example: years=[86, 88, 93, 99], h=-2 -> test=[86, 88], train=[93, 99]
num_communes=100 # can be negative to run over all available communes
election_type=type_legis
year_start=2012
year_end=2022

#FEATURES CONFIG
feature_list = [
    'codecommune',
    'annee',
    'type',

    # y:='pvoteppar',
    # y:='pvotepvoteC',
    # y:='pvotepvoteCD',
    # y:='pvotepvoteCG',
    y:='pvotepvoteD',
    # y:='pvotepvoteG',

    "lat",
    "long",

    'pibcommunes/pibtot_pctchange', 
    'capitalimmobiliercommunes/capitalratio_pctchange', 
    'capitalimmobiliercommunes/capitalratioagglo_pctchange', 
    'popcommunes/peragglo', 
    'capitalimmobiliercommunes/capitalimmo', 
    'pibcommunes/pibtot', 
    'proprietairescommunes/nlogement',
    'popcommunes/percommu', 
    'agesexcommunes/popf', 
    'diplomescommunes/supf_rank', 
    'popcommuneselecteurs/electeurs_rank', 
    'capitalimmobiliercommunes/capitalimmoagglo_delta', 
    'diplomescommunes/suph_rank', 
    'cspcommunes/cadr_rank', 
    'pibcommunes/pibtot_delta', 
    'cspcommunes/capi_rank'
]
static_features = ["lat","long"] if "lat" in feature_list else []

#SERIE TEMP CONFIG
lags=[
    1,
    # 2,
    # 3,
    # 4
]
target_transforms=[
    # Differences([1])
]
freq=1

#MODEL CONFIG
model=RandomForestRegressor
params={
    "n_estimators": 1000,
    "criterion": "squared_error",
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0,
    "max_features": "sqrt",
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0,
    "bootstrap": True,
    "random_state":random_seed
}

# ADD more models if needed
# model2=ObliqueRandomForestRegressor
# params2={
#     "n_estimators": 600,
#     "max_depth": 6,
#     "min_samples_split": 4,
#     "min_samples_leaf": 2,
#     "feature_combinations": 0.5,
#     "random_state":random_seed
# }