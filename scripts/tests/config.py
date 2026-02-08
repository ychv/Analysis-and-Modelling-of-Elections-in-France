from mlforecast.target_transforms import Differences
from src.config import type_legis, type_pres, random_seed
from sklearn.ensemble import RandomForestRegressor

id_col="codecommune" #"series_id"
time_col="annee" #"step"
horizon=1
num_communes=100
election_type=type_legis
year_start=2012
year_end=2022

#FEATURES CONFIG
feature_list = [
    'codecommune',
    'annee',
    'type',

    y:='pvoteppar',
    # y:='pvotepvoteC',
    # y:='pvotepvoteCD',
    # y:='pvotepvoteCG',
    # y:='pvotepvoteD',
    # y:='pvotepvoteG',

    "lat",
    "long",

    'popcommunes/pop',
    'popcommuneselecteurs/electeurs',
    
    'capitalimmobiliercommunes/capitalimmo',
    'capitalimmobiliercommunes/capitalimmoagglo',
    'capitalimmobiliercommunes/capitalratio',
    'capitalimmobiliercommunes/capitalratioagglo',

    # 'capitalimmobiliercommunes/prixm2',
    # 'capitalimmobiliercommunes/prixm2ratio',

    # 'revcommunes/nadult',
    # 'capitalimmobiliercommunes/surface',
    # 'agesexcommunes/perage_rank'
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
freq=5

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
# model2=RandomForestRegressor
# params2={
#     "n_estimators": 100,
#     "criterion": "squared_error"
# }