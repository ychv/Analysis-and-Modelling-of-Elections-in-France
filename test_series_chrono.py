#%%
import pandas as pd
from dataImport import importCommuneData, filter_large_parquet

def sample_df(df,num_communes, rdm_seed=42):
    import random
    random.seed(rdm_seed)
    sample=random.sample(list(df["codecommune"].unique()), num_communes)
    return df[df["codecommune"].isin(sample)].sort_values(by=["codecommune", "annee"])

def select_election_type(df, type:{0,1}=1):
    df=df[df["type"]==type]
    df=df.drop(columns=[
        "type", 
        # "codecommune"
    ])
    return df

feature_list = [
    'annee',
    'type',

    y:='pvoteppar',
    # y:='pvotepvoteG',
    # y:='pvotepvoteC',
    # y:='pvotepvoteD',

    # "lat",
    # "long",

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
#%%
from feature_explo import select_no_na_features, get_all_features

feature_list=select_no_na_features(get_all_features())

feature_list=[
    'annee',
    'codecommune',

    y:='pvoteppar',
    # y:='pvotepvoteC',
    # y:='pvotepvoteCD',
    # y:='pvotepvoteCG',
    # y:='pvotepvoteD',
    # y:='pvotepvoteG',

    'inscrits',
    'type',
    # 'lat',
    # 'long',

    # 'agesexcommunes/perprop014',
    # 'agesexcommunes/perprop014_rank',
    # 'agesexcommunes/perpropf',
    # 'agesexcommunes/perpropf_rank',
    # 'agesexcommunes/perprop60p',
    # 'agesexcommunes/perprop60p_rank',
    # 'agesexcommunes/perpropf_pctchange',
    # 'agesexcommunes/perprop014_pctchange',
    # 'agesexcommunes/perprop60p_delta',
    # 'agesexcommunes/perprop60p_pctchange',
    # 'agesexcommunes/perage',
    # 'agesexcommunes/perage_rank',
    # 'agesexcommunes/perage_pctchange',
    # 'agesexcommunes/perprop014_delta',
    # 'agesexcommunes/perage_delta',
    # 'agesexcommunes/perpropf_delta',

    'popcommunes/peragglo',
    'popcommunes/peragglo_delta',
    'popcommunes/peragglo_rank',
    'popcommunes/percommu',
    'popcommunes/percommu_delta',
    'popcommunes/percommu_rank',
    'popcommunes/pop',
    'popcommunes/pop_delta',
    'popcommunes/pop_rank',
    'popcommunes/popagglo',
    'popcommunes/popagglo_delta',
    'popcommunes/popagglo_rank',
    # 'popcommunes/peragglo_pctchange',
    # 'popcommunes/percommu_pctchange',
    # 'popcommunes/pop_pctchange',
    # 'popcommunes/popagglo_pctchange',
    # 'popcommunesvbbm/vbbm',
    # 'popcommunesvbbm/vbbm_delta',
    # 'popcommunesvbbm/vbbm_pctchange',
    # 'popcommunesvbbm/vbbm_rank',
    # 'popcommunesvbbm/vbbmpauvresriches',
    # 'popcommunesvbbm/vbbmpauvresriches_pctchange',
    # 'popcommunesvbbm/vbbmpauvresriches_rank',
    # 'popcommunesvbbm/vbbmpauvresrichescap',
    # 'popcommunesvbbm/vbbmpauvresrichescap_pctchange',
    # 'popcommunesvbbm/vbbmpauvresrichescap_rank',
    # 'popcommunesvbbm/vbbmpauvresriches_delta',
    # 'popcommunesvbbm/vbbmpauvresrichescap_delta',
    # 'popcommuneselecteurs/electeurs',
    # 'popcommuneselecteurs/electeurs_rank',
    # 'popcommuneselecteurs/electeurs_delta'
    # 'popcommuneselecteurs/electeurs_pctchange',

    'pvotepreviousppar',
    'pvotepreviouspvoteC',
    'pvotepreviouspvoteCD',
    'pvotepreviouspvoteCG',
    'pvotepreviouspvoteD',
    'pvotepreviouspvoteG',
    
    'pvotepreviouspreviousppar',
    'pvotepreviouspreviouspvoteC',
    'pvotepreviouspreviouspvoteCD',
    'pvotepreviouspreviouspvoteCG',
    'pvotepreviouspreviouspvoteD',
    'pvotepreviouspreviouspvoteG',
]
path = "data/data_merged_20250922.parquet"
data=filter_large_parquet(path, feature_list)
data=select_election_type(data, 1)
data=sample_df(data, 100)
data
#%% Test avec MLForecast
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
from utilsforecast.plotting import plot_series

mlf = MLForecast(
    models=[
        # make_pipeline(LinearRegression()),
        make_pipeline(RandomForestRegressor(
            n_estimators = 1000,
            criterion = "squared_error",
            max_depth = 10,
            min_samples_split = 2,
            min_samples_leaf = 1,
            min_weight_fraction_leaf = 0,
            max_features = "sqrt",
            max_leaf_nodes = None,
            min_impurity_decrease = 0,
            bootstrap = True,
            random_state=42
        )),
        # make_pipeline(XGBRegressor()),
    ],
    lags=[1,2],
    freq=5,
    target_transforms=[Differences([1])],
)

mlf.preprocess(
    data,
    id_col="codecommune",
    time_col="annee",
    target_col=y,
    static_features = [
        # "lat",
        # "long"
    ],
)
#%%
horizon = 2
level = [95]
n = len(data)

annees = data["annee"].unique()
annees.sort()

train = data[data["annee"].isin(annees[:-horizon])]
test  = data[data["annee"].isin(annees[-horizon:])]

# DROP the communes where some years miss
future_df = mlf.make_future_dataframe(h=horizon)

merged = future_df.merge(
    test,
    on=["annee", "codecommune"],
    how="outer",
    indicator=True
)
expected_count = future_df["annee"].nunique()

complete_communes = (
    merged.groupby("codecommune")
    .size()
    .loc[lambda x: x == expected_count]
    .index
)
train = train[train["codecommune"].isin(complete_communes)]
test  = test[test["codecommune"].isin(complete_communes)]

print(train.shape, test.shape)
#%%
mlf.fit(
    train,    
    id_col="codecommune",
    time_col="annee",
    target_col=y,
    static_features = [
        # "lat",
        # "long"
    ],
)

fcst = mlf.predict(
    h=horizon, 
    X_df=test, 
    level=level,
)

print(data[y].describe())
for model in mlf.models_:
    rmse=root_mean_squared_error(test[y],fcst[model])
    print("\n____________________________")
    print(f"Model {model}")
    print(f"RMSE : {rmse}")

# fig = plot_series(
#     data, 
#     fcst, 
#     max_ids=4, 
#     plot_random=False,
#     id_col="codecommune",
#     time_col="annee",
#     target_col=y,
# )

# fig
#%%
data_stats = data[y].describe().to_dict()

stat_text = (
    f"Target Variable ({y}) Statistics:\n"
    f"Mean: {data_stats['mean']:.2f}\n"
    f"Std Dev: {data_stats['std']:.2f}\n"
    f"Min: {data_stats['min']:.2f}\n"
    f"Max: {data_stats['max']:.2f}"
)

rmse_text = "Model Performance (RMSE):\n"
for model in mlf.models_:
    rmse = root_mean_squared_error(test[y], fcst[model])
    model_name = str(model).split('(')[0].replace('Pipeline', '').strip()
    
    rmse_text += f"{model_name}: {rmse:.2f}\n"

full_text = stat_text + "\n\n" + rmse_text

fig = plot_series(
    data, 
    fcst, 
    max_ids=4, 
    plot_random=False,
    id_col="codecommune",
    time_col="annee",
    target_col=y,
    )

ax = fig.axes[0] 

ax.text(
    x=1.02,
    y=1,
    s=full_text,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment='top',
    bbox={
        'boxstyle': "round,pad=0.5", 
        'facecolor': 'white', 
        'alpha': 0.7, 
        'edgecolor': 'white'
    }
)

fig