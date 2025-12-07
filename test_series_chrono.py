#%%
import pandas as pd
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
from utilsforecast.plotting import plot_series
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

def get_data_years(data : pd.DataFrame, start:int, end:int, horizon:int):
    years = np.sort(data["annee"].unique())

    if start is not None:
        i_start = np.argmin([abs(start - y) for y in years])
    else:
        i_start = 0

    if end is not None:
        i_end = np.argmin([abs(end - y) for y in years])
    else:
        i_end = len(years) - 1

    selected_years = years[i_start : i_end + 1]

    if len(selected_years) <= horizon:
        raise ValueError("Not enough elections in the selected period for horizon h.")

    # ----------------------------------------------
    # Train : all except the last h elections
    # Test  : last h elections
    # ----------------------------------------------
    train_years = selected_years[:-horizon]
    test_years  = selected_years[-horizon:]

    train = data[data["annee"].isin(train_years)].copy()
    test  = data[data["annee"].isin(test_years)].copy()

    all_years_needed = np.concatenate([train_years, test_years])
    expected_count = len(all_years_needed)

    merged = data[data["annee"].isin(all_years_needed)]

    complete_communes = (
        merged.groupby("codecommune")
            .size()
            .loc[lambda x: x == expected_count]
            .index
    )

    train = train[train["codecommune"].isin(complete_communes)]
    test  = test[test["codecommune"].isin(complete_communes)]
    return train, test, selected_years

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

    # 'pvotepreviousppar',
    # 'pvotepreviouspvoteC',
    # 'pvotepreviouspvoteCD',
    # 'pvotepreviouspvoteCG',
    # 'pvotepreviouspvoteD',
    # 'pvotepreviouspvoteG',
    
    # 'pvotepreviouspreviousppar',
    # 'pvotepreviouspreviouspvoteC',
    # 'pvotepreviouspreviouspvoteCD',
    # 'pvotepreviouspreviouspvoteCG',
    # 'pvotepreviouspreviouspvoteD',
    # 'pvotepreviouspreviouspvoteG',
]
path = "data/data_merged_20250922.parquet"
data=filter_large_parquet(path, feature_list)
data=select_election_type(data, 1)
data=sample_df(data, 100)
data
#%% Test avec MLForecast


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
    lags=[
        1,
        # 2
        ],
    freq=5,
    target_transforms=[
        Differences([1])
        ],
)
#%%
import numpy as np

horizon = 2
level = [95]
year_start : int | None = 1973
year_end : int | None = 1988

train, test, selected_years = get_data_years(data, year_start, year_end, horizon)

# mlf.preprocess(
#     train,
#     id_col="codecommune",
#     time_col="annee",
#     target_col=y,
#     static_features = [
#         # "lat",
#         # "long"
#     ],
# )
# #%%

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

future_df = mlf.make_future_dataframe(h=horizon)

# Filter to the same communes as training
future_df = future_df[future_df["codecommune"].isin(train["codecommune"].unique())]
X_df = future_df.merge(
    test.drop(columns=[y], errors='ignore'),
    on=["codecommune", "annee"],
    how="left"
)

fcst = mlf.predict(
    h=horizon,
    X_df=X_df,
    level=level
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
#%% Some plots for 1 commune
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
#%% Grid search start end
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm

def rmse_grid_search_years(data, y, horizon=2, 
                           year_min=None, year_max=None, 
                           cmap="viridis"):
    """
    Computes RMSE for all valid combinations of (year_start, year_end)
    and plots a heatmap.
    """

    # all years in dataset
    years = np.sort(data["annee"].unique())
    if year_min is None:
        year_min = years[0]
    if year_max is None:
        year_max = years[-1]

    year_range = [y for y in years if year_min <= y <= year_max]

    # Initialize matrix
    rmse_matrix = pd.DataFrame(
        index=year_range, 
        columns=year_range, 
        dtype=float
    )

    # Loop over year_start, year_end
    for ys in tqdm(year_range):
        for ye in year_range:
            if ye <= ys:             # invalid interval
                continue

            try:
                train, test, selected_years = get_data_years(
                    data, ys, ye, horizon
                )
                mlf = MLForecast(
                    models=[
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
                    ],
                    lags=[
                        k for k in range(1,len(selected_years) - horizon-1)
                        ],
                    freq=5,
                    target_transforms=[
                        Differences([1])
                        ],
                )
                # Fit model
                mlf.fit(
                    train,
                    id_col="codecommune",
                    time_col="annee",
                    target_col=y,
                    static_features=[]
                )

                # Build future dataframe (required by MLForecast)
                future_df = mlf.make_future_dataframe(h=horizon)
                future_df = future_df[future_df["codecommune"].isin(train["codecommune"].unique())]

                # Merge test covariates if exist
                X_df = future_df.merge(
                    test.drop(columns=[y], errors="ignore"),
                    on=["codecommune", "annee"],
                    how="left"
                )

                # Predict
                fcst = mlf.predict(
                    h=horizon,
                    X_df=X_df,
                    level=[95]
                )
                # Compute RMSE on the first model
                for model in mlf.models_:
                    rmse=root_mean_squared_error(test[y],fcst[model])

                rmse_matrix.loc[ys, ye] = rmse

            except Exception as e:
                # Assign NaN on failure (e.g. not enough data)
                print(f'failed over {ys}, {ye}')
                print(e)
                rmse_matrix.loc[ys, ye] = np.nan
                continue

    # -------------------
    # PLOT THE HEATMAP
    # -------------------
    plt.figure(figsize=(10, 8))
    plt.imshow(rmse_matrix.T.values, 
               cmap=cmap, 
               origin='lower',
               aspect='auto')
    plt.colorbar(label="RMSE")
    plt.xticks(range(len(year_range)), year_range, rotation=90)
    plt.yticks(range(len(year_range)), year_range)
    plt.xlabel("year_start")
    plt.ylabel("year_end")
    plt.title("RMSE over (year_start, year_end) ranges")
    plt.tight_layout()
    plt.show()

    return rmse_matrix

rmse_map = rmse_grid_search_years(
    data=data,
    y=y,
    horizon=1,
    year_min=(year_min:=1932),
    year_max=(year_max:=2022),
    cmap='inferno'
)
#%% Plot 2D RMSE
years = np.sort(data["annee"].unique())
if year_min is None:
    year_min = years[0]
if year_max is None:
    year_max = years[-1]
year_range = [y for y in years if year_min <= y <= year_max]

plt.figure(figsize=(10, 8))
plt.imshow(rmse_map.T.values, 
            cmap='inferno', 
            origin='lower',
            aspect='auto')
plt.colorbar(label="RMSE")
plt.xticks(range(len(year_range)), year_range, rotation=90)
plt.yticks(range(len(year_range)), year_range)
plt.xlabel("year_start")
plt.ylabel("year_end")
plt.title("RMSE over (year_start, year_end) ranges")
plt.tight_layout()
plt.show()
#%% Plot 1D-MEAN RMSE
diff=rmse_map.mean(axis=0)
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(
    diff,
    label='Series A (Primary)',    # Label for the legend
    color='darkblue',              # Line color
    linestyle='-',                 # Line style
    linewidth=2,                   # Line thickness
    marker='o',                    # Marker style
    markersize=4,                  # Marker size
    alpha=0.8                      # Transparency
)
ax.grid(
    True,
    linestyle=':',
    alpha=0.6,
    color='lightgray'
)
plt.xticks(diff.index, rotation=70)