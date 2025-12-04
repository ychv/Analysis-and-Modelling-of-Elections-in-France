#%%
import pandas as pd
from dataImport import importCommuneData

feature_list = [
    'annee',
    'type',

    # y:='pvoteppar',
    # y:='pvotepvoteG',
    # y:='pvotepvoteC',
    y:='pvotepvoteD',

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

data = importCommuneData(89394,feature_list).sort_values(by=['type', 'annee'])
data=data[data["type"]==1]
data=data.drop(columns=["type", 
                        # "codecommune"
                        ])
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
        make_pipeline(LinearRegression()),
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
        make_pipeline(XGBRegressor()),
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
    static_features = [],
)
#%%
horizon = 3
level = [95]
n = len(data)
train = data.iloc[: 9*n//10].copy()
test  = data.iloc[9*n//10:].copy()
test = test.iloc[:horizon,:]

mlf.fit(
    train,    
    id_col="codecommune",
    time_col="annee",
    target_col=y,
    static_features = [],
)
fcst = mlf.predict(
    h=horizon, 
    X_df=test, 
    level=level,
)
fig = plot_series(
    data, 
    fcst, 
    max_ids=4, 
    plot_random=False,
    id_col="codecommune",
    time_col="annee",
    target_col=y,
)

print(data[y].describe())
for model in mlf.models_:
    rmse=root_mean_squared_error(test[y],fcst[model])
    print("\n____________________________")
    print(f"Model {model}")
    print(f"RMSE : {rmse}")
fig
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