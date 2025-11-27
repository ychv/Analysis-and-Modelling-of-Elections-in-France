#%%
import pandas as pd
from dataImport import importCommuneData

feature_list = [
    'annee',
    'type',
    # 'pvoteppar',
    # 'pvotepvoteG',
    # 'pvotepvoteC',
    'pvotepvoteD',

    'popcommunes/percommu',
    'popcommunesvbbm/vbbmpauvresriches',
    'capitalimmobiliercommunes/capitalimmo',
    # 'revcommunes/nadult',
    # 'capitalimmobiliercommunes/surface',
    # 'agesexcommunes/perage_rank'
    ]

data = importCommuneData(89394,feature_list).sort_values(by=['type', 'annee'])
data=data[data["type"]==1]
data=data.drop(columns=["type", 
                        # "codecommune"
                        ])
y='pvotepvoteD'
pd.plotting.autocorrelation_plot(data[y])
data
#%% Train Test split
n = len(data)
train = data.iloc[: 9*n//10].copy()
test  = data.iloc[9*n//10:].copy()
#%% Test avec plus de features
from statsforecast import StatsForecast
from statsforecast.models import AutoETS,AutoRegressive,AutoARIMA,AutoCES

sf = StatsForecast(
    models=[
            AutoARIMA(),
            AutoRegressive(lags=1,alias="AutoRegLag1"),
            # AutoRegressive(lags=2,alias="AutoRegLag2"),
            # AutoRegressive(lags=3,alias="AutoRegLag3"),
            # AutoRegressive(lags=4,alias="AutoRegLag4"),
            # AutoRegressive(lags=5,alias="AutoRegLag5"),
            AutoCES(),
            AutoETS(),
            ],
    freq=5
)

horizon = 4
level = [95]
sf.fit(
    train,    
    id_col="codecommune",
    time_col="annee",
    target_col=y
)
fcst = sf.forecast(
    df=train, 
    h=horizon, 
    X_df=test.drop(columns=[y]), 
    level=level,
    id_col="codecommune",
    time_col="annee",
    target_col=y
)
sf.plot(
    data, 
    fcst, 
    level=level,
    id_col="codecommune",
    time_col="annee",
    target_col=y
)
#%% Display model parameters
num_model=0
print(sf.fitted_[0][num_model])
sf.fitted_[0][num_model].model_
#%% Test avec MLForecast
from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from utilsforecast.plotting import plot_series

mlf = MLForecast(
    models=[
        make_pipeline(StandardScaler(), LinearRegression()),
        make_pipeline(StandardScaler(), RandomForestRegressor()),
        make_pipeline(StandardScaler(), XGBRegressor()),
            ],
    freq=5
)

horizon = 4
level = [95]
mlf.fit(
    train,    
    id_col="codecommune",
    time_col="annee",
    target_col=y,
    static_features = [],
)
fcst = mlf.predict(
    h=horizon, 
    X_df=test.drop(columns=[y]), 
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
fig