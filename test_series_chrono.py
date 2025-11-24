#%%
import pandas as pd

path = "data/data_merged_20250922.parquet"

def importCommuneData(postcode,feature_list = None):

    path = "data/data_merged_20250922.parquet"
    col = None
    if feature_list is not None:
        col = ["codecommune"] + feature_list

    sel = [("codecommune","==",str(postcode))]
    data = pd.read_parquet(path,engine='pyarrow',
                           columns = col,
                           filters=sel
                           )

    return data

feature_list = [
    'annee',
    'type',
    'pvoteppar',
    # 'pvotepvoteG',
    # 'pvotepvoteC','pvotepvoteD',
    # 'agesexcommunes/perage_rank'
    ]

data = importCommuneData(91100,feature_list).sort_values(by=['type', 'annee'])
data=data[data["type"]==1]
data=data.drop(columns=["type", 
                        # "codecommune"
                        ])

pd.plotting.autocorrelation_plot(data["pvoteppar"])
#%%
from statsforecast import StatsForecast
from statsforecast.models import AutoETS,AutoRegressive,AutoARIMA,AutoCES
from sklearn.model_selection import train_test_split
n=len(data)
X_train, X_test =  data[:9*n//10], data[9*n//10:]
X_train
#%%
AutoARIMA()
sf = StatsForecast(
    models=[
        AutoARIMA(
    )],
    freq=5
)
sf.fit(
    X_train,    
    id_col="codecommune",
    time_col="annee",
    target_col="pvoteppar"
    )
forecast_df = sf.predict(h=5, level=[90])
sf.plot(X_train, 
        forecast_df, 
        level=[90],
    id_col="codecommune",
    time_col="annee",
    target_col="pvoteppar"
        )
#%%
sf.plot(
    X_train, 
        X_test, 
        level=[90],
    id_col="codecommune",
    time_col="annee",
    target_col="pvoteppar"
)
#%%
sf.fitted_[0][0].model_
