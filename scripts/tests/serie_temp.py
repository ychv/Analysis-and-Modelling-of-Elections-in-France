#Truc moche pour que les imports de src marchent
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.data_utils import filter_large_parquet
from src.config import full_dataset_path, random_seed
from src.data.prepare_data import sample_df, split_serie_temp, prepare_data, get_display_window_lagged
from src.data.prepare_data import create_election_mapping, apply_time_mapping, calculate_election_duration
from src.model.models_rf import RegressorWrapper
from src.model.serie_temp import TimeSeriesWrapper
from mlforecast import MLForecast

from config import feature_list, horizon, num_communes, election_type, year_start, year_end
from config import static_features, id_col, time_col, y
from config import model, params, lags, freq, target_transforms

import polars as pl

extended_start_year=get_display_window_lagged(full_dataset_path, election_type, year_start, year_end, lags)

# Charger les données avec la borne étendue
data = filter_large_parquet(
    file_path=full_dataset_path, 
    columns_to_keep=feature_list,
    dropna_subset=[y],
    filter=(pl.col("annee").is_between(extended_start_year, year_end), pl.col("type") == election_type)
)

if num_communes>0:
    data=sample_df(data, num_communes, random_seed)

data= calculate_election_duration(data, time_col, id_col)
train, test, years = split_serie_temp(data, horizon=horizon)
# We use mapping with time difference instead of just time because elections don't happen regularly (ex : 1981, 1986, 1988)
year_to_idx, idx_to_year = create_election_mapping(years)
print(year_to_idx)
train, test = apply_time_mapping(train, time_col, year_to_idx), apply_time_mapping(test, time_col, year_to_idx)

#Define then fit model
mlf = TimeSeriesWrapper(
    target_col=y,
    id_col=id_col,
    time_col=time_col,
    static_features=static_features
)

# CHANGE IMPORTS IF YOU NEED MORE MODELS
# from config import model2, params2
path=Path(f"results/models/{year_start}_2_{year_end}_h{horizon}_el{election_type}_nco{num_communes}")
if list(path.glob("*.pkl"))!=[]:
    mlf.mlf=MLForecast.load(path)
    print('Model Loaded')
else:
    mlf.setup_mlf(
        wrapped_models=[
            RegressorWrapper(model,**params),
            # RegressorWrapper(model2,**params2), #ADD more models if needed
        ],
        lags=lags,
        freq=freq,
        target_transforms=target_transforms,
    )
    loaded=False
    mlf.fit(train,dropna=False)
    mlf.mlf.save(path)
    print("Model Trained")

fcst=mlf.run_forecast(test, horizon)

cv_rmse = mlf.get_cv_rmse(train, window_size=2, dropna=False)

comparison_df=mlf.evaluate_performance(test, idx_to_year, cv_rmse)