#Truc moche pour que les imports de src marchent
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.data_utils import filter_large_parquet
from src.config import full_dataset_path
from src.data.prepare_data import sample_df, split_serie_temp, prepare_data
from src.model.models_rf import RegressorWrapper
from src.model.serie_temp import TimeSeriesWrapper

from config import feature_list, horizon, num_communes, election_type, year_start, year_end
from config import static_features, id_col, time_col, y
from config import model, params, lags, freq, target_transforms

import polars as pl
#Fetch then prepare data
data=filter_large_parquet(
    file_path=full_dataset_path, 
    columns_to_keep=feature_list,
    dropna_subset=feature_list,
    filter=pl.col("annee").is_between(year_start, year_end)

)

data=sample_df(data, num_communes)
data=prepare_data(data, 0, election_type)
train, test, _ = split_serie_temp(data, horizon=horizon)

#Define then fit model
mlf = TimeSeriesWrapper(
    target_col=y,
    id_col=id_col,
    time_col=time_col,
    static_features=static_features
)

# CHANGE IMPORTS IF YOU NEED MORE MODELS
# from config import model2, params2
mlf.setup_mlf(
    wrapped_models=[
        RegressorWrapper(model,**params),
        # RegressorWrapper(model2,**params2), #ADD more models if needed
    ],
    lags=lags,
    freq=freq,
    target_transforms=target_transforms,
)

mlf.fit(train,dropna=False)

all_results=mlf.run_all_permutation_importances(test, dropna=False)