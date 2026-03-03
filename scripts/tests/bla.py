#Truc moche pour que les imports de src marchent
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.data_utils import filter_large_parquet, get_all_years
from src.config import full_dataset_path, random_seed
from src.data.prepare_data import sample_df, split_serie_temp, prepare_data, get_display_window_lagged
from src.data.prepare_data import create_election_mapping, apply_time_mapping, calculate_election_duration
from src.model.models_rf import RegressorWrapper
from src.model.serie_temp import TimeSeriesWrapper

from config import feature_list, horizon, num_communes, election_type, year_start, year_end
from config import static_features, id_col, time_col, y
from config import model, params, lags, freq, target_transforms

import polars as pl
print(election_type)
print(get_all_years(full_dataset_path, election_type))