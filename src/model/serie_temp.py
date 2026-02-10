import pandas as pd
from typing import List, Optional
from sklearn.metrics import root_mean_squared_error
from .models_rf import RegressorWrapper
from mlforecast import MLForecast
from collections import Counter
from typing import Dict, Any
from sklearn.inspection import permutation_importance

from config import random_seed

class TimeSeriesWrapper:
    """
    Handles the high-level forecasting logic, evaluation, and 
    integration between MLForecast and RegressorWrapper.
    """

    def __init__(self, target_col: str, id_col: str, time_col: str, static_features: list[str]):
        self.y = target_col
        self.id_col = id_col
        self.time_col = time_col
        self.static_features=static_features
        self.mlf=None
        self.fcst = None
        self.comparison_df = None

    def setup_mlf(self, wrapped_models: List[RegressorWrapper], lags: List[int], freq:int, target_transforms: List):
        """
        Initializes MLForecast with a list of RegressorWrappers.
        Handles multiples models having the same name
        """        
        # We create a dictionary where the key is the string name of the model
        # This is what MLForecast uses in its __repr__
        models_dict = {}
        name_counts = Counter()

        for wrapper in wrapped_models:
            # Get the base name (e.g., 'RandomForestRegressor')
            base_name = str(wrapper)
            name_counts[base_name] += 1
            
            # If it's a duplicate, append a suffix (e.g., 'RandomForestRegressor_2')
            if name_counts[base_name] > 1:
                unique_name = f"{base_name}_{name_counts[base_name]}"
            else:
                unique_name = base_name
            
            models_dict[unique_name] = wrapper  
                  
        self.mlf = MLForecast(
            models=models_dict,
            lags=lags,
            freq=freq,
            target_transforms=target_transforms,
        )

    def fit(self,train,dropna=False):
        self.mlf=self.mlf.fit(train,self.id_col,self.time_col,self.y,self.static_features,dropna)
        return self.mlf

    def run_forecast(
        self, 
        test_data: pd.DataFrame, 
        horizon: int, 
    ) -> pd.DataFrame:
        """
        Executes the forecast, merges with ground truth, and computes errors.
        """        
        # 1. Generate future feature grid
        future_df = self.mlf.make_future_dataframe(h=horizon)
        future_df = future_df[future_df[self.id_col].isin(test_data[self.id_col].unique())]
        
        # 2. Merge exogenous features from the test set
        X_df = future_df.merge(
            test_data.drop(columns=[self.y], errors='ignore'),
            on=[self.id_col, self.time_col],
            how="left"
        )

        # 3. Predict using MLForecast
        fcst = self.mlf.predict(
            h=horizon,
            X_df=X_df.drop(columns=self.static_features, errors='ignore'),
        )

        self.fcst=fcst
        return self.fcst

    def evaluate_performance(self, test_data):
        """
        Prints RMSE, relative error, and identifies the biggest outliers.
        """
        #Create comparison dataframe
        self.comparison_df = self.fcst.merge(
            test_data[[self.id_col, self.time_col, self.y]], 
            on=[self.id_col, self.time_col], 
            how="inner"
        )
        if self.comparison_df is None:
            raise ValueError("You must run run_forecast() before evaluating.")

        print(f"Target variable statistics ({self.y}):")
        print(self.comparison_df[self.y].describe())

        for model_name in self.mlf.models:
            # Calculate metrics
            actual = self.comparison_df[self.y]
            predicted = self.comparison_df[model_name]
            
            rmse = root_mean_squared_error(actual, predicted)
            mean_val = actual.mean()
            relative_err = (rmse / mean_val) * 100 if mean_val != 0 else 0

            print("\n" + "_"*30)
            print(f"MODEL: {model_name}")
            print(f"RMSE: {rmse:.4f}")
            print(f"Relative Error: {relative_err:.2f}%")

            # Store error for outlier analysis
            error_col = f'abs_error_{model_name}'
            self.comparison_df[error_col] = (actual - predicted).abs()
            
            print("\nTop 5 Biggest Errors:")
            print(self.comparison_df.sort_values(error_col, ascending=False).head(5))
        return self.comparison_df

    def get_model_from_mlf(self, model_name: str):
        """
        Helper to retrieve a specific RegressorWrapper from the MLForecast object.
        Useful if you want to call .tune_cv_hyperparams() on a fitted model.
        """
        return self.mlf.models[model_name]

    def run_all_permutation_importances(
        self,
        test_df: pd.DataFrame,
        dropna: bool=False,
        perm_n_repeats: int = 15,
        perm_scoring: str = "r2",
        random_state: int = random_seed
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluates permutation importance for ALREADY FITTED models.
        """
        if self.mlf is None:
            raise ValueError("MLForecast instance not initialized.")

        # 1. Transform the test data to match the model's features (Lags, etc.)
        # We set fitted=True to ensure it uses the transforms learned during training
        prep_test = self.mlf.preprocess(
            test_df, 
            id_col=self.id_col, 
            time_col=self.time_col, 
            target_col=self.y,
            static_features=self.static_features,
            dropna=dropna
        )

        # 2. Extract X and y for the Scikit-Learn evaluation
        cols_to_drop = [self.id_col, self.time_col, self.y]
        X_test = prep_test.drop(columns=cols_to_drop, errors='ignore')
        y_test = prep_test[self.y]

        all_results = {}

        for model_name, wrapper in self.mlf.models_.items():
            print(f"\nEvaluating Importance for: {model_name}")

            pi_df=wrapper.compute_permutation_importance(
                X_test, 
                y_test,
                random_state=random_state,
                perm_n_repeats=perm_n_repeats,
                perm_scoring=perm_scoring,
            )

            all_results[model_name] = {"importance": pi_df}
            print(pi_df.head(5))

        return all_results