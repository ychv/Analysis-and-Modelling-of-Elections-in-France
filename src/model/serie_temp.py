import pandas as pd
from typing import List, Optional
from sklearn.metrics import root_mean_squared_error,r2_score
from .models_rf import RegressorWrapper
from mlforecast import MLForecast
from collections import Counter
from typing import Dict, Any
from sklearn.inspection import permutation_importance
from src.data.prepare_data import reverse_time_mapping

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
    
    def get_cv_rmse(
        self, 
        train:pd.DataFrame, 
        window_size:int=2, 
        dropna:bool=False
        ) -> dict:
        """
        Calcule la RMSE moyenne par backtesting sur les données historiques.
        window_size: nombre d'élections à tester en remontant le temps.
        """
        cv_df = self.mlf.cross_validation(
            df=train,
            n_windows=window_size,
            h=1,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.y,
            static_features=self.static_features,
            step_size=1,
            dropna=dropna,
            refit=True # On réentraîne à chaque fenêtre pour être rigoureux
        )

        # 2. Calcul de la RMSE pour chaque modèle enregistré dans le wrapper
        performance_report = {}
        
        for model_name in self.mlf.models.keys():
            # Nixtla retourne un DF avec 'y' (réel) et le nom du modèle (prédiction)
            actuals = cv_df[self.y]
            preds = cv_df[model_name]
            
            # Nettoyage des NaNs éventuels pour le calcul
            mask = actuals.notnull() & preds.notnull()
            rmse = root_mean_squared_error(actuals[mask], preds[mask])
            
            performance_report[model_name] = (train[self.time_col].unique(), rmse)
            
        return performance_report
    
    def evaluate_performance(self, test_data, inverse_mapping, cv_rmse=None):
        """
        Calcule le RMSE et le R2 pour chaque année prédite et identifie les outliers.
        """

        comparison_df = self.fcst.merge(
            test_data[[self.id_col, self.time_col, self.y]], 
            on=[self.id_col, self.time_col], 
            how="inner"
        )

        if comparison_df is None or comparison_df.empty:
            raise ValueError("La fusion a échoué. Vérifiez run_forecast() et vos colonnes temporelles.")
        
        comparison_df = reverse_time_mapping(comparison_df, self.time_col, inverse_mapping)

        print(f"Statistiques de la variable cible ({self.y}) :")
        print(comparison_df[self.y].describe())

        for model_name in self.mlf.models:
            print("\n" + "="*50)
            print(f" ANALYSE DU MODÈLE : {model_name} ")
            print("="*50)

            # 2. Calcul des métriques groupées par année
            def get_metrics(group):
                actual = group[self.y]
                pred = group[model_name]
                return pd.Series({
                    'RMSE': root_mean_squared_error(actual, pred),
                    'R2': r2_score(actual, pred),
                    'MAE': (actual - pred).abs().mean(),
                    'N_Samples': len(group)
                })

            yearly_metrics = comparison_df.groupby(self.time_col).apply(get_metrics, include_groups=False)

            if cv_rmse:
                cv_years, cv = cv_rmse[model_name]
                cv_years = [int(inverse_mapping[k]) for k in cv_years]
                print(f'RMSE over elections {cv_years}: {cv}')
            
            print("\nMétriques par année :")
            print(yearly_metrics.to_string())

            # 3. Calcul global pour rappel
            total_rmse = root_mean_squared_error(comparison_df[self.y], comparison_df[model_name])
            total_r2 = r2_score(comparison_df[self.y], comparison_df[model_name])
            print(f"\nGLOBAL -> RMSE: {total_rmse:.4f} | R2: {total_r2:.4f}")
        self.comparison_df=comparison_df
        return comparison_df

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
        random_state: int = random_seed,
        num_display: int = 5,
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
            print(pi_df.head(num_display))

        return all_results