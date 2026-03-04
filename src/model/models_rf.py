import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from src.config import random_seed, workers
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


class RegressorWrapper(BaseEstimator):
    """
    A wrapper for scikit-learn compatible regressors.
    Works with RandomForest, ObliqueRandomForest, etc.
    """

    def __init__(self, base, **params):
        """
        Initialize the Wrapper with a Regressor base and provided parameters.
        Inputs:
            - base, example: sklearn.ensemble.RandomForestRegressor
            - params, kwargs, base parameters, Optional

        """
        self.base = base
        self.params = params
        self.model = self.base(**self.params)

    def __str__(self):
        """Used by TimeSeriesWrapper to set the key in MLForecast."""
        return self.base.__name__ if self.base else "UninitializedModel"

    def __repr__(self):
        """Standard sklearn-style representation."""
        return f"{self.__str__()}({self.params})"

    def fit(self, X, y):
        """
        Resets and fits the model on provided data.
        Inputs:
            - X, {array-like, sparse matrix} of shape (n_samples, n_features)
            - y, array-like of shape (n_samples,) or (n_samples, n_outputs)

        Returns self
        """
        self.model = clone(self.base(**self.params))
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts regression target on X.

        Inputs:
            - X, {array-like, sparse matrix} of shape (n_samples, n_features)

        Ouput:
            - y, ndarray of shape (n_samples,) or (n_samples, n_outputs)
        """
        return self.model.predict(X)

    def compute_permutation_importance(
        self,
        X_test: pd.DataFrame | np.ndarray,
        y_test: pd.Series | np.ndarray,
        random_state: int,
        perm_n_repeats: int,
        perm_scoring: str,
    ) -> pd.DataFrame:
        """
        Calculates permutation importance for the current fitted model.

        This technique measures the contribution of each feature by calculating the
        decrease in model performance when a single feature's values are randomly shuffled.

        Args:
            X_test (Union[pd.DataFrame, np.ndarray]): Testing features of shape (n_samples, n_features).
            y_test (Union[pd.Series, np.ndarray]): Testing target labels of shape (n_samples,).
            random_state (int): Seed for the random number generator to ensure reproducible results.
            perm_n_repeats (int): Number of times to permute each feature.
            perm_scoring (str): Scikit-learn scoring string (e.g., 'r2', 'neg_mean_squared_error').

        Returns:
            pd.DataFrame: A DataFrame sorted by importance, containing:
                - 'feature': Feature name or index.
                - 'importance_mean': Average decrease in score across repeats.
                - 'importance_std': Standard deviation of the decrease in score.
        """

        perm = permutation_importance(
            self.model,
            X_test,
            y_test,
            n_repeats=perm_n_repeats,
            random_state=random_state,
            n_jobs=self.params.get("n_jobs", int(np.ceil(workers / 2))),
            scoring=perm_scoring,
        )

        pi = pd.DataFrame(
            {
                "feature": (
                    X_test.columns
                    if hasattr(X_test, "columns")
                    else np.arange(X_test.shape[1])
                ),
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        y_pred = self.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"R2: {r2:.4f} | RMSE: {rmse:.4f}")
        print(f"y_test Std: {np.std(y_test):.3f} | Mean: {np.mean(y_test):.3f}")

        return pi, {"r2": float(r2), "rmse": float(rmse)}

    def tune_cv_hyperparams(
        self,
        X,
        y,
        param_grid,
        cv_splits=5,
        scoring="r2",
        verbose=0,
        refit=True,
    ):
        """
        Time-Series aware GridSearch tuning.
        Note: random_state is removed from cv as TimeSeriesSplit does not shuffle.
        """
        base_instance = self.base(**self.params)

        cv = TimeSeriesSplit(n_splits=cv_splits)

        gs = GridSearchCV(
            estimator=base_instance,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=self.params.get("n_jobs", workers),
            verbose=verbose,
            refit=refit,
        )

        gs.fit(X, y)

        if refit:
            self.params.update(gs.best_params_)
            self.model = gs.best_estimator_

        results = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score")
        return gs.best_estimator_, results, gs.best_params_, float(gs.best_score_)
