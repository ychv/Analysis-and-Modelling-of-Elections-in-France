import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from src.config import random_seed


class RegressorWrapper:
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

    def train_test_with_permutation_importance(
        self,
        X,
        y,
        test_size=0.15,
        random_state=42,
        perm_n_repeats=15,
        perm_scoring="r2",
    ):
        """
        Splits provided dataset (X, y) into test and train sub datasets.
        Fits on the train then evaluates performance on test data.
        Also performs permutation importance onto the data's features.

        Returns trained model, permuation importance and test performance metrics.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Update random state if the model supports it
        if "random_state" in self.params:
            self.params["random_state"] = random_state

        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"R2: {r2:.4f} | RMSE: {rmse:.4f}")
        print(f"y_test Std: {np.std(y_test):.3f} | Mean: {np.mean(y_test):.3f}")

        perm = permutation_importance(
            self.model,
            X_test,
            y_test,
            n_repeats=perm_n_repeats,
            random_state=random_state,
            n_jobs=self.params.get("n_jobs", -1),
            scoring=perm_scoring,
        )

        pi = pd.DataFrame(
            {
                "feature": (
                    X.columns if hasattr(X, "columns") else np.arange(X.shape[1])
                ),
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)

        return self.model, pi, {"r2": float(r2), "rmse": float(rmse)}

    def tune_cv_hyperparams(
        self,
        X,
        y,
        param_grid,
        cv_splits=5,
        scoring="r2",
        random_state=random_seed,
        verbose=1,
        refit=True,
    ):
        """
        Regular GrisSearch tuning.

        Inputs:
            - X, pd.DataFrame corresponding to (n_samples, n_features)
            - y, pd.Series corresponding to the taget (n_samples, )
            - param_grid, dict, corresponds to the base model params to be optimized
            - cv_split, int, number of cross validation splits.
            - scoring: str, metric used in param selection


        Returns:
            - Tuple: best_estimator, results of tuning, best params found, best score during tuning.
        """
        base_instance = self.base(**self.params)

        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

        gs = GridSearchCV(
            estimator=base_instance,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=self.params.get("n_jobs", -1),
            verbose=verbose,
            refit=refit,
        )

        gs.fit(X, y)

        if refit:
            self.params.update(gs.best_params_)
            self.model = gs.best_estimator_

        results = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score")
        return gs.best_estimator_, results, gs.best_params_, float(gs.best_score_)
