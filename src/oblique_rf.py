import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

from sktree.ensemble import ObliqueRandomForestRegressor


class ObliqueRF:
    """
    Simple wrapper around sktree.ensemble.ObliqueRandomForestRegressor.
    """

    def __init__(
        self,
        n_estimators=600,
        max_depth=6,
        min_samples_leaf=2,
        min_samples_split=4,
        feature_combinations=None,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            feature_combinations=feature_combinations,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.model = ObliqueRandomForestRegressor(**self.params)

    def fit(self, X, y):
        self.model = ObliqueRandomForestRegressor(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X):
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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.params["random_state"] = random_state
        self.fit(X_train, y_train)

        y_pred = self.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5

        print("R2:", r2)
        print("RMSE:", rmse)
        print(f"Std de y: {np.std(y_test):.3f} --- Moyenne de y: {np.mean(y_test):.3f}")

        perm = permutation_importance(
            self.model,
            X_test,
            y_test,
            n_repeats=perm_n_repeats,
            random_state=random_state,
            n_jobs=self.params["n_jobs"],
            scoring=perm_scoring,
        )

        pi = pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)

        print("Permutation Importance:\n", pi)

        metrics = {"r2": float(r2), "rmse": float(rmse)}
        return self.model, pi, metrics

    def tune_cv_hyperparams(
        self,
        X,
        y,
        cv_splits=5,
        scoring="r2",
        random_state=42,
        n_estimators_grid=(300, 600, 900),
        max_depth_grid=(None, 6, 10),
        feature_combinations_grid=(None, 0.3, 0.6, 1.0),
        verbose=0,
        refit=True,
    ):
        """
        Tunes 3 impactful hyperparams:
          - n_estimators
          - max_depth
          - feature_combinations
        """

        base = ObliqueRandomForestRegressor(
            min_samples_leaf=self.params["min_samples_leaf"],
            min_samples_split=self.params["min_samples_split"],
            bootstrap=self.params["bootstrap"],
            n_jobs=self.params["n_jobs"],
            random_state=random_state,
        )

        param_grid = {
            "n_estimators": list(n_estimators_grid),
            "max_depth": list(max_depth_grid),
            "feature_combinations": list(feature_combinations_grid),
        }

        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

        gs = GridSearchCV(
            estimator=base,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=self.params["n_jobs"],
            verbose=verbose,
            refit=refit,
            return_train_score=True,
        )

        gs.fit(X, y)

        print("Best params:", gs.best_params_)
        print("Best CV score:", gs.best_score_)

        results = (
            pd.DataFrame(gs.cv_results_)
            .sort_values("rank_test_score")
            .reset_index(drop=True)
        )

        if refit:
            self.params.update(gs.best_params_)
            self.params["random_state"] = random_state
            self.model = gs.best_estimator_

        return gs.best_estimator_, results, gs.best_params_, float(gs.best_score_)

    def feature_importances(self):
        """
        Impurity-based importances (MDI).
        """
        return pd.Series(
            self.model.feature_importances_, index=self.model.feature_names_in_
        ).sort_values(ascending=False)
