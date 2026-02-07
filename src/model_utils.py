import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import TransformedTargetRegressor
from sktree.ensemble import ObliqueRandomForestRegressor

### ----------------------------- Random forest ----------------------------------------------------


def train_test_model_rf(
    X,
    y,
    n_estimators=600,
    min_samples_leaf=2,
    max_depth=6,
    min_samples_split=4,
    train_indices=None,
):
    if train_indices is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = (
            X[train_indices],
            X[~train_indices],
            y_train[train_indices],
            y_test[~train_indices],
        )

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        n_jobs=-1,
        bootstrap=True,
        random_state=42,
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    ### Métriques classiques d'évaluation du modèle
    print("R2:", r2_score(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
    print(f"Std de y: {np.std(y_test):.3f} --- Moyenne de y: {np.mean(y_test):.3f}")

    ### Calcul de l'importance par permutation des features
    perm = permutation_importance(
        rf, X_test, y_test, n_repeats=15, random_state=42, n_jobs=-1
    )

    pi = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    print("Permutation Importance: ", pi)
    return rf, pi


### ----------------------------- HGR ----------------------------------------------------


def train_test_model_HGR(
    X,
    y,
    max_iter=2000,
    learning_rate=0.05,
    max_depth=6,
    train_indices=None,
):
    if train_indices is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = (
            X[train_indices],
            X[~train_indices],
            y_train[train_indices],
            y_test[~train_indices],
        )

    reg = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=learning_rate,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        max_depth=max_depth,
        min_samples_leaf=50,
        l2_regularization=0.1,
        max_bins=255,
        random_state=42,
    )
    model = TransformedTargetRegressor(regressor=reg, func=logit, inverse_func=expit)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    ### Métriques classiques d'évaluation du modèle
    print("R2:", r2_score(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
    print(f"Std de y: {np.std(y_test):.3f} --- Moyenne de y: {np.mean(y_test):.3f}")

    ### Calcul de l'importance par permutation des features
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=15, random_state=42, n_jobs=-1
    )

    pi = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    print("Permutation Importance: ", pi)
    return model, pi


def train_test_model_oblique_rf_regressor(
    X,
    y,
    n_estimators=600,
    min_samples_leaf=2,
    max_depth=6,
    min_samples_split=4,
    feature_combinations=None,
    train_indices=None,
    test_size=0.15,
    random_state=42,
    perm_n_repeats=15,
):

    # --- split ---
    if train_indices is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = (
            X[train_indices],
            X[~train_indices],
            y_train[train_indices],
            y_test[~train_indices],
        )

    # --- model ---
    rf = ObliqueRandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        n_jobs=-1,
        bootstrap=True,
        random_state=random_state,
        feature_combinations=feature_combinations,
    )

    rf.fit(X_train, y_train)

    # --- predict + metrics ---
    y_pred = rf.predict(X_test)

    print("R2:", r2_score(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
    print(f"Std de y: {np.std(y_test):.3f} --- Moyenne de y: {np.mean(y_test):.3f}")

    # --- permutation importance ---
    perm = permutation_importance(
        rf,
        X_test,
        y_test,
        n_repeats=perm_n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring="r2",
    )

    feature_names = X.columns if hasattr(X, "columns") else np.arange(X_train.shape[1])

    pi = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    print("Permutation Importance:\n", pi)
    return rf, pi
