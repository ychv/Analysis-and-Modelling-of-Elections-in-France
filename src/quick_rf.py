import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error


def train_test_model_rf(
    X,
    y,
    n_estimators=600,
    min_samples_leaf=2,
    max_depth=6,
    min_samples_split=4,
    train_indices: np.array | None = None,
):
    if not train_indices:
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
