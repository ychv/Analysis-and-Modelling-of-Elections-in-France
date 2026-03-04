import argparse

from src.model.models_rf import RegressorWrapper
from sklearn.ensemble import RandomForestRegressor
from src.config import (
    full_dataset_path,
    random_seed,
    param_grid_rf,
)
from src.data.data_handler import DataHandler


def main(args):
    data_handler = DataHandler(full_dataset_path, random_seed)
    data_handler.load_data(args.s_year, args.l_year, args.type)
    X, y = data_handler.split_X_y(args.predict_col)

    model = RegressorWrapper(RandomForestRegressor)
    best_estimator, _, best_params, best_score = model.tune_cv_hyperparams(
        X, y, param_grid=param_grid_rf, cv_splits=5, scoring="r2", verbose=1
    )

    print(f"\nBest Parameters Found: {best_params}")
    print(f"Best CV R2 Score: {best_score:.4f}")

    _, X_val_spatial, _, y_val_spatial = data_handler.split_train_test(args.predict_col)

    pi_results, metrics_dict = model.compute_permutation_importance(
        X_val_spatial,
        y_val_spatial,
        random_state=random_seed,
        perm_n_repeats=10,
        perm_scoring="r2",
    )

    print("\nTop 15 Most Important Features (Permutation Importance):")
    print(pi_results.head(15))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RandomForest pipeline with Temporal CV and Permutation Importance."
    )

    parser.add_argument(
        "--s_year",
        type=int,
        required=True,
        help="Starting year for the dataset slice (inclusive).",
    )
    parser.add_argument(
        "--l_year",
        type=int,
        required=True,
        help="Last year for the dataset slice.",
    )
    parser.add_argument(
        "--type",
        type=int,
        choices=[0, 1],
        required=True,
        help="Election type: 0 for 'presidentielle', 1 for 'legislative'.",
    )

    parser.add_argument(
        "--predict_col",
        type=str,
        default="pvotepvoteD",
        help="Col to be predicted by the model",
    )

    args = parser.parse_args()
    main(args)
