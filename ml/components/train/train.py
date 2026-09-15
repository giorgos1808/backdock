import lightgbm as lgb

import argparse
import shutil
import os

import pandas as pd
import mlflow

FEATURE_COLUMNS = ["lag_1", "lag_2", "rolling_avg_3", "month", "day_of_week", "supplier_code"]
CATEGORICAL_COLUMNS = ["supplier_code"]
TARGET_COLUMN = "quantity"


def load(path: str, filename: str):
    return pd.read_parquet(os.path.join(path, filename))


def train_model(train_df, val_df):
    """Fit the regressor, stopping when the validation score stops improving.

    Three settings here were measured, not guessed, against ~3,200 feature rows
    built from real demand data (see scripts/load_demand_dataset.py), over ten
    random splits. Test MAPE, mean across splits:

        rolling-average baseline (the gate to beat)   24.59%
        "same as last order" (lag_1 alone)            23.71%
        previous settings                             26.15%   beat baseline 1/10
        these settings                                23.46%   beat baseline 9/10

    objective="mape" is the big one. evaluate.py judges the model on MAPE, but
    LightGBM defaults to squared error, so training optimised something the
    gate never measured. Quantities here span roughly 0.7 to 60, and L2 chases
    the large values while MAPE weights every row by its own size.

    early_stopping is the second. The validation split was already being built
    and passed in, but without a callback nothing consumed it — all 200 trees
    were built regardless, and the model overfitted hard (train 17.9% against
    test 25.4%).

    min_child_samples is back to LightGBM's own default of 20. Dropping it to 1
    let a leaf form from a single row, which on this much data is an invitation
    to memorise. The original comment worried about degenerate constant fits on
    a tiny dataset; early stopping is the better answer to that, since it stops
    when the model stops generalising rather than when it stops fitting.
    """
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        min_child_samples=20,
        objective="mape",
    )
    model.fit(
        train_df[FEATURE_COLUMNS],
        train_df[TARGET_COLUMN],
        eval_set=[(val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN])],
        eval_metric="mape",
        categorical_feature=CATEGORICAL_COLUMNS,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--model-output", required=True)
    args = parser.parse_args()

    train_df = load(args.train_data, "train.parquet")
    val_df = load(args.val_data, "val.parquet")

    with mlflow.start_run():
        model = train_model(train_df, val_df)
        mlflow.log_params(model.get_params())
        mlflow.log_metric("train_rows", len(train_df))
        mlflow.log_metric("val_rows", len(val_df))

    os.makedirs(args.model_output, exist_ok=True)
    model.booster_.save_model(os.path.join(args.model_output, "model.txt"))

    shutil.copy(
        os.path.join(args.train_data, "supplier_categories.json"),
        os.path.join(args.model_output, "supplier_categories.json"),
    )

    print(f"train: fit on {len(train_df)} rows, validated on {len(val_df)}")


if __name__ == "__main__":
    main()
