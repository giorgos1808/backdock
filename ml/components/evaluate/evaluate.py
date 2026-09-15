import lightgbm as lgb

import argparse
import json
import os

import numpy as np
import pandas as pd
import mlflow

FEATURE_COLUMNS = ["lag_1", "lag_2", "rolling_avg_3", "month", "day_of_week", "supplier_code"]
TARGET_COLUMN = "quantity"


def mape(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    nonzero = actual != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--model-input", required=True)
    parser.add_argument("--result-output", required=True)
    args = parser.parse_args()

    test_df = pd.read_parquet(os.path.join(args.test_data, "test.parquet"))

    model = lgb.Booster(model_file=os.path.join(args.model_input, "model.txt"))
    model_predictions = model.predict(test_df[FEATURE_COLUMNS])

    baseline_predictions = test_df["rolling_avg_3"]

    model_mape = mape(test_df[TARGET_COLUMN], model_predictions)
    baseline_mape = mape(test_df[TARGET_COLUMN], baseline_predictions)
    passed = model_mape < baseline_mape

    with mlflow.start_run():
        mlflow.log_metric("model_mape", model_mape)
        mlflow.log_metric("baseline_mape", baseline_mape)
        mlflow.log_metric("passed", int(passed))

    os.makedirs(args.result_output, exist_ok=True)
    with open(os.path.join(args.result_output, "result.json"), "w") as f:
        json.dump({"model_mape": model_mape, "baseline_mape": baseline_mape, "passed": passed}, f)

    print(f"evaluate: model_mape={model_mape:.2f} baseline_mape={baseline_mape:.2f} passed={passed}")


if __name__ == "__main__":
    main()
