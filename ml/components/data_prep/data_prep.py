import argparse
import json
import os

import pandas as pd

from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient

from sqlalchemy import create_engine, text

HISTORY_QUERY = """
    SELECT
        p.id AS product_id,
        so.order_date,
        so.supplier_name,
        oi.quantity
    FROM order_items oi
    JOIN supplier_orders so ON so.id = oi.order_id
    JOIN products p ON p.id = oi.product_id
    WHERE oi.quantity IS NOT NULL
    ORDER BY p.id, so.order_date ASC NULLS FIRST
"""


def build_features(history: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per (product, order) with lag/rolling/seasonality features,
    plus the supplier_name -> supplier_code mapping used to build them.

    Pure function (no I/O) so it can be unit-tested directly against a
    synthetic DataFrame without a database or Azure credentials.

    The mapping must travel with the trained model (see main() below,
    written to supplier_categories.json) — pandas assigns codes by sorting
    category names alphabetically, and that sort order shifts whenever a
    new supplier name enters the mix, so "Acme" being code 0 today doesn't
    mean it's code 0 in the next training run. Whatever scores with this
    model later needs *this exact* mapping, not a freshly recomputed one.
    """
    history = history.copy()
    history["order_date"] = pd.to_datetime(history["order_date"])
    history = history.sort_values(["product_id", "order_date"])

    grouped = history.groupby("product_id")["quantity"]
    history["lag_1"] = grouped.shift(1)
    history["lag_2"] = grouped.shift(2)
    history["rolling_avg_3"] = grouped.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    history["month"] = history["order_date"].dt.month
    history["day_of_week"] = history["order_date"].dt.dayofweek

    supplier_categorical = history["supplier_name"].astype("category")
    history["supplier_code"] = supplier_categorical.cat.codes
    category_mapping = {name: code for code, name in enumerate(supplier_categorical.cat.categories)}

    features = history.dropna(subset=["lag_1"]).reset_index(drop=True)
    return features, category_mapping


def split(features: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shuffled = features.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    return shuffled.iloc[:train_end], shuffled.iloc[train_end:val_end], shuffled.iloc[val_end:]


def fetch_database_url(key_vault_url: str, managed_identity_client_id: str | None) -> str:
    credential = (
        ManagedIdentityCredential(client_id=managed_identity_client_id)
        if managed_identity_client_id
        else ManagedIdentityCredential()
    )
    client = SecretClient(vault_url=key_vault_url, credential=credential)
    return client.get_secret("database-url").value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-vault-url", required=True)
    parser.add_argument("--managed-identity-client-id", default=None)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--test-output", required=True)
    args = parser.parse_args()

    database_url = fetch_database_url(args.key_vault_url, args.managed_identity_client_id)
    engine = create_engine(database_url)
    with engine.connect() as conn:
        history = pd.read_sql(text(HISTORY_QUERY), conn)

    history["product_id"] = history["product_id"].astype(str)

    features, category_mapping = build_features(history)
    train_df, val_df, test_df = split(features)

    for output_dir, df, filename in (
        (args.train_output, train_df, "train.parquet"),
        (args.val_output, val_df, "val.parquet"),
        (args.test_output, test_df, "test.parquet"),
    ):
        os.makedirs(output_dir, exist_ok=True)
        df.to_parquet(os.path.join(output_dir, filename))

        with open(os.path.join(output_dir, "supplier_categories.json"), "w") as f:
            json.dump(category_mapping, f)

    print(f"data_prep: {len(features)} total rows -> train={len(train_df)} val={len(val_df)} test={len(test_df)}")


if __name__ == "__main__":
    main()
