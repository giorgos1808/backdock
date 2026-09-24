"""Load a public retail demand dataset into supplier_orders/order_items so the
forecasting pipeline has real history to train on.

Built for FreshRetailNet-50K (CC-BY-4.0, fresh/perishable grocery, ~90 days of
daily per-store per-product sales):

    https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K

    curl -sSL -o train.parquet \\
      https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/main/data/train.parquet

    python -m scripts.load_demand_dataset --parquet train.parquet --stores 3

WHAT THIS IS AND ISN'T
----------------------
The dataset is *daily store sales*. This app models *supplier orders*. Those
aren't the same thing, so the loader makes one explicit modelling assumption:

    a period's total sales ≈ the quantity that had to be reordered to
    replenish it

That gives each product a periodic order series with realistic level,
seasonality and noise — which is exactly the shape
ml/components/data_prep/data_prep.py consumes (order_date, supplier_name,
quantity per product). It is a stand-in for real order history, not a
substitute for it. Anything you conclude about model accuracy here is a
statement about this proxy, not about your own suppliers.

Product names are placeholders: the source dataset is anonymised, with integer
product and category ids and no names. So this data is useful for the
forecaster and useless for exercising the fuzzy product/label matching in
app/services/products.py — those need real names (Open Food Facts is the
obvious source).

Every row written is tagged in supplier_orders.raw_result under `_synthetic`,
so loaded data can always be told apart from real scans and removed with
--reset without touching anything you actually scanned.

Requires pandas + pyarrow (ml/requirements.txt or requirements-dev.txt); they
are deliberately not in the web image.
"""

import argparse
import hashlib
import sys
from datetime import datetime, timedelta, timezone

from app import create_app
from app.extensions import db
from app.models import OrderItem, Product, ProductForecast, SupplierOrder

SOURCE_ID = "freshretailnet-50k"
SOURCE_LICENSE = "CC-BY-4.0"
SOURCE_URL = "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K"

PRODUCT_PREFIX = "FRN"

SUPPLIER_COUNT = 6
SUPPLIER_NAMES = [
    "Aegean Fresh Produce",
    "Northgate Wholesale",
    "Meridian Food Supply",
    "Harbour Provisions",
    "Valley Cold Chain",
    "Continental Dry Goods",
]


def supplier_for(supplier_index: int) -> tuple[str, int]:
    """Returns (supplier_name, delivery_weekday).

    Each supplier delivers on its own weekday. That matters more than it
    looks: data_prep.py derives a `day_of_week` feature, and if every order
    landed on the same weekday that feature would be constant and carry no
    signal at all.
    """
    return SUPPLIER_NAMES[supplier_index], supplier_index % 7


def product_name_for(product_id: int, category_id: int) -> str:
    return f"{PRODUCT_PREFIX} product {product_id:05d} (cat {category_id})"


def stable_unit_price(product_id: int) -> float:
    """Deterministic pseudo-price. The dataset has no prices, and unit_price is
    not a forecasting feature — this exists so rows look like real order lines
    rather than carrying a suspicious column of NULLs.
    """
    digest = hashlib.sha256(f"{SOURCE_ID}:{product_id}".encode()).hexdigest()
    return round(0.80 + (int(digest[:8], 16) % 4200) / 100.0, 2)


def load_frame(parquet_path: str, stores: int, min_periods: int, period: str):
    import pandas as pd

    columns = ["store_id", "product_id", "third_category_id", "dt", "sale_amount"]
    return reshape(pd.read_parquet(parquet_path, columns=columns), stores, min_periods, period)


def reshape(df, stores: int, min_periods: int, period: str):
    """Daily per-product sales -> periodic per-product order quantities.

    Kept separate from the parquet read so the reshaping rules — which is
    where all the judgement lives — are unit-testable against a small frame.
    """
    import pandas as pd

    kept_stores = sorted(df["store_id"].unique())[:stores]
    df = df[df["store_id"].isin(kept_stores)].copy()

    df["dt"] = pd.to_datetime(df["dt"])

    freq = {"weekly": "W-MON", "fortnightly": "2W-MON", "monthly": "MS"}[period]

    span = timedelta(days=62)
    calendar = pd.DataFrame({"dt": pd.date_range(df["dt"].min() - span, df["dt"].max() + span, freq="D")})
    expected = calendar.groupby(pd.Grouper(key="dt", freq=freq))["dt"].nunique()

    observed = df.groupby(pd.Grouper(key="dt", freq=freq))["dt"].nunique()
    complete = {bucket for bucket, days in observed.items() if days == expected.get(bucket, -1)}
    dropped = [str(b.date()) for b in observed.index if b not in complete]

    grouped = (
        df.groupby(["store_id", "product_id", "third_category_id", pd.Grouper(key="dt", freq=freq)])["sale_amount"]
        .sum()
        .reset_index()
    )
    grouped = grouped[grouped["dt"].isin(complete)]
    if dropped:
        print(f"dropped {len(dropped)} incomplete period(s) at the range edges: {', '.join(dropped)}")

    grouped["supplier_index"] = grouped["third_category_id"] % SUPPLIER_COUNT

    grouped = grouped[grouped["sale_amount"] > 0]

    counts = grouped.groupby(["store_id", "product_id"])["sale_amount"].transform("size")
    grouped = grouped[counts >= min_periods]

    return grouped.sort_values(["store_id", "product_id", "dt"]), kept_stores


def purge_synthetic() -> tuple[int, int, int]:
    """Removes only what this loader created, identified by the `_synthetic`
    marker and the product-name prefix. Real scanned orders are untouched.
    """
    orders = SupplierOrder.query.filter(SupplierOrder.raw_result.has_key("_synthetic")).all()  # noqa: W601
    order_count = len(orders)
    for order in orders:
        db.session.delete(order)
    db.session.flush()

    candidates = Product.query.filter(Product.canonical_name.like(f"{PRODUCT_PREFIX} %")).all()
    removable = [p for p in candidates if not p.order_items and not p.labels]

    if removable:
        forecast_count = ProductForecast.query.filter(
            ProductForecast.product_id.in_([p.id for p in removable])
        ).delete(synchronize_session=False)
        db.session.flush()
    else:
        forecast_count = 0

    for product in removable:
        db.session.delete(product)

    db.session.commit()
    return order_count, len(removable), forecast_count


def run(args) -> int:
    frame, kept_stores = load_frame(args.parquet, args.stores, args.min_periods, args.period)

    if frame.empty:
        print("Nothing to load: no product met the filters.", file=sys.stderr)
        return 1

    n_products = frame.groupby(["store_id", "product_id"]).ngroups
    n_orders = frame.groupby(["store_id", "supplier_index", "dt"]).ngroups
    print(f"source      : {SOURCE_ID} ({SOURCE_LICENSE})")
    print(f"stores      : {[int(s) for s in kept_stores]}")
    print(f"period      : {args.period}")
    print(f"products    : {n_products}")
    print(f"order lines : {len(frame):,}")
    print(f"orders      : {n_orders:,}")
    print(f"date range  : {frame.dt.min().date()} -> {frame.dt.max().date()}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    if args.reset:
        removed_orders, removed_products, removed_forecasts = purge_synthetic()
        print(
            f"\nreset: removed {removed_orders} synthetic order(s), "
            f"{removed_products} product(s), {removed_forecasts} forecast(s)"
        )

    loaded_at = datetime.now(timezone.utc).isoformat()

    products: dict[tuple, Product] = {}
    for (store_id, product_id, category_id), _ in frame.groupby(["store_id", "product_id", "third_category_id"]):
        name = product_name_for(int(product_id), int(category_id))
        if args.stores > 1:
            name = f"{name} [store {int(store_id)}]"
        product = Product(canonical_name=name, category=f"cat-{int(category_id)}", unit_of_measure="kg")
        db.session.add(product)
        products[(store_id, product_id)] = product
    db.session.flush()
    print(f"\ninserted {len(products)} product(s)")

    orders_written = items_written = 0
    for (store_id, supplier_index, period_end), rows in frame.groupby(["store_id", "supplier_index", "dt"]):
        supplier_name, weekday = supplier_for(int(supplier_index))
        order_date = (period_end.date() + timedelta(days=weekday))
        order_number = f"FRN-{int(store_id)}-S{int(supplier_index)}-{period_end.date():%Y%m%d}"

        order = SupplierOrder(
            supplier_name=supplier_name,
            order_number=order_number,
            order_date=order_date,
            blob_name=f"synthetic/{SOURCE_ID}/{order_number}",
            blob_url=f"synthetic://{SOURCE_ID}/{order_number}",
            raw_result={
                "_synthetic": {
                    "source": SOURCE_ID,
                    "license": SOURCE_LICENSE,
                    "url": SOURCE_URL,
                    "loaded_at": loaded_at,
                    "period": args.period,
                    "store_id": int(store_id),
                    "supplier_index": int(supplier_index),
                }
            },
        )
        db.session.add(order)
        db.session.flush()
        orders_written += 1

        for row in rows.itertuples():
            product = products[(row.store_id, row.product_id)]
            db.session.add(
                OrderItem(
                    order_id=order.id,
                    product_id=product.id,
                    product_name=product.canonical_name,
                    product_code=f"FRN{int(row.product_id):05d}",
                    quantity=round(float(row.sale_amount), 3),
                    unit_price=stable_unit_price(int(row.product_id)),
                )
            )
            items_written += 1

        if orders_written % 200 == 0:
            db.session.commit()
            print(f"  ... {orders_written:,} orders, {items_written:,} items")

    db.session.commit()
    print(f"\nwrote {orders_written:,} supplier order(s) and {items_written:,} order item(s)")
    print("\nNext: python -m scripts.run_forecasts")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", required=True, help="Path to FreshRetailNet-50K train.parquet")
    parser.add_argument("--stores", type=int, default=1, help="How many stores to load (default 1)")
    parser.add_argument(
        "--period",
        choices=["weekly", "fortnightly", "monthly"],
        default="weekly",
        help="How often orders are placed (default weekly)",
    )
    parser.add_argument(
        "--min-periods",
        type=int,
        default=6,
        help="Skip products with fewer than this many order periods (default 6)",
    )
    parser.add_argument("--reset", action="store_true", help="Remove previously loaded synthetic data first")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be loaded and stop")
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
