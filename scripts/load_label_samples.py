"""Load scanned product labels into the app so real extraction output can be
browsed in the UI.

Built for ExpDate (CC BY 4.0, 1,767 real packaging photos from KIST):

    https://felizang.github.io/expdate/

Two modes:

    # free — ingest Document Intelligence responses already saved to disk
    python -m scripts.load_label_samples --responses sample_data/expdate_responses

    # costs money — analyse images through your Document Intelligence resource
    python -m scripts.load_label_samples --images path/to/images --limit 25

WHAT YOU WILL SEE, AND WHAT YOU WON'T
-------------------------------------
A ProductLabel only exists relative to an order — labels_bp is mounted at
/orders/<order_id>/labels and there is no standalone label page — so these are
attached to one dedicated order created for the purpose.

That order deliberately has **no line items**, which makes every label come out
as `no_order` rather than `unmatched`. Both are honest; `no_order` is the more
accurate of the two, because these products genuinely were not ordered from
anyone. Giving the order invented line items to force `matched` statuses would
be fabricating a reconciliation that never happened.

So this demonstrates extraction — expiry dates, lot numbers, product names off
real packaging — and not reconciliation. Seeing reconciliation work needs a
real supplier order and labels from those same goods, which no public dataset
can supply.

Labels are not linked to catalogue Products either. The real scan route calls
resolve_product(), but the product names these photos yield are often OCR
fragments ('RY', 'SIcc'), and creating catalogue entries from them would
clutter /products with junk. See --link-products to mirror the real flow.

Every row is tagged in supplier_orders.raw_result under `_synthetic` and
removed by --reset, so it never gets confused with a real scan. Blob storage is
not involved, so the label image endpoint will 404 for these — the source
images stay wherever you downloaded them.
"""

import argparse
import json
import pathlib
import sys
from datetime import date, datetime, timezone

from app import create_app
from app.extensions import db
from app.models import ProductLabel, SupplierOrder
from app.services.document_intelligence import extract_label_fields
from app.services.products import resolve_product
from app.services.reconciliation import match_label

SOURCE_ID = "expdate"
SOURCE_LICENSE = "CC BY 4.0"
SOURCE_URL = "https://felizang.github.io/expdate/"
ORDER_NUMBER = "EXPDATE-SAMPLES"
SUPPLIER_NAME = "ExpDate sample labels"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def as_date(value):
    """extract_label_fields returns a date; a response loaded from JSON may
    carry it as an ISO string.
    """
    if value is None or isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def purge(session) -> tuple[int, int]:
    """Removes only this script's own order and its labels."""
    orders = SupplierOrder.query.filter(
        SupplierOrder.order_number == ORDER_NUMBER,
        SupplierOrder.raw_result.has_key("_synthetic"),  # noqa: W601
    ).all()
    labels = sum(len(o.labels) for o in orders)
    for order in orders:
        for label in list(order.labels):
            session.delete(label)
        session.delete(order)
    session.commit()
    return len(orders), labels


def collect(args, app):
    """Yields (source_name, raw_result) pairs."""
    if args.responses:
        root = pathlib.Path(args.responses)
        files = sorted(root.glob("*.json"))[: args.limit]
        if not files:
            print(f"No .json responses found in {root}", file=sys.stderr)
        for path in files:
            yield path.stem, json.loads(path.read_text(encoding="utf-8"))
        return

    root = pathlib.Path(args.images)
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)[: args.limit]
    if not images:
        print(f"No images found under {root}", file=sys.stderr)
    for path in images:
        print(f"  analysing {path.name} ...", flush=True)
        yield path.name, app.document_intelligence.analyze_label(path.read_bytes())


def run(args, app) -> int:
    if args.reset:
        orders, labels = purge(db.session)
        print(f"reset: removed {orders} sample order(s) and {labels} label(s)\n")

    pairs = list(collect(args, app))
    if not pairs:
        return 1

    if args.dry_run:
        print(f"--dry-run: {len(pairs)} label(s) would be loaded\n")
        for name, raw in pairs[:10]:
            f = extract_label_fields(raw)
            print(f"  {name:<18} exp={str(as_date(f['expiration_date'])):<12} lot={str(f['lot_number']):<10} {(f['product_name'] or '')[:36]!r}")
        return 0

    existing = SupplierOrder.query.filter_by(order_number=ORDER_NUMBER).first()
    if existing:
        print(f"A sample order already exists ({existing.id}). Re-run with --reset to replace it.", file=sys.stderr)
        return 1

    order = SupplierOrder(
        supplier_name=SUPPLIER_NAME,
        order_number=ORDER_NUMBER,
        order_date=date.today(),
        blob_name=f"synthetic/{SOURCE_ID}/{ORDER_NUMBER}",
        blob_url=f"synthetic://{SOURCE_ID}/{ORDER_NUMBER}",
        raw_result={
            "_synthetic": {
                "source": SOURCE_ID,
                "license": SOURCE_LICENSE,
                "url": SOURCE_URL,
                "loaded_at": datetime.now(timezone.utc).isoformat(),
                "note": (
                    "Sample labels for browsing extraction output. This order has no line "
                    "items on purpose, so every label reconciles as 'no_order' — these "
                    "products were never actually ordered."
                ),
            }
        },
    )
    db.session.add(order)
    db.session.flush()

    with_expiry = with_lot = 0
    for name, raw in pairs:
        fields = extract_label_fields(raw)
        expiry = as_date(fields["expiration_date"])

        product = resolve_product(fields["product_name"]) if args.link_products and fields["product_name"] else None

        label = ProductLabel(
            order_id=order.id,
            product_id=product.id if product else None,
            blob_name=f"synthetic/{SOURCE_ID}/{name}",
            blob_url=f"synthetic://{SOURCE_ID}/{name}",
            raw_result=raw,
            product_name=fields["product_name"],
            expiration_date=expiry,
            lot_number=fields["lot_number"],
        )
        order.labels.append(label)
        match_label(label)
        db.session.add(label)

        with_expiry += bool(expiry)
        with_lot += bool(fields["lot_number"])

    db.session.commit()

    n = len(pairs)
    print(f"loaded {n} label(s) onto order {ORDER_NUMBER} ({order.id})")
    print(f"  expiry extracted : {with_expiry}/{n}")
    print(f"  lot extracted    : {with_lot}/{n}")
    print(f"\nBrowse: /orders/{order.id}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--responses", help="Directory of saved Document Intelligence responses (.json). Free.")
    source.add_argument("--images", help="Directory of label images to analyse. Calls Azure; costs money.")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument(
        "--link-products",
        action="store_true",
        help="Link each label to a catalogue Product, as the real scan route does. Off by default: "
        "OCR'd label names are often fragments and would clutter /products.",
    )
    parser.add_argument("--reset", action="store_true", help="Remove a previously loaded sample order first")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be loaded and stop")
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        return run(args, app)


if __name__ == "__main__":
    raise SystemExit(main())
