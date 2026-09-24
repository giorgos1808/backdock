from app import create_app
from app.extensions import db
from app.models import OrderItem, ProductLabel
from app.services.products import resolve_product


def run() -> None:
    updated = 0

    for item in OrderItem.query.filter_by(product_id=None).all():
        item.product_id = resolve_product(item.product_name).id
        updated += 1

    for label in ProductLabel.query.filter_by(product_id=None).all():
        label.product_id = resolve_product(label.product_name).id
        updated += 1

    db.session.commit()
    print(f"Backfilled product_id for {updated} row(s).")


if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        run()
