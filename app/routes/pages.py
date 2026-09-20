from itertools import chain

from flask import Blueprint, render_template

from app.models import ProductLabel, Scan, SupplierOrder

pages_bp = Blueprint("pages", __name__)

GALLERY_LIMIT = 60


@pages_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@pages_bp.route("/gallery", methods=["GET"])
def gallery():
    scans = Scan.query.order_by(Scan.created_at.desc()).limit(GALLERY_LIMIT).all()
    orders = SupplierOrder.query.order_by(SupplierOrder.created_at.desc()).limit(GALLERY_LIMIT).all()
    labels = (
        ProductLabel.query.join(SupplierOrder)
        .order_by(ProductLabel.created_at.desc())
        .limit(GALLERY_LIMIT)
        .all()
    )

    items = chain(
        (
            {
                "kind": "scan",
                "title": scan.source_filename,
                "subtitle": scan.caption or "Image scan",
                "created_at": scan.created_at,
                "is_image": scan.content_type.startswith("image/"),
                "url": f"/scans/{scan.id}/image",
            }
            for scan in scans
        ),
        (
            {
                "kind": "order",
                "title": order.order_number or "Supplier order",
                "subtitle": order.supplier_name or "Unknown supplier",
                "created_at": order.created_at,
                "is_image": not order.blob_name.lower().endswith(".pdf"),
                "url": f"/orders/{order.id}/document",
            }
            for order in orders
        ),
        (
            {
                "kind": "label",
                "title": label.product_name or "Product label",
                "subtitle": f"Order {label.order.order_number or label.order.id}",
                "created_at": label.created_at,
                "is_image": True,
                "url": f"/orders/{label.order_id}/labels/{label.id}/image",
            }
            for label in labels
        ),
    )

    uploads = sorted(items, key=lambda item: item["created_at"], reverse=True)[:GALLERY_LIMIT]

    return render_template("gallery.html", uploads=uploads)
