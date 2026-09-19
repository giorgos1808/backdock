from datetime import datetime

from flask import current_app

from app.extensions import db
from app.metrics import orders_processed
from app.models import OrderItem, SupplierOrder
from app.services.document_intelligence import extract_order_summary
from app.services.products import resolve_product


def _parse_order_date(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def create_order_from_upload(file) -> SupplierOrder:
    """Runs Document Intelligence on an uploaded supplier-order document and
    persists a SupplierOrder + its OrderItems. Committed before returning.

    Shared by v1's orders_bp.scan_order and v2's camera-upload route so both
    UIs go through identical extraction/persistence — extracted from the
    former (see git history) rather than duplicated for v2.

    Raises azure.core.exceptions.HttpResponseError if the Document
    Intelligence call fails; callers translate that into whatever HTTP
    response fits their route.
    """
    document_bytes = file.read()
    raw_result = current_app.document_intelligence.analyze_invoice(document_bytes)

    blob_name, blob_url = current_app.order_blob_storage.upload(document_bytes, file.filename, file.mimetype)
    summary = extract_order_summary(raw_result)

    order = SupplierOrder(
        supplier_name=summary["supplier_name"],
        order_number=summary["order_number"],
        order_date=_parse_order_date(summary["order_date"]),
        blob_name=blob_name,
        blob_url=blob_url,
        raw_result=raw_result,
    )
    for item in summary["items"]:
        product = resolve_product(item["product_name"])
        order.items.append(
            OrderItem(
                product_id=product.id,
                product_name=item["product_name"] or "Unknown product",
                product_code=item["product_code"],
                quantity=item["quantity"],
                unit_price=item["unit_price"],
                raw_line=item["raw_line"],
            )
        )

    db.session.add(order)
    db.session.commit()
    # See the docstring on the v1 route this was extracted from for why
    # has_items is tracked separately from the plain success/failure outcome.
    orders_processed.add(1, {"outcome": "success", "has_items": bool(summary["items"])})
    return order
