from flask import current_app

from app.extensions import db
from app.metrics import labels_processed
from app.models import ProductLabel
from app.services.document_intelligence import extract_label_fields
from app.services.products import resolve_product
from app.services.reconciliation import match_label


def create_label_from_upload(order, file) -> ProductLabel:
    """Runs Document Intelligence on an uploaded product-label photo,
    persists a ProductLabel against `order`, matches it, and returns it.
    Committed before returning.

    Shared by v1's labels_bp.scan_label and v2's camera-upload route — see
    app/services/orders.py for why this was extracted rather than
    duplicated.

    Raises azure.core.exceptions.HttpResponseError if the Document
    Intelligence call fails; callers translate that into whatever HTTP
    response fits their route.
    """
    image_bytes = file.read()
    raw_result = current_app.document_intelligence.analyze_label(image_bytes)

    blob_name, blob_url = current_app.label_blob_storage.upload(image_bytes, file.filename, file.mimetype)
    fields = extract_label_fields(raw_result)
    product = resolve_product(fields["product_name"])

    label = ProductLabel(
        order_id=order.id,
        product_id=product.id,
        blob_name=blob_name,
        blob_url=blob_url,
        raw_result=raw_result,
        product_name=fields["product_name"],
        expiration_date=fields["expiration_date"],
        lot_number=fields["lot_number"],
    )
    order.labels.append(label)
    match_label(label)

    db.session.add(label)
    db.session.commit()
    labels_processed.add(1, {"outcome": "success"})
    return label
