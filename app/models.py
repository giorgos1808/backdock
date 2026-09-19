import uuid
from datetime import datetime, timezone

from sqlalchemy.dialects.postgresql import JSONB

from app.extensions import db

SEARCH_VECTOR_SQL = "to_tsvector('english', coalesce(caption, '') || ' ' || coalesce(ocr_text, ''))"


class Scan(db.Model):
    __tablename__ = "scans"

    id = db.Column(db.Uuid, primary_key=True, default=uuid.uuid4)

    source_filename = db.Column(db.String(255), nullable=False)
    content_type = db.Column(db.String(100), nullable=False)
    size_bytes = db.Column(db.Integer, nullable=False)

    blob_name = db.Column(db.String(512), nullable=False)
    blob_url = db.Column(db.String(1024), nullable=False)

    caption = db.Column(db.Text)
    caption_confidence = db.Column(db.Float)
    tags = db.Column(JSONB)
    ocr_text = db.Column(db.Text)

    raw_result = db.Column(JSONB, nullable=False)

    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "source_filename": self.source_filename,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "blob_url": self.blob_url,
            "caption": self.caption,
            "caption_confidence": self.caption_confidence,
            "tags": self.tags,
            "ocr_text": self.ocr_text,
            "created_at": self.created_at.isoformat(),
        }


class ScanReview(db.Model):
    """A human-checked/corrected version of a Scan's OCR text.

    One review per scan (scan_id is unique) — saving a review again just
    overwrites the previous one rather than keeping a history, since nothing
    so far needs to see prior revisions.
    """

    __tablename__ = "scan_reviews"

    id = db.Column(db.Uuid, primary_key=True, default=uuid.uuid4)
    scan_id = db.Column(db.Uuid, db.ForeignKey("scans.id"), nullable=False, unique=True)

    lines = db.Column(JSONB, nullable=False)

    reviewed_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    scan = db.relationship("Scan", backref=db.backref("review", uselist=False))

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "scan_id": str(self.scan_id),
            "lines": self.lines,
            "reviewed_at": self.reviewed_at.isoformat(),
        }


class Product(db.Model):
    """Canonical product catalog entry.

    OrderItem/ProductLabel each carry their own free-text product_name (as
    extracted from a specific document/label) plus a product_id here once
    resolved — see app/services/products.py. Without this, "Tomato Sauce
    400g" and "TOMATO SAUCE 400g" from two different scans look unrelated,
    which makes per-product history (needed for quantity forecasting)
    impossible to assemble.
    """

    __tablename__ = "products"

    id = db.Column(db.Uuid, primary_key=True, default=uuid.uuid4)

    canonical_name = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(255))
    unit_of_measure = db.Column(db.String(50))

    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    order_items = db.relationship("OrderItem", backref="product")
    labels = db.relationship("ProductLabel", backref="product")

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "canonical_name": self.canonical_name,
            "category": self.category,
            "unit_of_measure": self.unit_of_measure,
            "created_at": self.created_at.isoformat(),
        }


class SupplierOrder(db.Model):
    __tablename__ = "supplier_orders"

    id = db.Column(db.Uuid, primary_key=True, default=uuid.uuid4)

    supplier_name = db.Column(db.String(255))
    order_number = db.Column(db.String(100))
    order_date = db.Column(db.Date)

    blob_name = db.Column(db.String(512), nullable=False)
    blob_url = db.Column(db.String(1024), nullable=False)
    raw_result = db.Column(JSONB, nullable=False)

    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    items = db.relationship("OrderItem", backref="order", cascade="all, delete-orphan", order_by="OrderItem.id")
    labels = db.relationship("ProductLabel", backref="order", order_by="ProductLabel.created_at")

    def to_dict(self, include_items: bool = True) -> dict:
        data = {
            "id": str(self.id),
            "supplier_name": self.supplier_name,
            "order_number": self.order_number,
            "order_date": self.order_date.isoformat() if self.order_date else None,
            "blob_url": self.blob_url,
            "created_at": self.created_at.isoformat(),
        }
        if include_items:
            data["items"] = [item.to_dict() for item in self.items]
        return data


class OrderItem(db.Model):
    __tablename__ = "order_items"

    id = db.Column(db.Uuid, primary_key=True, default=uuid.uuid4)
    order_id = db.Column(db.Uuid, db.ForeignKey("supplier_orders.id"), nullable=False)
    product_id = db.Column(db.Uuid, db.ForeignKey("products.id"), nullable=True)

    product_name = db.Column(db.Text, nullable=False)
    product_code = db.Column(db.String(100))
    quantity = db.Column(db.Numeric(12, 3))
    unit_price = db.Column(db.Numeric(12, 2))

    raw_line = db.Column(JSONB)

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "product_id": str(self.product_id) if self.product_id else None,
            "product_name": self.product_name,
            "product_code": self.product_code,
            "quantity": float(self.quantity) if self.quantity is not None else None,
            "unit_price": float(self.unit_price) if self.unit_price is not None else None,
        }


class ProductLabel(db.Model):
    __tablename__ = "product_labels"

    id = db.Column(db.Uuid, primary_key=True, default=uuid.uuid4)
    order_id = db.Column(db.Uuid, db.ForeignKey("supplier_orders.id"), nullable=True)
    matched_order_item_id = db.Column(db.Uuid, db.ForeignKey("order_items.id"), nullable=True)
    product_id = db.Column(db.Uuid, db.ForeignKey("products.id"), nullable=True)

    blob_name = db.Column(db.String(512), nullable=False)
    blob_url = db.Column(db.String(1024), nullable=False)
    raw_result = db.Column(JSONB, nullable=False)

    product_name = db.Column(db.Text)
    expiration_date = db.Column(db.Date)
    lot_number = db.Column(db.String(100))

    match_status = db.Column(db.String(30), nullable=False, default="unmatched")
    match_score = db.Column(db.Float)

    received_quantity = db.Column(db.Numeric(12, 3))

    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    matched_item = db.relationship("OrderItem", foreign_keys=[matched_order_item_id])

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "order_id": str(self.order_id) if self.order_id else None,
            "product_id": str(self.product_id) if self.product_id else None,
            "blob_url": self.blob_url,
            "product_name": self.product_name,
            "expiration_date": self.expiration_date.isoformat() if self.expiration_date else None,
            "lot_number": self.lot_number,
            "match_status": self.match_status,
            "match_score": self.match_score,
            "received_quantity": float(self.received_quantity) if self.received_quantity is not None else None,
            "matched_order_item_id": str(self.matched_order_item_id) if self.matched_order_item_id else None,
            "created_at": self.created_at.isoformat(),
        }


class ProductForecast(db.Model):
    """A predicted quantity for a product, generated at a point in time.

    Deliberately append-only (a new row per prediction, never updated) rather
    than one row per product — keeping the history is what lets forecast
    accuracy be checked later against what was actually ordered afterwards.
    `method` distinguishes the Phase 1 rolling-average baseline ("baseline")
    from a future registered ML model (its version string).
    """

    __tablename__ = "product_forecasts"

    id = db.Column(db.Uuid, primary_key=True, default=uuid.uuid4)
    product_id = db.Column(db.Uuid, db.ForeignKey("products.id"), nullable=False)

    predicted_quantity = db.Column(db.Numeric(12, 3), nullable=False)
    method = db.Column(db.String(50), nullable=False)

    predicted_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    product = db.relationship("Product", backref="forecasts")

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "product_id": str(self.product_id),
            "predicted_quantity": float(self.predicted_quantity),
            "method": self.method,
            "predicted_at": self.predicted_at.isoformat(),
        }
