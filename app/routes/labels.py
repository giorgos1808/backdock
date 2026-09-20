from datetime import datetime

from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from flask import Blueprint, Response, current_app, jsonify, render_template, request

from app.extensions import db
from app.metrics import labels_processed
from app.models import ProductLabel, SupplierOrder
from app.services.labels import create_label_from_upload
from app.services.products import resolve_product
from app.services.reconciliation import match_label, reconciliation_summary

labels_bp = Blueprint("labels", __name__, url_prefix="/orders/<uuid:order_id>/labels")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _parse_date(value):
    if not value:
        return None
    return datetime.fromisoformat(value).date()


@labels_bp.route("/scan", methods=["GET", "POST"])
def scan_label(order_id):
    order = db.session.get(SupplierOrder, order_id)
    if order is None:
        return jsonify({"error": "Order not found"}), 404

    if request.method == "GET":
        return render_template("label_upload.html", order=order.to_dict(include_items=False))

    if "image" not in request.files:
        return jsonify({"error": "No file part named 'image' in the request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}), 400

    try:
        label = create_label_from_upload(order, file)
    except HttpResponseError as exc:
        labels_processed.add(1, {"outcome": "document_intelligence_error"})
        return jsonify({"error": "Azure Document Intelligence request failed", "details": str(exc)}), 502

    response = label.to_dict()

    if request.accept_mimetypes.best == "application/json" or request.is_json:
        return jsonify(response)
    return render_template("label_result.html", label=response)


@labels_bp.route("", methods=["GET"])
def list_labels(order_id):
    order = db.session.get(SupplierOrder, order_id)
    if order is None:
        return jsonify({"error": "Order not found"}), 404

    return jsonify(
        {
            "items": [label.to_dict() for label in order.labels],
            "reconciliation": reconciliation_summary(order),
        }
    )


@labels_bp.route("/<uuid:label_id>/image", methods=["GET"])
def get_label_image(order_id, label_id):
    label = db.session.get(ProductLabel, label_id)
    if label is None or label.order_id != order_id:
        return jsonify({"error": "Label not found"}), 404

    try:
        data, content_type = current_app.label_blob_storage.download(label.blob_name)
    except ResourceNotFoundError:
        return jsonify({"error": "Image not found in storage"}), 404

    return Response(data, mimetype=content_type)


@labels_bp.route("/<uuid:label_id>", methods=["PATCH"])
def update_label(order_id, label_id):
    label = db.session.get(ProductLabel, label_id)
    if label is None or label.order_id != order_id:
        return jsonify({"error": "Label not found"}), 404

    payload = request.get_json(silent=True) or {}

    if "product_name" in payload:
        label.product_name = payload["product_name"]
        label.product_id = resolve_product(payload["product_name"]).id
    if "expiration_date" in payload:
        label.expiration_date = _parse_date(payload["expiration_date"])
    if "lot_number" in payload:
        label.lot_number = payload["lot_number"]

    match_label(label)
    db.session.commit()

    return jsonify(label.to_dict())
