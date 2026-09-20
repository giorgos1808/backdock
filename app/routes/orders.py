from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from flask import Blueprint, Response, abort, current_app, jsonify, render_template, request

from app.extensions import db
from app.metrics import orders_processed
from app.models import SupplierOrder
from app.services.orders import create_order_from_upload
from app.services.reconciliation import order_status, reconciliation_summary

orders_bp = Blueprint("orders", __name__, url_prefix="/orders")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _wants_json() -> bool:
    return request.accept_mimetypes.best == "application/json" or request.is_json


@orders_bp.route("/scan", methods=["GET", "POST"])
def scan_order():
    if request.method == "GET":
        return render_template("order_upload.html")

    if "document" not in request.files:
        return jsonify({"error": "No file part named 'document' in the request"}), 400

    file = request.files["document"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}), 400

    try:
        order = create_order_from_upload(file)
    except HttpResponseError as exc:
        orders_processed.add(1, {"outcome": "document_intelligence_error"})
        return jsonify({"error": "Azure Document Intelligence request failed", "details": str(exc)}), 502

    response = order.to_dict()

    if _wants_json():
        return jsonify(response)
    return render_template("order_result.html", order=response)


@orders_bp.route("", methods=["GET"])
def list_orders():
    page = request.args.get("page", default=1, type=int)
    per_page = min(request.args.get("per_page", default=20, type=int), 100)
    pagination = SupplierOrder.query.order_by(SupplierOrder.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    if _wants_json():
        return jsonify(
            {
                "items": [order.to_dict(include_items=False) for order in pagination.items],
                "page": pagination.page,
                "per_page": pagination.per_page,
                "total": pagination.total,
            }
        )

    rows = []
    for order in pagination.items:
        summary = reconciliation_summary(order)
        rows.append({"order": order, "reconciliation": summary, "status": order_status(order, summary)})

    return render_template("orders_list.html", rows=rows, pagination=pagination)


@orders_bp.route("/<uuid:order_id>", methods=["GET"])
def get_order(order_id):
    order = db.session.get(SupplierOrder, order_id)
    if order is None:
        if _wants_json():
            return jsonify({"error": "Order not found"}), 404
        abort(404)

    summary = reconciliation_summary(order)

    if _wants_json():
        response = order.to_dict()
        response["labels"] = [label.to_dict() for label in order.labels]
        response["reconciliation"] = summary
        return jsonify(response)

    return render_template("order_detail.html", order=order, reconciliation=summary, status=order_status(order, summary))


@orders_bp.route("/<uuid:order_id>/document", methods=["GET"])
def get_order_document(order_id):
    order = db.session.get(SupplierOrder, order_id)
    if order is None:
        return jsonify({"error": "Order not found"}), 404

    try:
        data, content_type = current_app.order_blob_storage.download(order.blob_name)
    except ResourceNotFoundError:
        return jsonify({"error": "Document not found in storage"}), 404

    return Response(data, mimetype=content_type)
