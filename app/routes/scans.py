from datetime import datetime, timezone

from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from flask import Blueprint, Response, current_app, jsonify, render_template, request
from sqlalchemy import text

from app.extensions import db
from app.metrics import scans_processed
from app.models import SEARCH_VECTOR_SQL, Scan, ScanReview
from app.services.vision import extract_summary

scans_bp = Blueprint("scans", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@scans_bp.route("/scan", methods=["POST"])
def scan():
    if "image" not in request.files:
        return jsonify({"error": "No file part named 'image' in the request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}), 400

    image_bytes = file.read()

    try:
        raw_result = current_app.vision_analyzer.analyze(image_bytes)
    except HttpResponseError as exc:
        scans_processed.add(1, {"outcome": "vision_error"})
        return jsonify({"error": "Azure Image Analysis request failed", "details": str(exc)}), 502

    blob_name, blob_url = current_app.blob_storage.upload(image_bytes, file.filename, file.mimetype)
    summary = extract_summary(raw_result)

    scan_record = Scan(source_filename=file.filename, content_type=file.mimetype, size_bytes=len(image_bytes), 
                       blob_name=blob_name, blob_url=blob_url, raw_result=raw_result, **summary)

    db.session.add(scan_record)
    db.session.commit()
    scans_processed.add(1, {"outcome": "success"})

    response = {"id": str(scan_record.id), "lines": (scan_record.ocr_text or "").splitlines()}

    if request.accept_mimetypes.best == "application/json" or request.is_json:
        return jsonify(response)

    return render_template("result.html", response=response)


@scans_bp.route("/scans/<uuid:scan_id>", methods=["GET"])
def get_scan(scan_id):
    scan_record = db.session.get(Scan, scan_id)
    if scan_record is None:
        return jsonify({"error": "Scan not found"}), 404

    return jsonify({"id": str(scan_record.id), "lines": (scan_record.ocr_text or "").splitlines()})


@scans_bp.route("/scans/<uuid:scan_id>/review", methods=["POST"])
def review_scan(scan_id):
    scan_record = db.session.get(Scan, scan_id)
    if scan_record is None:
        return jsonify({"error": "Scan not found"}), 404

    if request.is_json:
        payload = request.get_json(silent=True) or {}
        lines = payload.get("lines")
        if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
            return jsonify({"error": "'lines' must be a list of strings"}), 400
    else:
        lines = request.form.get("lines", "").splitlines()

    review = ScanReview.query.filter_by(scan_id=scan_id).first()
    if review is None:
        review = ScanReview(scan_id=scan_id, lines=lines)
        db.session.add(review)
    else:
        review.lines = lines
        review.reviewed_at = datetime.now(timezone.utc)

    db.session.commit()

    if request.accept_mimetypes.best == "application/json" or request.is_json:
        return jsonify(review.to_dict())

    return render_template("review_result.html", scan_id=str(scan_id), review=review.to_dict())


@scans_bp.route("/scans/<uuid:scan_id>/review", methods=["GET"])
def get_scan_review(scan_id):
    review = ScanReview.query.filter_by(scan_id=scan_id).first()
    if review is None:
        return jsonify({"error": "No review found for this scan"}), 404

    return jsonify(review.to_dict())


@scans_bp.route("/scans/<uuid:scan_id>/image", methods=["GET"])
def get_scan_image(scan_id):
    scan_record = db.session.get(Scan, scan_id)
    if scan_record is None:
        return jsonify({"error": "Scan not found"}), 404

    try:
        data, content_type = current_app.blob_storage.download(scan_record.blob_name)
    except ResourceNotFoundError:
        return jsonify({"error": "Image not found in storage"}), 404

    return Response(data, mimetype=content_type)


@scans_bp.route("/scans/labels", methods=["GET"])
def scan_labels():
    scan_records = Scan.query.order_by(Scan.created_at.desc()).limit(60).all()

    scans = []
    for scan_record in scan_records:
        data = scan_record.to_dict()
        data["tags"] = sorted(data["tags"] or [], key=lambda tag: tag["confidence"], reverse=True)
        scans.append(data)

    return render_template("image_labels.html", scans=scans)


@scans_bp.route("/scans/<uuid:scan_id>/view", methods=["GET"])
def view_scan(scan_id):
    scan_record = db.session.get(Scan, scan_id)
    if scan_record is None:
        return jsonify({"error": "Scan not found"}), 404

    data = scan_record.to_dict()
    data["tags"] = sorted(data["tags"] or [], key=lambda tag: tag["confidence"], reverse=True)
    lines = (scan_record.ocr_text or "").splitlines()
    review = scan_record.review.to_dict() if scan_record.review else None

    return render_template("scan_detail.html", scan=data, lines=lines, review=review)


@scans_bp.route("/scans", methods=["GET"])
def list_scans():
    query = Scan.query.order_by(Scan.created_at.desc())

    search = request.args.get("q")
    if search:
        query = query.filter(text(f"{SEARCH_VECTOR_SQL} @@ plainto_tsquery('english', :q)")).params(q=search)

    page = request.args.get("page", default=1, type=int)
    per_page = min(request.args.get("per_page", default=20, type=int), 100)
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    return jsonify(
        {
            "items": [s.to_dict() for s in pagination.items],
            "page": pagination.page,
            "per_page": pagination.per_page,
            "total": pagination.total,
        }
    )
