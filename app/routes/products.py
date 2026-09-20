import uuid

from flask import Blueprint, jsonify, redirect, render_template, request, url_for

from app.extensions import db
from app.models import Product, ProductForecast
from app.services.forecasting import generate_forecast

products_bp = Blueprint("products", __name__, url_prefix="/products")


def _wants_json() -> bool:
    return request.accept_mimetypes.best == "application/json" or request.is_json


def _line_chart(history, width=560, height=200, pad_x=28, pad_y=20):
    """Lay out a product's forecast history as an inline SVG polyline.

    No charting library — a handful of (x, y) points is simple enough to
    place by hand, and it keeps this dependency-free like the rest of the
    templates.
    """
    if not history:
        return None

    values = [float(f.predicted_quantity) for f in history]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0

    count = len(values)
    step = (width - 2 * pad_x) / (count - 1) if count > 1 else 0

    points = []
    for i, forecast in enumerate(history):
        value = float(forecast.predicted_quantity)
        x = pad_x + i * step
        y = height - pad_y - ((value - lo) / span) * (height - 2 * pad_y)
        points.append(
            {
                "x": round(x, 1),
                "y": round(y, 1),
                "value": value,
                "label": forecast.predicted_at.strftime("%Y-%m-%d"),
            }
        )

    return {
        "width": width,
        "height": height,
        "points": points,
        "polyline": " ".join(f"{p['x']},{p['y']}" for p in points),
        "min_value": lo,
        "max_value": hi,
    }


@products_bp.route("", methods=["GET"])
def list_products():
    page = request.args.get("page", default=1, type=int)
    per_page = min(request.args.get("per_page", default=20, type=int), 100)
    pagination = Product.query.order_by(Product.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify(
        {
            "items": [product.to_dict() for product in pagination.items],
            "page": pagination.page,
            "per_page": pagination.per_page,
            "total": pagination.total,
        }
    )


@products_bp.route("/forecasts", methods=["GET"])
def forecasts_page():
    return render_template("forecasts_landing.html")


@products_bp.route("/forecasts/table", methods=["GET"])
def forecasts_table():
    products = Product.query.order_by(Product.canonical_name.asc()).all()

    rows = []
    for product in products:
        latest =  ProductForecast.query.filter_by(product_id=product.id).order_by(ProductForecast.predicted_at.desc()).first()

        rows.append(
            {
                "product": product.to_dict(),
                "forecast": latest.to_dict() if latest else None,
                "has_history": any(item.quantity is not None for item in product.order_items),
            }
        )

    return render_template("forecasts.html", rows=rows)


@products_bp.route("/forecasts/dashboard", methods=["GET"])
def forecasts_dashboard():
    products = Product.query.order_by(Product.canonical_name.asc()).all()

    default_id = None
    for product in products:
        if ProductForecast.query.filter_by(product_id=product.id).first():
            default_id = product.id
            break

    selected_id = request.args.get("product_id", type=uuid.UUID) or default_id

    selected_product = None
    history = []
    if selected_id:
        selected_product = db.session.get(Product, selected_id)
        history = ProductForecast.query.filter_by(product_id=selected_id).order_by(ProductForecast.predicted_at.asc()).all()

    return render_template(
        "forecasts_dashboard.html",
        products=[p.to_dict() for p in products],
        selected_id=str(selected_id) if selected_id else None,
        selected_product=selected_product.to_dict() if selected_product else None,
        chart=_line_chart(history),
    )


@products_bp.route("/<uuid:product_id>", methods=["GET"])
def get_product(product_id):
    product = db.session.get(Product, product_id)
    if product is None:
        return jsonify({"error": "Product not found"}), 404

    response = product.to_dict()
    response["order_items"] = [item.to_dict() for item in product.order_items]
    response["labels"] = [label.to_dict() for label in product.labels]
    return jsonify(response)


@products_bp.route("/<uuid:product_id>/forecast", methods=["POST"])
def create_forecast(product_id):
    product = db.session.get(Product, product_id)
    if product is None:
        return jsonify({"error": "Product not found"}), 404

    forecast = generate_forecast(product_id)
    if forecast is None:
        if _wants_json():
            return jsonify({"error": "No order history yet for this product — nothing to forecast from"}), 422
        return redirect(url_for("products.forecasts_table"))

    if _wants_json():
        return jsonify(forecast.to_dict())
    return redirect(url_for("products.forecasts_table"))


@products_bp.route("/<uuid:product_id>/forecast", methods=["GET"])
def get_latest_forecast(product_id):
    product = db.session.get(Product, product_id)
    if product is None:
        return jsonify({"error": "Product not found"}), 404

    forecast = ProductForecast.query.filter_by(product_id=product_id).order_by(ProductForecast.predicted_at.desc()).first()

    if forecast is None:
        return jsonify({"error": "No forecast yet for this product"}), 404

    return jsonify(forecast.to_dict())
