from datetime import date

from flask import current_app

from app.extensions import db
from app.metrics import forecasts_generated
from app.models import OrderItem, ProductForecast, SupplierOrder

DEFAULT_WINDOW = 3


def product_history(product_id):
    """Chronological (order_date, quantity, supplier_name) history for a
    product, oldest first. Orders with a quantity of None are excluded —
    nothing to learn from an unparsed line. Orders with no order_date sort
    first (treated as "oldest known"), since Document Intelligence doesn't
    always find one — simple default, not date inference.
    """
    return (
        db.session.query(SupplierOrder.order_date, OrderItem.quantity, SupplierOrder.supplier_name)
        .join(OrderItem, OrderItem.order_id == SupplierOrder.id)
        .filter(OrderItem.product_id == product_id, OrderItem.quantity.isnot(None))
        .order_by(SupplierOrder.order_date.asc().nulls_first())
        .all()
    )


def baseline_forecast(product_id, window: int = DEFAULT_WINDOW):
    """Rolling average of the product's last `window` order quantities.

    Naturally degrades to "same as last order" when there's only one data
    point, and returns None when there's no history at all yet — the caller
    decides what "no forecast possible" means (skip, 422, etc.), this just
    reports it.
    """
    history = product_history(product_id)
    if not history:
        return None

    recent_quantities = [float(row.quantity) for row in history[-window:]]
    return sum(recent_quantities) / len(recent_quantities)


def forecast_range(product_id, window: int = DEFAULT_WINDOW) -> float | None:
    """Sample standard deviation of the same rolling window baseline_forecast
    averages over — a real ± spread around the forecast, not an invented
    number. None when fewer than 2 data points exist (stdev undefined).
    """
    history = product_history(product_id)
    recent = [float(row.quantity) for row in history[-window:]]
    if len(recent) < 2:
        return None

    mean = sum(recent) / len(recent)
    variance = sum((value - mean) ** 2 for value in recent) / (len(recent) - 1)
    return variance**0.5


def _ml_forecast(history, window: int = DEFAULT_WINDOW) -> float | None:
    """Tries the currently-registered ML model (see
    app/services/model_registry.py — no deployed endpoint, this just fetches
    whatever's currently registered, refreshing on every call so a freshly
    trained-and-registered model gets picked up without an app restart).
    Returns None if no model is registered — caller falls back to the
    baseline.
    """
    model = current_app.forecast_model
    if not model.refresh():
        return None

    recent_quantities = [float(row.quantity) for row in history[-window:]]
    today = date.today()

    return model.predict(
        lag_1=float(history[-1].quantity),
        lag_2=float(history[-2].quantity) if len(history) >= 2 else None,
        rolling_avg_3=sum(recent_quantities) / len(recent_quantities),
        month=today.month,
        day_of_week=today.weekday(),
        supplier_name=history[-1].supplier_name,
    )


def generate_forecast(product_id) -> ProductForecast | None:
    """Tries the registered ML model first, falling back to the rolling-
    average baseline if none is registered yet. Persists a new
    ProductForecast row either way. Returns None (persists nothing) if the
    product has no order history at all.
    """
    history = product_history(product_id)
    if not history:
        forecasts_generated.add(1, {"outcome": "skipped_no_history", "method": "none"})
        return None

    ml_prediction = _ml_forecast(history)
    if ml_prediction is not None:
        predicted_quantity = ml_prediction
        method = f"ml:{current_app.forecast_model.version}"
    else:
        recent_quantities = [float(row.quantity) for row in history[-DEFAULT_WINDOW:]]
        predicted_quantity = sum(recent_quantities) / len(recent_quantities)
        method = "baseline"

    forecast = ProductForecast(product_id=product_id, predicted_quantity=predicted_quantity, method=method)
    db.session.add(forecast)
    db.session.commit()
    forecasts_generated.add(1, {"outcome": "generated", "method": method})
    return forecast
