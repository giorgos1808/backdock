import time

from app import create_app
from app.metrics import forecast_job_duration
from app.models import Product
from app.services.forecasting import generate_forecast


def run() -> None:
    start = time.monotonic()
    generated = 0
    skipped = 0

    try:
        for product in Product.query.all():
            if generate_forecast(product.id) is None:
                skipped += 1
            else:
                generated += 1
    except Exception:
        forecast_job_duration.record(time.monotonic() - start, {"outcome": "failed"})
        raise

    forecast_job_duration.record(time.monotonic() - start, {"outcome": "success"})
    print(f"Generated {generated} forecast(s), skipped {skipped} product(s) with no history yet.")


if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        run()

    from opentelemetry import metrics as otel_metrics
    from opentelemetry import trace

    for provider in (trace.get_tracer_provider(), otel_metrics.get_meter_provider()):
        force_flush = getattr(provider, "force_flush", None)
        if force_flush is not None:
            force_flush()
