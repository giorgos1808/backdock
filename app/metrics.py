from opentelemetry import metrics

meter = metrics.get_meter("backdock")

scans_processed = meter.create_counter(
    "crm.scans.processed",
    description="Image scans processed, by outcome",
    unit="1",
)

orders_processed = meter.create_counter(
    "crm.orders.processed",
    description="Supplier order documents processed, by outcome",
    unit="1",
)

labels_processed = meter.create_counter(
    "crm.labels.processed",
    description="Product label images processed, by outcome",
    unit="1",
)

label_matches = meter.create_counter(
    "crm.labels.match_result",
    description="Label-to-order-item fuzzy match attempts, by result",
    unit="1",
)

label_match_score = meter.create_histogram(
    "crm.labels.match_score",
    description="Fuzzy match score (0-100) for every match attempt, regardless of result",
    unit="1",
)

forecasts_generated = meter.create_counter(
    "crm.forecasts.generated",
    description="Per-product baseline forecast attempts, by outcome",
    unit="1",
)

forecast_job_duration = meter.create_histogram(
    "crm.forecasts.job_duration",
    description="Wall-clock duration of a full run_forecasts job",
    unit="s",
)
