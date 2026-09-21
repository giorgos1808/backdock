import logging
import os

from flask import Flask
from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider

from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


def configure_telemetry(app: Flask) -> None:
    """Wires traces/metrics/logs to the OTLP collector (see
    deploy/docker-compose.observability.yml and, in Azure, the observability
    Terraform module).

    No-op unless OTEL_EXPORTER_OTLP_ENDPOINT is set, so the app behaves
    exactly as before for anyone running it without the observability stack.
    Azure SDK calls (Vision/Document Intelligence/Blob) get traced for free
    once a global TracerProvider exists — azure-core-tracing-opentelemetry
    picks it up automatically, no per-client wiring needed.
    """
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return

    resource = Resource.create({SERVICE_NAME: os.environ.get("OTEL_SERVICE_NAME", "backdock")})

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    meter_provider = MeterProvider(
        resource=resource, metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())]
    )
    metrics.set_meter_provider(meter_provider)

    LoggingInstrumentor().instrument(set_logging_format=True)

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
    set_logger_provider(logger_provider)
    logging.getLogger().addHandler(LoggingHandler(logger_provider=logger_provider))

    FlaskInstrumentor().instrument_app(app)
    PsycopgInstrumentor().instrument()
    RequestsInstrumentor().instrument()
    URLLib3Instrumentor().instrument()

    logger.info("OpenTelemetry enabled (endpoint=%s)", os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"])
