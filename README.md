# Backdock

Scan supplier orders and product labels, reconcile one against the other, and
forecast what to reorder.

*Named for the back dock — the receiving door behind the shop, where the
delivery meets the paperwork.*

A Flask app on Azure that turns paper documents into queryable data. Three
scanning flows, all backed by prebuilt Azure AI services:

1. **General image scan** — Azure AI Vision (captions, tags, OCR, objects).
2. **Supplier order** — Document Intelligence `prebuilt-invoice` extracts
   structured line items from an order document.
3. **Product label** — Document Intelligence `prebuilt-layout` plus
   heuristics pull out product name, expiration date, and lot number.

Scanned labels are fuzzy-matched against their order's line items, flagging
missing items and expired stock. That reconciliation is the point — not
storing two kinds of JSON. Matched products accumulate per-product order
history, which feeds a quantity forecast: a rolling-average baseline, or a
LightGBM model from the Azure ML pipeline once one beats that baseline.

Originals live in Azure Blob Storage; all structured data lives in Postgres.

## Repository status

**This repository is mid-import and is not yet runnable from a clone.** The
application's backend — models, services, routes — is here, along with the
database schema, the ML pipeline, and the templates for the label and
forecasting flows. The stylesheets, the remaining templates, the entry point,
tests, infrastructure and CI have not been pushed yet.

If you are reading the code to see how it works, everything below is real and
complete. If you are trying to run it, come back once the sections under
[Not here yet](#not-here-yet) have landed.

## What's here

```
app/
  models.py          SQLAlchemy models — scans, orders, line items,
                     product labels, the products catalogue, forecasts
  config.py          Every Azure value read from the environment, no defaults
  extensions.py      SQLAlchemy and Migrate instances
  metrics.py         The crm.* OpenTelemetry counters
  telemetry.py       OTLP/HTTP tracing, metrics and logs; inert unless
                     OTEL_EXPORTER_OTLP_ENDPOINT is set
  routes/            Blueprints: pages, scans, orders, labels, products
  services/          Where the work happens:
                       vision.py                  Azure AI Vision client
                       document_intelligence.py   Invoice and layout
                                                  extraction, plus the
                                                  expiry/lot heuristics
                       reconciliation.py          Labels against line items
                       labels.py, orders.py       Matching and status
                       products.py                Fuzzy catalogue merging
                       forecasting.py             Baseline and ML forecasts
                       model_registry.py          Azure ML model loading,
                                                  with a local-file fallback
                       storage.py                 Blob Storage
                       text.py                    Normalisation shared by
                                                  the matchers
  templates/         Jinja templates for the label and forecasting flows,
                     plus base.html, index.html and gallery.html.

migrations/          Six Alembic revisions, from the initial scans table
                     through supplier orders, the products catalogue and
                     product_forecasts

ml/                  The Azure ML pipeline that trains the quantity
                     forecaster: data prep, train, evaluate, register.
                     A model is only registered when it beats the
                     rolling-average baseline

observability/       Tempo, Loki, Prometheus, Promtail and OTel Collector
                     configuration for the self-hosted stack

requirements.txt                Runtime dependencies for the web image
requirements-forecaster.txt     Adds the scoring stack — azure-ai-ml and
                                lightgbm — for the forecaster image only
requirements-dev.txt            The above plus pytest, ruff and pip-audit
```

The dependency split is deliberate: the forecaster's scoring stack is roughly
two thirds of the image, and `app/services/model_registry.py` imports it
lazily, so the web image falls back to the rolling-average baseline when it is
absent.

## Not here yet

| | |
| --- | --- |
| `app/static/` | The stylesheets — nothing here renders styled without them |
| The order and general-scan templates | `order_*.html`, `orders_list.html`, `result.html`, `review_result.html`, `scan_detail.html` |
| `app/templates/v2/`, `app/routes/v2.py` | The redesigned interface, desktop and mobile |
| `wsgi.py` | The entry point |
| `tests/` | Unit and integration suites |
| `terraform/` | Azure infrastructure as code |
| `.github/workflows/` | CI and deployment pipelines |
| `Dockerfile`, `deploy/` | Container image and the local Compose stacks |
| `docs/` | Architecture, usage, deployment, runbook, ML pipeline |
| `scripts/` | Dataset loaders and the batch forecast run |

## Requirements

Python 3.12. The models use JSONB throughout, so Postgres is required.

The application authenticates to Azure AI Vision, Document Intelligence and
Blob Storage with `DefaultAzureCredential`, falling back to key-based
authentication where a key is set. No credentials are stored in this
repository, and none should ever be committed to it.

## Status

The Document Intelligence field mappings are verified against live API
responses rather than written from the documentation, and the Greek expiry and
lot-number keywords follow EU labelling conventions.

Known gaps, in rough priority: there is no health endpoint, so a
booted-but-broken revision is indistinguishable from a healthy one; there is
no alerting; no backup restore has been drilled; and there is no
authentication in front of the app. Do not put real customer data behind an
internet-reachable deployment until that last one is addressed.

## Licence

MIT — see [LICENSE](LICENSE).
