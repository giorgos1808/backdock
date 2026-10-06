# Local development

## Prerequisites

Three Azure resources: **Azure AI Vision**, **Azure AI Document
Intelligence**, and a **Storage account**. Grab each resource's endpoint/key
(and the storage connection string) from the Azure Portal. Blob containers
are created automatically on first upload.

## Setup

```bash
python -m venv venv
pip install -r requirements-dev.txt
```

Activating the venv differs by shell, and getting it wrong is the most common
way to lose time here — the failure is a `ModuleNotFoundError: No module named
'flask'` raised by your *system* Python, not by anything in this repo:

| Shell | Command |
| --- | --- |
| PowerShell | `.\venv\Scripts\Activate.ps1` |
| cmd.exe | `venv\Scripts\activate.bat` |
| Git Bash (Windows) | `source venv/Scripts/activate` |
| macOS / Linux | `source venv/bin/activate` |

**Every `python`, `pytest`, `ruff` and `flask` command below assumes the venv
is active.** To avoid depending on that, call the interpreter directly instead
— `.\venv\Scripts\python.exe -m ...` — which can't be affected by whichever
venv happens to be active in a given terminal.

`requirements-dev.txt` pulls in `requirements.txt`, `ml/requirements.txt`,
and the test/lint tooling, so a dev environment has everything. The runtime
files are split to match the two container images:

| File | Installed into | Holds |
| --- | --- | --- |
| `requirements.txt` | `web` image | Flask, the Azure AI/Blob SDKs, Postgres, OpenTelemetry |
| `requirements-forecaster.txt` | `forecaster` image | the above plus `azure-ai-ml` and `lightgbm` |
| `ml/requirements.txt` | nothing — local dev only | the training pipeline's libraries |

The web image deliberately has no scoring stack: `ForecastModel.refresh()`
treats `azure-ai-ml`/`lightgbm` as optional and falls back to the
rolling-average baseline when they're missing.

Then configure credentials:

```bash
copy .env.example .env
```

Fill in all `AZURE_*` variables. Leave `DATABASE_URL` as-is if you'll run
everything via Compose (option B below overrides it); set it to a real
reachable Postgres if you run the app directly on the host (option A).

`.env` is gitignored and must stay that way — it holds live keys.

## Running it

All Compose files live in `deploy/` and are written to be run **from the repo
root**, so their build context (`..`) and bind mounts resolve correctly.

### A. App on the host, Postgres in Docker

Best for active development — no image rebuild after every code change.

```bash
docker compose -f deploy/docker-compose.yml up -d postgres
flask db upgrade
python wsgi.py
```

Open http://localhost:5000

### B. Everything in Docker

Closer to how it actually runs in production.

```bash
docker compose -f deploy/docker-compose.yml up --build
```

Compose brings up Postgres, runs a one-shot `migrate` service
(`flask db upgrade`), and only starts the app if that succeeded — the same
migrate-then-serve split the Azure deploy uses. The app then serves with
gunicorn on http://localhost:5000. `DATABASE_URL` is overridden in the compose
file to reach the `postgres` service by container name; everything else comes
from `.env`. Rebuild with `--build` after dependency or code changes.

If the `migrate` service fails, `up` stops there and the app never starts —
check it with `docker compose -f deploy/docker-compose.yml logs migrate`.

### Running the forecaster

The batch forecaster is a separate image and sits behind a Compose profile, so
a plain `up` doesn't run it:

```bash
docker compose -f deploy/docker-compose.yml --profile forecaster run --rm forecaster
```

Without `AZURE_ML_*` set it regenerates every product's rolling-average
baseline forecast, which is the useful thing to exercise locally. With it set
— and only once a model has been trained and registered — it uses the
registered model instead.

### C. Against the real Azure Postgres

The dev database is VNet-injected with no public access, so this needs an SSH
tunnel through the jumpbox first:

```bash
ssh -N -L 5433:<postgres-server>.postgres.database.azure.com:5432 azureuser@<jumpbox-ip>
```

Then, in another terminal:

```bash
docker compose -f deploy/docker-compose.azure.yml up --build
```

It runs on port **5001** and as its own Compose project, so it can't collide
with the stack from option B — but don't run both app services at once.

### Observability stack (optional)

```bash
docker compose -f deploy/docker-compose.observability.yml up -d
```

Grafana on http://localhost:3000 (anonymous viewer access — dev only). Point
`OTEL_EXPORTER_OTLP_ENDPOINT` at the collector's OTLP/**HTTP** port `4318`
(not the gRPC port 4317 — see `app/telemetry.py` for why this app uses the
HTTP exporter):

- app running directly on the host: `http://localhost:4318`
- app running via Compose: `http://host.docker.internal:4318`

## Getting data to look at

A fresh database is empty, so `/orders`, `/products` and the forecast pages
all render as empty shells. Two ways to fill it:

**Scan something.** Upload a supplier order at `/orders/scan`, then product
labels against it. This is the real flow and needs working `AZURE_*`
credentials.

**Load the demand dataset.** No Azure needed, and it gives the forecaster
enough history to be worth looking at:

```bash
curl -sSL -o train.parquet \
  https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/main/data/train.parquet

python -m scripts.load_demand_dataset --parquet train.parquet --stores 3 --reset
python -m scripts.run_forecasts
```

That loads ~291 products, 216 supplier orders and 3,491 order lines, then
generates a forecast for every product. `train.parquet` is gitignored and can
be deleted afterwards. See
[ml-pipeline.md](ml-pipeline.md#getting-training-data).

**Load sample product labels.** To browse real label-extraction output —
expiry dates and lot numbers read off actual packaging — ingest saved
Document Intelligence responses:

```bash
python -m scripts.load_label_samples --responses sample_data/expdate_responses --reset
```

They land on one dedicated order that deliberately has no line items, so every
label reconciles as `no_order`. That demonstrates extraction, not
reconciliation: seeing matching, `expired` detection and the missing-items
summary work needs a real supplier order plus labels from those same goods,
which no public dataset can provide. The script can also analyse fresh images
with `--images`, which calls Azure and costs money.

Everything both loaders write is tagged `raw_result._synthetic` and removed by
`--reset`, so loaded data never gets confused with a real scan.

## Stopping the stack

```bash
docker compose -f deploy/docker-compose.yml down      # keeps the database
docker compose -f deploy/docker-compose.yml down -v   # also deletes it
```

The `-v` is worth being deliberate about: it removes the `postgres_data`
volume, and everything scanned or loaded goes with it.

## Tests

```bash
pytest tests/unit               # no database, no credentials, runs anywhere
pytest -m integration           # needs Postgres up and migrations applied
pytest                          # everything
```

The unit suite constructs nothing that touches the network. The integration
suite builds the real app and hits real routes, so it needs a live Postgres —
the models use JSONB, which SQLite cannot create. `tests/conftest.py` loads
your `.env` first and only falls back to placeholder credentials for whatever
is missing, so locally you get your real resources and in CI you get
placeholders.

## Linting

```bash
ruff check .
ruff check . --fix
```

Config lives in `pyproject.toml`. `ruff format` is **not** enforced in CI —
running it would reformat 8 existing files, which is a deliberate, separate
change rather than something to slip into an unrelated PR.

## Schema changes

This project uses Flask-Migrate/Alembic. After changing `app/models.py`:

```bash
flask db migrate -m "describe the change"
flask db upgrade
```

Review autogenerated migrations before applying:

- Alembic doesn't reliably autodetect expression/functional indexes (see the
  `scans_search_idx` GIN index, added by hand in the initial migration).
- Autogenerate will also propose *dropping* that index on any later
  migration, since it isn't represented in the ORM metadata — that's
  spurious; strip it out (see the second migration for an example of what was
  removed).
