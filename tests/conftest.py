"""Shared test setup.

`app.config.Config` reads and validates env vars at *class definition* time
(i.e. on import), so anything the config requires has to be in the
environment before the first `from app...` import anywhere in the suite.
pytest imports conftest before collecting tests, which makes this the only
reliable place to do it.

Order matters: load the real .env first so a developer running locally gets
their actual resources (needed by the `integration` tests), then fill in
throwaway placeholders for whatever is still missing — which in CI is
everything. The placeholders only have to be well-formed enough to construct
the Azure clients; every client in this app defers network I/O past its
constructor, so nothing here ever dials out.
"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://crm:crm@localhost:5432/image_scanner")
os.environ.setdefault("AZURE_VISION_ENDPOINT", "https://placeholder.cognitiveservices.azure.com/")
os.environ.setdefault("AZURE_VISION_KEY", "placeholder-key")
os.environ.setdefault("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "https://placeholder.cognitiveservices.azure.com/")
os.environ.setdefault("AZURE_DOCUMENT_INTELLIGENCE_KEY", "placeholder-key")
os.environ.setdefault("AZURE_STORAGE_CONNECTION_STRING", "DefaultEndpointsProtocol=https;AccountName=placeholder;AccountKey=cGxhY2Vob2xkZXI=;EndpointSuffix=core.windows.net")


@pytest.fixture(scope="session")
def app():
    """The real application, wired exactly as production wires it.

    Used only by tests/integration/ — creating it is cheap (no network), but
    anything that then touches the database needs a real Postgres, because
    the models use JSONB and SQLite cannot create those tables.
    """
    from app import create_app

    application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app):
    return app.test_client()
