"""
docker compose -f deploy/docker-compose.yml up -d postgres
flask db upgrade
pytest -m integration
"""

import io
import pytest

pytestmark = pytest.mark.integration

# Any well-formed UUID that will not exist in a freshly migrated database.
MISSING_ID = "00000000-0000-0000-0000-000000000000"


def test_app_factory_wires_every_service_client(app):
    """create_app() attaches the clients that routes reach through
    current_app; a missing one otherwise only surfaces as an AttributeError
    deep inside a request handler.
    """
    for attribute in ("vision_analyzer", "blob_storage", "document_intelligence", "order_blob_storage", "label_blob_storage", "forecast_model"):
        assert getattr(app, attribute, None) is not None, f"create_app() did not attach {attribute}"


@pytest.mark.parametrize("path", ["/", "/gallery", "/scans", "/orders", "/products"])
def test_listing_pages_render(client, path):
    response = client.get(path)

    assert response.status_code == 200


def test_scans_listing_returns_json_when_asked(client):
    response = client.get("/scans", headers={"Accept": "application/json"})

    assert response.status_code == 200
    assert response.is_json


@pytest.mark.parametrize("path", [f"/scans/{MISSING_ID}", f"/orders/{MISSING_ID}", f"/products/{MISSING_ID}"])
def test_unknown_records_are_404_not_500(client, path):
    response = client.get(path)

    assert response.status_code == 404


def test_malformed_uuid_does_not_reach_the_database(client):
    """The `<uuid:...>` converter should reject this at routing time."""
    response = client.get("/scans/not-a-uuid")

    assert response.status_code == 404


def test_upload_over_the_size_limit_is_rejected(client):
    """MAX_CONTENT_LENGTH is 10 MB; Flask should refuse the request itself
    rather than a route streaming it into memory first.
    """
    oversized = io.BytesIO(b"x" * (11 * 1024 * 1024))

    response = client.post("/scan", data={"image": (oversized, "big.jpg")})

    assert response.status_code == 413
