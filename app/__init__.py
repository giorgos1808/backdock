from flask import Flask

from app.config import Config
from app.extensions import db, migrate
from app.services.document_intelligence import DocumentIntelligenceAnalyzer
from app.services.model_registry import ForecastModel
from app.services.storage import BlobStorage
from app.services.vision import VisionAnalyzer
from app.telemetry import configure_telemetry

MAX_CONTENT_LENGTH = 10 * 1024 * 1024


def create_app(config_class: type = Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    configure_telemetry(app)

    db.init_app(app)
    migrate.init_app(app, db)

    app.vision_analyzer = VisionAnalyzer(app.config["AZURE_VISION_ENDPOINT"], app.config["AZURE_VISION_KEY"])

    storage_auth = {
        "connection_string": app.config["AZURE_STORAGE_CONNECTION_STRING"],
        "account_url": app.config["AZURE_STORAGE_ACCOUNT_URL"],
    }
    app.blob_storage = BlobStorage(app.config["AZURE_STORAGE_CONTAINER"], **storage_auth)

    app.document_intelligence = DocumentIntelligenceAnalyzer(
        app.config["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"], app.config["AZURE_DOCUMENT_INTELLIGENCE_KEY"]
    )
    app.order_blob_storage = BlobStorage(app.config["AZURE_STORAGE_CONTAINER_ORDERS"], **storage_auth)
    app.label_blob_storage = BlobStorage(app.config["AZURE_STORAGE_CONTAINER_LABELS"], **storage_auth)

    app.forecast_model = ForecastModel(
        app.config["AZURE_ML_SUBSCRIPTION_ID"],
        app.config["AZURE_ML_RESOURCE_GROUP"],
        app.config["AZURE_ML_WORKSPACE_NAME"],
        app.config["LOCAL_FORECAST_MODEL_PATH"],
    )

    from app.routes import register_blueprints

    register_blueprints(app)

    return app
