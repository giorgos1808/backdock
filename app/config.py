import os

from dotenv import load_dotenv

load_dotenv()


def require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required (see .env.example)")
    return value


class Config:
    SQLALCHEMY_DATABASE_URI = require("DATABASE_URL")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    AZURE_VISION_ENDPOINT = require("AZURE_VISION_ENDPOINT")
    AZURE_VISION_KEY = os.environ.get("AZURE_VISION_KEY")

    AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    AZURE_STORAGE_ACCOUNT_URL = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")

    if not AZURE_STORAGE_CONNECTION_STRING and not AZURE_STORAGE_ACCOUNT_URL:
        raise RuntimeError("Either AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL is required (see .env.example)")

    AZURE_STORAGE_CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER", "scanned-images")

    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = require("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    AZURE_DOCUMENT_INTELLIGENCE_KEY = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    AZURE_STORAGE_CONTAINER_ORDERS = os.environ.get("AZURE_STORAGE_CONTAINER_ORDERS", "supplier-orders")
    AZURE_STORAGE_CONTAINER_LABELS = os.environ.get("AZURE_STORAGE_CONTAINER_LABELS", "product-labels")

    AZURE_ML_SUBSCRIPTION_ID = os.environ.get("AZURE_ML_SUBSCRIPTION_ID")
    AZURE_ML_RESOURCE_GROUP = os.environ.get("AZURE_ML_RESOURCE_GROUP")
    AZURE_ML_WORKSPACE_NAME = os.environ.get("AZURE_ML_WORKSPACE_NAME")

    LOCAL_FORECAST_MODEL_PATH = os.environ.get("LOCAL_FORECAST_MODEL_PATH")
