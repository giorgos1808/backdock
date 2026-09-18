import os
import uuid

from azure.core.exceptions import ResourceExistsError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings


class BlobStorage:
    """Thin wrapper around Azure Blob Storage for the original scanned images.

    Container creation is deferred to the first upload so constructing this
    class (e.g. at Flask app creation time, for CLI commands like `flask db
    migrate`) never requires network access to Azure.

    Accepts either a connection string (local/key-based auth) or an account
    URL (Managed Identity via DefaultAzureCredential, used in production).
    """

    def __init__(self, container_name: str, connection_string: str | None = None, account_url: str | None = None):
        if connection_string:
            service_client = BlobServiceClient.from_connection_string(connection_string)
        elif account_url:
            service_client = BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())
        else:
            raise ValueError("BlobStorage requires either connection_string or account_url")

        self._container_client = service_client.get_container_client(container_name)
        self._container_ready = False

    def _ensure_container(self) -> None:
        if self._container_ready:
            return
        try:
            self._container_client.create_container()
        except ResourceExistsError:
            pass
        self._container_ready = True

    def upload(self, data: bytes, filename: str, content_type: str) -> tuple[str, str]:
        self._ensure_container()

        ext = os.path.splitext(filename)[1]
        blob_name = f"{uuid.uuid4().hex}{ext}"

        blob_client = self._container_client.get_blob_client(blob_name)
        blob_client.upload_blob(data, content_settings=ContentSettings(content_type=content_type))

        return blob_name, blob_client.url

    def download(self, blob_name: str) -> tuple[bytes, str]:
        """Fetch a blob's bytes and content type.

        Containers are private (no anonymous read access), so the browser
        can't load `blob_url` directly — routes proxy the bytes through this
        instead, using the app's own credentials.
        """
        blob_client = self._container_client.get_blob_client(blob_name)
        downloader = blob_client.download_blob()
        content_type = downloader.properties.content_settings.content_type or "application/octet-stream"
        return downloader.readall(), content_type
