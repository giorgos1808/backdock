import logging
import os
import glob
import json

logger = logging.getLogger(__name__)

_warned_missing_dependencies = False


class ForecastModel:
    MODEL_NAME = "product-quantity-forecaster"

    def __init__(self, subscription_id: str | None, resource_group: str | None, workspace_name: str | None, local_model_path: str | None = None):
        self._subscription_id = subscription_id
        self._resource_group = resource_group
        self._workspace_name = workspace_name
        self._local_model_path = local_model_path
        self._booster = None
        self._supplier_categories: dict = {}
        self._version: str | None = None


    @property
    def configured(self) -> bool:
        return bool(self._local_model_path) or bool(
            self._subscription_id and self._resource_group and self._workspace_name
        )


    @property
    def version(self) -> str | None:
        return self._version


    def refresh(self) -> bool:
        """(Re)loads the latest registered model version if newer than
        whatever's currently loaded. Returns True if a model is loaded
        afterward (new or already-current), False if none is registered yet
        (or the ML workspace isn't configured at all — every caller treats
        that the same as "no model available", falling back to the baseline).
        """
        if not self.configured:
            return False

        if self._local_model_path:
            return self._load_local()

        try:
            import lightgbm as lgb
            from azure.ai.ml import MLClient
            from azure.core.exceptions import ResourceNotFoundError
            from azure.identity import DefaultAzureCredential
        except ImportError as exc:
            global _warned_missing_dependencies
            if not _warned_missing_dependencies:
                logger.warning(
                    "AZURE_ML_* is configured but the forecaster's scoring stack is not installed (%s). "
                    "Falling back to the baseline forecast. This is expected in the web image — ML "
                    "forecasts are produced by the forecaster image (scripts/run_forecasts.py).",
                    exc,
                )
                _warned_missing_dependencies = True
            return False

        client = MLClient(DefaultAzureCredential(), self._subscription_id, self._resource_group, self._workspace_name)

        try:
            latest = client.models.get(name=self.MODEL_NAME, label="latest")
        except ResourceNotFoundError:
            self._booster = None
            self._version = None
            return False

        if latest.version == self._version:
            return True

        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            client.models.download(name=self.MODEL_NAME, version=latest.version, download_path=tmp_dir)
            model_matches = glob.glob(os.path.join(tmp_dir, "**", "model.txt"), recursive=True)
            categories_matches = glob.glob(os.path.join(tmp_dir, "**", "supplier_categories.json"), recursive=True)
            if not model_matches or not categories_matches:
                raise FileNotFoundError(
                    f"Downloaded model {self.MODEL_NAME} v{latest.version} is missing model.txt or "
                    "supplier_categories.json — was it registered by train.py, or something else?"
                )

            self._booster = lgb.Booster(model_file=model_matches[0])
            with open(categories_matches[0]) as f:
                self._supplier_categories = json.load(f)

        self._version = latest.version
        return True


    def _load_local(self) -> bool:
        """Load a model straight off disk instead of from the registry.

        A development affordance, not how production works. The Azure ML round
        trip — train on a cluster, register, fetch the registered version — is
        slow to set up and needs a workspace, which makes it awkward to see the
        scoring path working at all. Pointing LOCAL_FORECAST_MODEL_PATH at a
        directory holding train.py's own outputs (model.txt and
        supplier_categories.json) short-circuits that.

        Forecasts produced this way are recorded as `ml:local` rather than
        `ml:<version>`, so a local model can never be mistaken in the database
        or the UI for one that actually went through the registration gate.
        """

        try:
            import lightgbm as lgb
        except ImportError as exc:
            global _warned_missing_dependencies
            if not _warned_missing_dependencies:
                logger.warning(
                    "LOCAL_FORECAST_MODEL_PATH is set but lightgbm is not installed (%s). Falling back "
                    "to the baseline forecast — the web image has no scoring stack by design.",
                    exc,
                )
                _warned_missing_dependencies = True
            return False

        if self._version == "local" and self._booster is not None:
            return True

        model_matches = glob.glob(os.path.join(self._local_model_path, "**", "model.txt"), recursive=True)
        categories_matches = glob.glob(
            os.path.join(self._local_model_path, "**", "supplier_categories.json"), recursive=True
        )
        if not model_matches or not categories_matches:
            logger.warning(
                "LOCAL_FORECAST_MODEL_PATH=%s has no model.txt / supplier_categories.json under it. "
                "Falling back to the baseline forecast.",
                self._local_model_path,
            )
            return False

        self._booster = lgb.Booster(model_file=model_matches[0])
        with open(categories_matches[0]) as f:
            self._supplier_categories = json.load(f)
        self._version = "local"
        logger.warning(
            "Scoring with a LOCAL model from %s. Forecasts are recorded as 'ml:local' and did not go "
            "through the registration gate.",
            self._local_model_path,
        )
        return True


    def predict(self, *, lag_1: float, lag_2: float | None, rolling_avg_3: float, month: int, day_of_week: int, supplier_name: str | None) -> float | None:
        """None if no model is currently loaded — caller falls back to the
        baseline. A supplier_name never seen during training (or None) gets
        code -1, an "unknown" bucket the model has never split on by
        construction (all training codes are >= 0), which is a reasonable
        conservative default rather than colliding with a real supplier.
        """
        if self._booster is None:
            return None

        supplier_code = self._supplier_categories.get(supplier_name, -1) if supplier_name else -1
        features = [[lag_1, lag_2 if lag_2 is not None else float("nan"), rolling_avg_3, month, day_of_week, supplier_code]]
        return float(self._booster.predict(features)[0])
