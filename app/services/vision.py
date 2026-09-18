from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential


VISUAL_FEATURES = [
    VisualFeatures.TAGS,
    VisualFeatures.OBJECTS,
    VisualFeatures.READ,
    VisualFeatures.PEOPLE,
]


class VisionAnalyzer:
    def __init__(self, endpoint: str, key: str | None = None):
        credential = AzureKeyCredential(key) if key else DefaultAzureCredential()
        self._client = ImageAnalysisClient(endpoint=endpoint, credential=credential)

    def analyze(self, image_bytes: bytes) -> dict:
        result = self._client.analyze(image_data=image_bytes, visual_features=VISUAL_FEATURES)
        if hasattr(result, "as_dict"):
            return result.as_dict()
        return dict(result)


def extract_summary(raw_result: dict) -> dict:
    """Best-effort extraction of searchable fields from the raw Azure response.

    Defensive by design: raw_result is always persisted in full regardless of
    whether these fields are found, so a shape mismatch here (e.g. a future
    API version) degrades search quality but never loses data.
    """
    caption_result = raw_result.get("captionResult") or {}
    tags_result = raw_result.get("tagsResult") or {}
    read_result = raw_result.get("readResult") or {}

    ocr_lines = [
        line.get("text")
        for block in (read_result.get("blocks") or [])
        for line in (block.get("lines") or [])
        if line.get("text")
    ]

    return {
        "caption": caption_result.get("text"),
        "caption_confidence": caption_result.get("confidence"),
        "tags": tags_result.get("values") or [],
        "ocr_text": "\n".join(ocr_lines) or None,
    }
