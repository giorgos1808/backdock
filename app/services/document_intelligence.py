import calendar
import re
from datetime import date

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentAnalysisFeature
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from app.services.text import fold_accents as _fold

EXPIRY_KEYWORDS = re.compile(
    r"\b(exp(?:iry|iration)?(?:\s*date)?|best\s*before|bb\s*date|use\s*by|αναλωση|ληξη[σς]?)\b",
    re.IGNORECASE,
)

EXPIRY_LOOKAHEAD_LINES = 4

EXPIRY_KEYWORDS = re.compile(
    r"\b(?:lot|batch|παρτιδα[σς]?|παρτ)\b[\s.:#-]*"
    r"(?:(?:n(?:o|r|um(?:ber)?)?|νο|αρ(?:ιθ(?:μος)?)?)\b[\s.:#-]*)?"
    r"([A-Za-z0-9][A-Za-z0-9\-]{2,})",
    re.IGNORECASE,
)

DATE_PATTERNS = [
    # Day first, e.g. 25/08/2023.
    re.compile(r"\b(?P<a>\d{1,2})\s*(?P<sep>[/.\-])\s*(?P<b>\d{1,2})\s*(?P=sep)\s*(?P<c>\d{2,4})\b"),
    # Year first, e.g. 2023-08-25.
    re.compile(r"\b(?P<a>\d{4})\s*(?P<sep>[/.\-])\s*(?P<b>\d{1,2})\s*(?P=sep)\s*(?P<c>\d{1,2})\b"),
    # Spaced with no punctuation at all, e.g. "25 08 2023".
    re.compile(r"\b(?P<a>\d{1,2})\s+(?P<b>\d{1,2})\s+(?P<c>\d{2,4})\b"),
    re.compile(r"\b(?P<a>\d{4})\s+(?P<b>\d{1,2})\s+(?P<c>\d{1,2})\b"),
]

MONTH_YEAR_PATTERN = re.compile(
    r"(?<![\d/.\-])(?P<month>\d{1,2})\s*[/.\-]\s*(?P<year>\d{4})(?![\d/.\-])"
)

MAX_NAME_LENGTH = 50


class DocumentIntelligenceAnalyzer:
    def __init__(self, endpoint: str, key: str | None = None):
        credential = AzureKeyCredential(key) if key else DefaultAzureCredential()
        self._client = DocumentIntelligenceClient(endpoint=endpoint, credential=credential)


    def analyze_invoice(self, document_bytes: bytes) -> dict:
        """Supplier order documents are structurally invoices/purchase orders:
        vendor + line items (description/quantity/price). prebuilt-invoice
        extracts those as structured fields instead of raw text.
        """
        poller = self._client.begin_analyze_document(
            "prebuilt-invoice", AnalyzeDocumentRequest(bytes_source=document_bytes)
        )
        return poller.result().as_dict()


    def analyze_label(self, document_bytes: bytes) -> dict:
        """Product labels vary too much per supplier for a prebuilt template,
        so this uses generic layout (text + key-value pairs) and leaves field
        extraction to extract_label_fields() heuristics below.
        """
        poller = self._client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(bytes_source=document_bytes),
            features=[DocumentAnalysisFeature.KEY_VALUE_PAIRS],
        )
        return poller.result().as_dict()


def _first(d: dict, *keys):
    """Azure SDK model .as_dict() key casing (camelCase vs snake_case) varies
    by SDK version; try a few spellings rather than assuming one.
    """
    for key in keys:
        value = d.get(key)
        if value is not None:
            return value
    return None


def extract_order_summary(raw_result: dict) -> dict:
    """Best-effort mapping of a prebuilt-invoice result onto our order shape."""
    documents = raw_result.get("documents") or []
    if not documents:
        return {"supplier_name": None, "order_number": None, "order_date": None, "items": []}

    fields = documents[0].get("fields") or {}

    def field_text(name: str):
        field = fields.get(name)
        if not field:
            return None
        return _first(field, "valueString", "value_string", "content")

    supplier_name = field_text("VendorName")
    order_number = field_text("PurchaseOrder") or field_text("InvoiceId")

    invoice_date_field = fields.get("InvoiceDate") or {}
    order_date = _first(invoice_date_field, "valueDate", "value_date")

    items = []
    items_field = fields.get("Items") or {}
    for item in _first(items_field, "valueArray", "value_array") or []:
        item_fields = _first(item, "valueObject", "value_object") or {}

        description_field = item_fields.get("Description") or {}
        quantity_field = item_fields.get("Quantity") or {}
        unit_price_field = item_fields.get("UnitPrice") or {}
        product_code_field = item_fields.get("ProductCode") or {}

        unit_price = _first(unit_price_field, "valueNumber", "value_number")
        if unit_price is None:
            currency = _first(unit_price_field, "valueCurrency", "value_currency") or {}
            unit_price = currency.get("amount")

        items.append(
            {
                "product_name": _first(description_field, "valueString", "value_string", "content"),
                "product_code": _first(product_code_field, "valueString", "value_string", "content"),
                "quantity": _first(quantity_field, "valueNumber", "value_number"),
                "unit_price": unit_price,
                "raw_line": item_fields,
            }
        )

    return {
        "supplier_name": supplier_name,
        "order_number": order_number,
        "order_date": order_date,
        "items": items,
    }


def _has_expiry_keyword(text: str) -> bool:
    return bool(EXPIRY_KEYWORDS.search(_fold(text)))


def _find_lot_number(text: str):
    """The lot code as printed, or None.

    Matched against the accent-folded text but sliced out of the original, so
    a Greek keyword is recognised without the returned value being altered.
    """
    match = EXPIRY_KEYWORDS.search(_fold(text))
    return text[match.start(1) : match.end(1)] if match else None


def _extract_lines(raw_result: dict) -> list:
    lines = []
    for page in raw_result.get("pages") or []:
        for line in page.get("lines") or []:
            content = line.get("content")
            if content:
                lines.append(content)
    return lines


def _parse_date(text: str):
    """First readable date in `text`, or None.

    Every match of each pattern is tried, not just the first: a label line can
    carry a number that looks date-shaped before the real date does ("LOT
    1-2-3 EXP 25/08/2023"), and stopping at the first match meant one bad
    candidate hid a perfectly good date behind it.

    Two-digit years are read day-first, which is ambiguous by construction:
    "21.12.22" is both a valid 21 Dec 2022 and a valid 2021-12-22, and nothing
    in the string says which. Against ExpDate that costs ~5% of dates, read as
    the wrong day rather than rejected. The PATCH correction endpoint exists
    for exactly this.
    """
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            a, b, c = int(match.group("a")), int(match.group("b")), int(match.group("c"))
            if a > 999:  # year first
                year, month, day = a, b, c
            else:  # day first
                day, month, y = a, b, c
                year = y if y > 99 else 2000 + y

            try:
                return date(year, month, day)
            except ValueError:
                continue

    for match in MONTH_YEAR_PATTERN.finditer(text):
        month, year = int(match.group("month")), int(match.group("year"))
        if 1 <= month <= 12:
            return date(year, month, calendar.monthrange(year, month)[1])
    return None


def _line_heights(raw_result: dict):
    """(content, relative height) for every OCR'd line.

    Height comes from the line's polygon and is divided by the page height, so
    it stays comparable across pages and image resolutions.
    """
    for page in raw_result.get("pages") or []:
        page_height = page.get("height") or 0
        for line in page.get("lines") or []:
            content = (line.get("content") or "").strip()
            polygon = line.get("polygon") or []
            if not content or len(polygon) < 8 or not page_height:
                continue
            ys = polygon[1::2]
            yield content, (max(ys) - min(ys)) / page_height


def _could_be_a_product_name(text: str) -> bool:
    """Rejects lines that are definitely something else.

    Deliberately a filter on the obviously-wrong rather than a test for the
    right answer: packaging carries plenty of text and there is no reliable
    signal for "this is the product". Dates, the expiry and lot lines,
    barcodes, prose and stray OCR fragments are all knowably not it.
    """
    if len(text) > MAX_NAME_LENGTH:
        return False
    if _has_expiry_keyword(text) or _find_lot_number(text) or _parse_date(text):
        return False
    letters = sum(1 for character in text if character.isalpha())
    if letters < 3:
        return False
    # Barcodes, weights and nutrition figures
    return letters >= len(text.replace(" ", "")) * 0.4


def _guess_product_name(raw_result: dict, lines: list):
    """Best guess at the product name, in descending order of trust.

    A document's own title, when Document Intelligence labels one, is by far
    the strongest signal — but prebuilt-layout assigns no paragraph roles to
    photographs of packaging, so on labels that never fires.

    Falling back to the first OCR'd line was wrong about a third of the time
    against real packaging: reading order starts wherever the model starts, so
    it landed on "BEST BEFORE 11/11/21", "Use By:", a barcode, or a two-letter
    scrap. Largest printed text is a much better proxy — a product name is the
    thing a package puts in the biggest type — and it drops that to 8%.

    It is still a guess. The remaining misses are real words printed large
    that simply aren't the product ("유통기한" — expiry date; "영양정보" —
    nutrition information), and no amount of geometry distinguishes those
    without reading the language. PATCH /orders/<id>/labels/<id> is the
    intended correction path.
    """

    for paragraph in raw_result.get("paragraphs") or []:
        content = paragraph.get("content")
        if paragraph.get("role") in ("title", "sectionHeading") and content and _could_be_a_product_name(content):
            return content

    sized = [(height, content) for content, height in _line_heights(raw_result) if _could_be_a_product_name(content)]
    if sized:
        return max(sized)[1]

    for line in lines:
        if _could_be_a_product_name(line):
            return line

    return None


def extract_label_fields(raw_result: dict) -> dict:
    """Heuristic extraction for highly variable label layouts.

    Deliberately conservative: expiration_date is only set when found next to
    an explicit keyword (never guessed from "the first date on the label",
    which risks picking a manufacture date instead). Expect these fields to
    need manual correction via PATCH /orders/<id>/labels/<id> — this is a
    starting point to tune against real samples, not a finished parser.
    """
    lines = _extract_lines(raw_result)

    expiration_date = None
    lot_number = None

    for i, line in enumerate(lines):
        if expiration_date is None and _has_expiry_keyword(line):
            expiration_date = _parse_date(line)
            for offset in range(1, EXPIRY_LOOKAHEAD_LINES + 1):
                if expiration_date is not None or i + offset >= len(lines):
                    break
                expiration_date = _parse_date(lines[i + offset])

        if lot_number is None:
            lot_number = _find_lot_number(line)

    return {
        "product_name": _guess_product_name(raw_result, lines),
        "expiration_date": expiration_date,
        "lot_number": lot_number,
    }
