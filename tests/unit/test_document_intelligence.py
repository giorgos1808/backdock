from datetime import date
import pytest
from app.services.document_intelligence import extract_label_fields, extract_order_summary


def label_result(*lines: str, paragraphs: list | None = None) -> dict:
    result = {"pages": [{"lines": [{"content": line} for line in lines]}]}
    if paragraphs is not None:
        result["paragraphs"] = paragraphs
    return result


class TestExtractLabelFields:
    def test_pulls_name_expiry_and_lot(self):
        raw = label_result("ACME Tomato Sauce 400g", "EXP 01/02/2027", "LOT ABC123")

        fields = extract_label_fields(raw)

        assert fields["product_name"] == "ACME Tomato Sauce 400g"
        assert fields["expiration_date"] == date(2027, 2, 1)
        assert fields["lot_number"] == "ABC123"

    def test_dates_are_read_day_first(self):
        """01/02/2027 is 1 February, not 2 January — day-first is the common
        convention outside the US and the one this parser commits to.
        """
        fields = extract_label_fields(label_result("Best before 03/04/2028"))

        assert fields["expiration_date"] == date(2028, 4, 3)

    def test_iso_dates_are_understood(self):
        fields = extract_label_fields(label_result("EXP 2027-03-15"))

        assert fields["expiration_date"] == date(2027, 3, 15)

    def test_expiry_on_the_following_line_is_found(self):
        """Labels routinely put the keyword and the date on separate lines."""
        fields = extract_label_fields(label_result("Tomato Sauce", "BEST BEFORE", "12/12/2028"))

        assert fields["expiration_date"] == date(2028, 12, 12)

    def test_a_bare_date_is_never_assumed_to_be_an_expiry(self):
        """The deliberate conservative case: a date with no expiry keyword
        next to it could just as easily be a packing or manufacture date, so
        the field is left for a human to correct via PATCH.
        """
        fields = extract_label_fields(label_result("Tomato Sauce", "Packed 05/05/2025"))

        assert fields["expiration_date"] is None

    def test_impossible_date_is_rejected_rather_than_crashing(self):
        fields = extract_label_fields(label_result("EXP 31/02/2027"))

        assert fields["expiration_date"] is None

    def test_title_paragraph_wins_over_the_first_line(self):
        raw = label_result(
            "Distributed by ACME Ltd",
            "EXP 01/01/2030",
            paragraphs=[{"role": "title", "content": "Organic Tomato Sauce"}],
        )

        assert extract_label_fields(raw)["product_name"] == "Organic Tomato Sauce"

    def test_empty_result_yields_all_none(self):
        assert extract_label_fields({}) == {"product_name": None, "expiration_date": None, "lot_number": None}


class TestExtractOrderSummary:
    def test_no_documents_yields_an_empty_order(self):
        assert extract_order_summary({"documents": []}) == {
            "supplier_name": None,
            "order_number": None,
            "order_date": None,
            "items": [],
        }

    def test_maps_vendor_number_date_and_line_items(self):
        raw = {
            "documents": [
                {
                    "fields": {
                        "VendorName": {"valueString": "ACME Foods Ltd"},
                        "InvoiceId": {"valueString": "INV-1001"},
                        "InvoiceDate": {"valueDate": "2026-01-05"},
                        "Items": {
                            "valueArray": [
                                {
                                    "valueObject": {
                                        "Description": {"valueString": "Tomato Sauce 400g"},
                                        "Quantity": {"valueNumber": 12},
                                        "UnitPrice": {"valueNumber": 1.5},
                                    }
                                }
                            ]
                        },
                    }
                }
            ]
        }

        summary = extract_order_summary(raw)

        assert summary["supplier_name"] == "ACME Foods Ltd"
        assert summary["order_number"] == "INV-1001"
        assert summary["order_date"] == "2026-01-05"
        assert len(summary["items"]) == 1
        assert summary["items"][0]["product_name"] == "Tomato Sauce 400g"
        assert summary["items"][0]["quantity"] == 12
        assert summary["items"][0]["unit_price"] == 1.5
        assert summary["items"][0]["product_code"] is None

    def test_purchase_order_number_is_preferred_over_invoice_id(self):
        raw = {
            "documents": [
                {"fields": {"PurchaseOrder": {"valueString": "PO-77"}, "InvoiceId": {"valueString": "INV-1001"}}}
            ]
        }

        assert extract_order_summary(raw)["order_number"] == "PO-77"

    def test_snake_case_key_casing_is_accepted(self):
        """The SDK's .as_dict() casing has varied between versions, which is
        why the parser tries several spellings of every key.
        """
        raw = {
            "documents": [
                {
                    "fields": {
                        "VendorName": {"value_string": "ACME Foods Ltd"},
                        "InvoiceDate": {"value_date": "2026-01-05"},
                        "Items": {
                            "value_array": [
                                {"value_object": {"Description": {"value_string": "Olive Oil 1L"}}},
                            ]
                        },
                    }
                }
            ]
        }

        summary = extract_order_summary(raw)

        assert summary["supplier_name"] == "ACME Foods Ltd"
        assert summary["order_date"] == "2026-01-05"
        assert summary["items"][0]["product_name"] == "Olive Oil 1L"

    def test_unit_price_falls_back_to_the_currency_object(self):
        raw = {
            "documents": [
                {
                    "fields": {
                        "Items": {
                            "valueArray": [
                                {
                                    "valueObject": {
                                        "Description": {"valueString": "Olive Oil 1L"},
                                        "UnitPrice": {"valueCurrency": {"amount": 8.25, "currencyCode": "EUR"}},
                                    }
                                }
                            ]
                        }
                    }
                }
            ]
        }

        assert extract_order_summary(raw)["items"][0]["unit_price"] == 8.25


class TestDateSeparators:
    """Format coverage measured against ExpDate (1,767 real packaging photos,
    CC BY 4.0). Space-separated dates were every single date the parser failed
    to read at all; fixing that took it from 90.1% to 95.4% correct on 593
    ground-truthed date strings.
    """

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("EXP 25/08/2023", date(2023, 8, 25)),
            ("EXP 25.08.2023", date(2023, 8, 25)),
            ("EXP 25-08-2023", date(2023, 8, 25)),
            ("EXP 25 08 2023", date(2023, 8, 25)),
            ("BEST BEFORE 2021 08 24", date(2021, 8, 24)),
            ("USE BY 10 06 21", date(2021, 6, 10)),
        ],
    )
    def test_every_separator_a_printer_uses(self, line, expected):
        assert extract_label_fields(label_result(line))["expiration_date"] == expected

    def test_separators_must_be_consistent(self):
        """Requiring the same separator twice is what stops the space variant
        from swallowing unrelated runs of numbers on a label.
        """
        assert extract_label_fields(label_result("EXP 25/08 2023"))["expiration_date"] is None

    def test_a_run_of_numbers_is_not_read_as_a_date(self):
        assert extract_label_fields(label_result("EXP NET 25 50 100 g"))["expiration_date"] is None

    def test_a_bad_candidate_does_not_hide_a_good_date(self):
        """Every match is tried, not just the first. An impossible date earlier
        in the line used to mask a perfectly readable one after it.
        """
        fields = extract_label_fields(label_result("LOT 99/99/99 EXP 25/08/2023"))

        assert fields["expiration_date"] == date(2023, 8, 25)

    def test_two_digit_years_are_ambiguous_and_read_day_first(self):
        """Known limitation, not an oversight: '21.12.22' is equally a valid
        21 Dec 2022 and a valid 2021-12-22, and nothing in the string decides
        it. ExpDate says the latter; this parser commits to day-first, which
        costs ~5% of real dates. The PATCH correction endpoint is the remedy.
        """
        assert extract_label_fields(label_result("EXP 21.12.22"))["expiration_date"] == date(2022, 12, 21)


class TestLotNumbers:
    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("LOT ABC123", "ABC123"),
            ("Lot: ABC123", "ABC123"),
            ("BATCH 55231", "55231"),
            ("Lot Nr .: 6714", "6714"),
            ("Lot No. A12", "A12"),
            ("Batch Number: X99-7", "X99-7"),
        ],
    )
    def test_lot_markers_used_on_real_packaging(self, line, expected):
        assert extract_label_fields(label_result("Some Product", line))["lot_number"] == expected

    def test_absent_when_there_is_no_marker(self):
        """A bare code with no lot/batch word is deliberately not guessed at —
        labels are covered in codes that aren't lot numbers.
        """
        fields = extract_label_fields(label_result("Some Product", "L22103127228011"))

        assert fields["lot_number"] is None


class TestExpiryLookahead:
    """OCR line order is a poor proxy for physical proximity on the package.
    On a real Lindt label the keyword was on line 37 and its date on line 41,
    with a barcode and two lines of Chinese between them.
    """

    def test_date_several_lines_after_the_keyword_is_found(self):
        raw = label_result(
            "LINDT & SPRUNGLI",
            "Best Before",
            "此日期前最佳",
            "(y年/m月/d日)",
            "4 894475 100176",
            "2021. 10. 31",
        )

        assert extract_label_fields(raw)["expiration_date"] == date(2021, 10, 31)

    def test_separator_may_carry_whitespace(self):
        """'2021. 10. 31' is how OCR renders a spaced-out printed date."""
        assert extract_label_fields(label_result("EXP", "2021. 10. 31"))["expiration_date"] == date(2021, 10, 31)

    def test_a_date_far_beyond_the_window_is_not_claimed(self):
        """The window is bounded on purpose. A wrong expiry is worse than a
        blank one — it drives the `expired` match status on reconciliation.
        """
        raw = label_result("Best Before", *["filler"] * 8, "01/01/2030")

        assert extract_label_fields(raw)["expiration_date"] is None


def sized_result(*lines, page_height=1000, paragraphs=None):
    """A response carrying line geometry. Each entry is (content, height in px)."""
    page_lines = []
    top = 0
    for content, height in lines:
        page_lines.append(
            {"content": content, "polygon": [0, top, 100, top, 100, top + height, 0, top + height]}
        )
        top += height + 5
    result = {"pages": [{"height": page_height, "width": 800, "lines": page_lines}]}
    if paragraphs is not None:
        result["paragraphs"] = paragraphs
    return result


class TestProductNameGuess:
    """Measured against 25 real packaging photos: reading-order ("first line")
    was obviously wrong 32% of the time. Largest printed text, filtered for
    things that are knowably not a name, brings that to 8%.
    """

    def test_largest_text_wins_over_reading_order(self):
        """A product name is what the package prints biggest — not whatever
        the OCR happened to read first.
        """
        raw = sized_result(("BEST BEFORE 11/11/21", 20), ("LEMONADE", 70), ("No Brand", 22))

        assert extract_label_fields(raw)["product_name"] == "LEMONADE"

    def test_ingredient_prose_is_not_a_name(self):
        """Back-of-pack small print is all set at one size, so without a length
        limit an ingredient list outranks the brand purely by line count.
        """
        raw = sized_result(
            ("Ingredients: sugar, cocoa butter, vegetable fat, whole milk powder, lactose", 32),
            ("LINDT", 30),
        )

        assert extract_label_fields(raw)["product_name"] == "LINDT"

    @pytest.mark.parametrize("noise", ["4 894475 100176", "1kg", "EXP 25/08/2023", "Lot No. A12", "RY"])
    def test_things_that_are_knowably_not_a_name(self, noise):
        raw = sized_result((noise, 90), ("Organic Tomato Sauce", 20))

        assert extract_label_fields(raw)["product_name"] == "Organic Tomato Sauce"

    def test_a_title_role_is_checked_not_trusted(self):
        """Regression: on a real label prebuilt-layout tagged the expiry line
        '유통기한 2021.08.08 까지' as the document title. Taking that unchecked
        beat a product name printed four times larger.
        """
        raw = sized_result(
            ("유통기한 2021.08.08 까지", 32),
            ("고구마 말랭이", 130),
            paragraphs=[{"role": "title", "content": "유통기한 2021.08.08 까지"}],
        )

        assert extract_label_fields(raw)["product_name"] == "고구마 말랭이"

    def test_a_plausible_title_role_is_still_preferred(self):
        raw = sized_result(
            ("SOMETHING HUGE", 200),
            paragraphs=[{"role": "title", "content": "Organic Tomato Sauce"}],
        )

        assert extract_label_fields(raw)["product_name"] == "Organic Tomato Sauce"

    def test_none_when_nothing_on_the_label_could_be_a_name(self):
        """Some photos catch only the date stamp. Saying so beats handing back
        'Use By: 26.11.2022', which the scan route would turn into a catalogue
        Product of that name.
        """
        raw = sized_result(("Use By: 26.11.2022", 40))

        assert extract_label_fields(raw)["product_name"] is None

    def test_falls_back_to_reading_order_without_geometry(self):
        """Responses with no polygons still have to work."""
        raw = label_result("EXP 01/01/2030", "ACME Tomato Sauce 400g")

        assert extract_label_fields(raw)["product_name"] == "ACME Tomato Sauce 400g"


class TestGreekLabels:
    """Greek is the second supported language. The terms are the EU-standard
    phrasings (Reg. 1169/2011) and have NOT been checked against real Greek
    packaging — the same footing the English list started on.
    """

    @pytest.mark.parametrize(
        "line",
        [
            "ΑΝΑΛΩΣΗ ΚΑΤΑ ΠΡΟΤΙΜΗΣΗ ΠΡΙΝ ΑΠΟ 25/08/2026",
            "ΑΝΑΛΩΣΗ ΕΩΣ 25.08.2026",
            "ΗΜΕΡΟΜΗΝΙΑ ΛΗΞΗΣ: 25-08-2026",
            "ΛΗΞΗ 25 08 2026",
            "ΗΜ. ΛΗΞΗΣ 25/08/2026",
        ],
    )
    def test_uppercase_greek_expiry_terms(self, line):
        assert extract_label_fields(label_result("Προϊόν", line))["expiration_date"] == date(2026, 8, 25)

    @pytest.mark.parametrize(
        "line",
        [
            "ανάλωση κατά προτίμηση πριν από 25/08/2026",
            "λήξη: 25/08/2026",
            "Ημερομηνία λήξης 25/08/2026",
        ],
    )
    def test_accented_lower_case_greek(self, line):
        """re.IGNORECASE handles Greek case and the final sigma on its own, but
        not accents: ΑΝΑΛΩΣΗ does not match ανάλωση. _fold() closes that.
        """
        assert extract_label_fields(label_result("Προϊόν", line))["expiration_date"] == date(2026, 8, 25)

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("ΠΑΡΤΙΔΑ: A1234", "A1234"),
            ("ΑΡ. ΠΑΡΤΙΔΑΣ: 5512", "5512"),
            ("ΠΑΡΤΙΔΑ ΑΡ. 5512", "5512"),
            ("ΠΑΡΤΙΔΑ ΑΡΙΘ. 5512", "5512"),
            ("ΠΑΡΤ. B99", "B99"),
            ("παρτίδα: A1234", "A1234"),
            # Greek nu + omicron, not the Latin letters. Identical to the eye.
            ("Παρτίδα Νο 7788", "7788"),
        ],
    )
    def test_greek_lot_markers(self, line, expected):
        assert extract_label_fields(label_result("Προϊόν", line))["lot_number"] == expected

    def test_the_lot_value_keeps_its_original_characters(self):
        """Matching happens on accent-folded text but the value is sliced out
        of the original, so folding can never alter what is stored.
        """
        assert extract_label_fields(label_result("ΠΑΡΤΙΔΑ: Aé12"))["lot_number"] == "Aé12"

    @pytest.mark.parametrize(
        "line",
        ["Γάλα πλήρες 1L", "Ελληνικό γιαούρτι", "Συστατικά: γάλα, αλάτι", "Διατηρείται στο ψυγείο"],
    )
    def test_ordinary_greek_text_is_not_mistaken_for_a_field(self, line):
        fields = extract_label_fields(label_result(line))

        assert fields["expiration_date"] is None
        assert fields["lot_number"] is None

    def test_english_still_works_alongside_greek(self):
        fields = extract_label_fields(label_result("ACME Sauce", "BEST BEFORE 11/11/21", "Lot Nr .: 6714"))

        assert fields["expiration_date"] == date(2021, 11, 11)
        assert fields["lot_number"] == "6714"


class TestMonthYearDates:
    """EU labelling allows month-and-year with no day for anything keeping
    longer than three months, so it is the normal form on dry goods rather
    than an edge case.
    """

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("ΑΝΑΛΩΣΗ ΚΑΤΑ ΠΡΟΤΙΜΗΣΗ ΠΡΙΝ ΑΠΟ ΤΟ ΤΕΛΟΣ 08/2026", date(2026, 8, 31)),
            ("BEST BEFORE END 08/2026", date(2026, 8, 31)),
            ("ΛΗΞΗ 08/2026", date(2026, 8, 31)),
            ("EXP 12-2025", date(2025, 12, 31)),
            ("BEST BEFORE END: 02.2024", date(2024, 2, 29)),
        ],
    )
    def test_resolves_to_the_last_day_of_the_month(self, line, expected):
        """"Best before end of August" means good through the 31st. Taking the
        1st would flag a month of good stock as expired, since reconciliation
        marks a label expired on `expiration_date <= today`.
        """
        assert extract_label_fields(label_result("Product", line))["expiration_date"] == expected

    def test_a_full_date_still_wins(self):
        assert extract_label_fields(label_result("EXP 25/08/2026"))["expiration_date"] == date(2026, 8, 25)

    @pytest.mark.parametrize("line", ["EXP 1/2", "EXP 13/2026", "EXP NET 1/2 kg"])
    def test_not_every_pair_of_numbers_is_a_month_and_year(self, line):
        assert extract_label_fields(label_result("Product", line))["expiration_date"] is None
