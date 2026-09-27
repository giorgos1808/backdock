from dataclasses import dataclass, field
from datetime import date, timedelta
import pytest
from app.services.reconciliation import MATCH_THRESHOLD, match_label, order_status, reconciliation_summary


@dataclass
class FakeItem:
    id: int
    product_name: str

    def to_dict(self) -> dict:
        return {"id": self.id, "product_name": self.product_name}


@dataclass
class FakeOrder:
    items: list = field(default_factory=list)
    labels: list = field(default_factory=list)


@dataclass
class FakeLabel:
    product_name: str | None = None
    expiration_date: date | None = None
    order: FakeOrder | None = None
    match_status: str | None = None
    matched_order_item_id: int | None = None
    match_score: float | None = None

    def to_dict(self) -> dict:
        return {"match_status": self.match_status, "product_name": self.product_name}


def make_order(*names: str) -> FakeOrder:
    return FakeOrder(items=[FakeItem(i, name) for i, name in enumerate(names, start=1)])


class TestMatchLabel:
    def test_label_with_no_order_is_flagged_no_order(self):
        label = FakeLabel(product_name="Tomato Sauce 400g", order=None)

        match_label(label)

        assert label.match_status == "no_order"
        assert label.matched_order_item_id is None
        assert label.match_score is None


    def test_order_with_no_line_items_is_flagged_no_order(self):
        label = FakeLabel(product_name="Tomato Sauce 400g", order=FakeOrder(items=[]))

        match_label(label)

        assert label.match_status == "no_order"


    def test_close_name_matches_the_right_line_item(self):
        order = make_order("Olive Oil 1L", "Tomato Sauce 400g")
        label = FakeLabel(product_name="Tomato Sauce 400 g", order=order)

        match_label(label)

        assert label.match_status == "matched"
        assert label.matched_order_item_id == 2
        assert label.match_score >= MATCH_THRESHOLD


    def test_casing_alone_does_not_break_a_match(self):
        """Regression: rapidfuzz is case-sensitive by default, so an all-caps
        invoice line against a mixed-case label used to score below threshold
        for what is plainly the same product. Both call sites pass
        `processor=default_process` to prevent that.
        """
        order = make_order("TOMATO SAUCE 400G")
        label = FakeLabel(product_name="Tomato Sauce 400g", order=order)

        match_label(label)

        assert label.match_status == "matched"
        assert label.match_score == pytest.approx(100.0)


    def test_unrelated_product_is_unmatched(self):
        order = make_order("Tomato Sauce 400g")
        label = FakeLabel(product_name="Dishwasher Tablets", order=order)

        match_label(label)

        assert label.match_status == "unmatched"
        assert label.matched_order_item_id is None
        assert label.match_score < MATCH_THRESHOLD


    def test_matched_but_past_expiry_is_flagged_expired(self):
        order = make_order("Tomato Sauce 400g")
        label = FakeLabel(
            product_name="Tomato Sauce 400g",
            expiration_date=date.today() - timedelta(days=1),
            order=order,
        )

        match_label(label)

        assert label.match_status == "expired"
        assert label.matched_order_item_id == 1


    def test_expiring_today_counts_as_expired(self):
        order = make_order("Tomato Sauce 400g")
        label = FakeLabel(product_name="Tomato Sauce 400g", expiration_date=date.today(), order=order)

        match_label(label)

        assert label.match_status == "expired"


    def test_future_expiry_is_matched(self):
        order = make_order("Tomato Sauce 400g")
        label = FakeLabel(
            product_name="Tomato Sauce 400g",
            expiration_date=date.today() + timedelta(days=365),
            order=order,
        )

        match_label(label)

        assert label.match_status == "matched"


class TestReconciliationSummary:
    def test_counts_matched_items_and_lists_the_rest(self):
        order = make_order("Olive Oil 1L", "Tomato Sauce 400g")
        matched = FakeLabel(product_name="Tomato Sauce 400g", match_status="matched", matched_order_item_id=2)
        stray = FakeLabel(product_name="Something Else", match_status="unmatched", matched_order_item_id=None)
        order.labels = [matched, stray]

        summary = reconciliation_summary(order)

        assert summary["total_items"] == 2
        assert summary["matched_items"] == 1
        assert [item["product_name"] for item in summary["missing_items"]] == ["Olive Oil 1L"]
        assert [label["product_name"] for label in summary["problem_labels"]] == ["Something Else"]


class TestOrderStatus:
    def test_no_items_detected(self):
        order = FakeOrder()

        assert order_status(order, reconciliation_summary(order))["tone"] == "neutral"


    def test_items_but_no_labels_yet(self):
        order = make_order("Tomato Sauce 400g")

        status = order_status(order, reconciliation_summary(order))

        assert status["label"] == "Awaiting labels"


    def test_expired_label_outranks_other_problems(self):
        """An order can be both incomplete and holding expired stock; the
        expired case is the more urgent one and has to win the badge.
        """
        order = make_order("Olive Oil 1L", "Tomato Sauce 400g")
        order.labels = [FakeLabel(product_name="Tomato Sauce 400g", match_status="expired", matched_order_item_id=2)]

        status = order_status(order, reconciliation_summary(order))

        assert status == {"label": "Expired label", "tone": "danger"}


    def test_missing_items_needs_attention(self):
        order = make_order("Olive Oil 1L", "Tomato Sauce 400g")
        order.labels = [FakeLabel(product_name="Tomato Sauce 400g", match_status="matched", matched_order_item_id=2)]

        status = order_status(order, reconciliation_summary(order))

        assert status == {"label": "Needs attention", "tone": "warning"}


    def test_fully_reconciled(self):
        order = make_order("Tomato Sauce 400g")
        order.labels = [FakeLabel(product_name="Tomato Sauce 400g", match_status="matched", matched_order_item_id=1)]

        status = order_status(order, reconciliation_summary(order))

        assert status == {"label": "Reconciled", "tone": "success"}


class TestMatchScoring:
    """The scorer and threshold were calibrated against 921 real Greek products
    from Open Food Facts. These pin the specific failures that calibration
    exposed, so a future "tidy-up" back to token_set_ratio fails loudly.
    """

    def test_a_subset_name_does_not_outrank_the_real_match(self):
        """The regression that motivated changing scorer. token_set_ratio
        compares the token intersection and returns 100 whenever one name's
        tokens are a subset of the other's, so the junk catalogue entry
        'Heinz Ketchup Heinz' scored 100 against this label and beat the
        correct line item, which only scored 98.
        """
        order = make_order("Heinz Ketchup Heinz", "Heinz Tomato Ketchup 342 g")
        label = FakeLabel(product_name="Heinz Tomato Ketchup 342g", order=order)

        match_label(label)

        assert label.match_status == "matched"
        assert label.matched_order_item_id == 2


    def test_accented_greek_matches_its_all_caps_rendering(self):
        """Greek capitals carry no accents, and invoices are typically set in
        capitals. rapidfuzz's default_process does not fold accents, so this
        pair scored 78.6 and fell below threshold — 27% of real accented Greek
        names failed to match themselves this way.
        """
        order = make_order("ΓΑΛΑ ΠΛΗΡΕΣ 1L")
        label = FakeLabel(product_name="Γάλα πλήρες 1L", order=order)

        match_label(label)

        assert label.match_status == "matched"


    def test_a_different_pack_size_is_a_different_product(self):
        """Caprice 115g and Caprice 400gr are separate SKUs. With both on the
        order, the label has to land on its own size.
        """
        order = make_order("Papadopoulos Caprice 400gr", "Papadopoulos Caprice 115g")
        label = FakeLabel(product_name="Papadopoulos Caprice 115 g", order=order)

        match_label(label)

        assert label.matched_order_item_id == 2


    def test_an_unrelated_product_still_does_not_match(self):
        order = make_order("Γάλα πλήρες 1L", "Tomato Sauce 400g")
        label = FakeLabel(product_name="Dishwasher Tablets 30pcs", order=order)

        match_label(label)

        assert label.match_status == "unmatched"
