from datetime import date

from rapidfuzz import fuzz

from app.metrics import label_match_score, label_matches
from app.services.text import match_key

# Scorer and threshold calibrated against 921 real Greek products from Open
# Food Facts (ODbL), as ~3,000 synthetic orders of 12 line items: a label
# rendered the way an invoice or a label photo would render it, against one
# order holding the product plus same-brand distractors.
#
#                            right item   wrong item   no match   false match
#                                                                 (product not
#                                                                  on the order)
#   token_set_ratio @ 70          94.5%         5.4%       0.1%         38.0%
#   WRatio          @ 88          96.3%         2.1%       1.6%          8.3%
#
# token_set_ratio was the wrong tool, not merely badly tuned. It compares the
# token *intersection* against each side and takes the best, so whenever one
# name's tokens are a subset of the other's it returns 100 — "Pepsi" scores
# 100 against "Pepsi Max", and a label reading "Heinz Tomato Ketchup 342g"
# picked "Heinz Ketchup Heinz" over the correct "Heinz Tomato Ketchup 342 g".
# No threshold fixes that: raising it from 70 to 90 left wrong picks at 5.1%
# while pushing misses to 3.7%.
#
# 88 sits in a deliberate window. WRatio scales partial matches by 0.9, so
# anything matched that way caps at 90; below ~88 almost everything matches
# (72% false match at 70), above 90 genuine matches start failing. 88-90 is
# the usable band and the numbers barely move across it.
#
# The trade is 1.5pp more misses for a third of the wrong picks and a fifth of
# the false matches. A miss surfaces as `unmatched` for a human to resolve; a
# wrong pick silently reports an order as reconciled when it isn't.

MATCH_THRESHOLD = 88
RESOLVED_STATUSES = {"matched", "rejected", "accepted_expired"}

def match_label(label) -> None:
    """Fuzzy-match a scanned label's product name against its order's line
    items, and flag expired products. Mutates `label` in place; caller commits.

    Runs on both the initial label scan and any later manual correction (PATCH
    /orders/<id>/labels/<id>) — instrumented here rather than at each call site
    so the match-rate metric reflects every attempt regardless of trigger.
    """
    if label.order is None or not label.order.items:
        label.match_status = "no_order"
        label.matched_order_item_id = None
        label.match_score = None
        label_matches.add(1, {"status": "no_order"})
        return

    best_item = None
    best_score = 0.0
    for item in label.order.items:
        score = fuzz.WRatio(label.product_name or "", item.product_name or "", processor=match_key)
        if score > best_score:
            best_score = score
            best_item = item

    label.match_score = best_score
    label_match_score.record(best_score)

    if best_item is None or best_score < MATCH_THRESHOLD:
        label.match_status = "unmatched"
        label.matched_order_item_id = None
        label_matches.add(1, {"status": "unmatched"})
        return

    label.matched_order_item_id = best_item.id

    if label.expiration_date and label.expiration_date <= date.today():
        label.match_status = "expired"
    else:
        label.match_status = "matched"
    label_matches.add(1, {"status": label.match_status})


def reconciliation_summary(order) -> dict:
    matched_item_ids = {label.matched_order_item_id for label in order.labels if label.matched_order_item_id}
    missing_items = [item for item in order.items if item.id not in matched_item_ids]
    problem_labels = [label for label in order.labels if label.match_status not in RESOLVED_STATUSES]

    return {
        "total_items": len(order.items),
        "matched_items": len(matched_item_ids),
        "missing_items": [item.to_dict() for item in missing_items],
        "problem_labels": [label.to_dict() for label in problem_labels],
    }


def order_status(order, summary: dict) -> dict:
    """Single human-facing status for an order, derived from its reconciliation
    summary. Shared by the orders list and order detail views so the same
    order never shows two different badges in different places.
    """
    if summary["total_items"] == 0:
        return {"label": "No items detected", "tone": "neutral"}
    if not order.labels:
        return {"label": "Awaiting labels", "tone": "neutral"}
    if any(label["match_status"] == "expired" for label in summary["problem_labels"]):
        return {"label": "Expired label", "tone": "danger"}
    if summary["missing_items"] or summary["problem_labels"]:
        return {"label": "Needs attention", "tone": "warning"}
    return {"label": "Reconciled", "tone": "success"}
