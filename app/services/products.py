from rapidfuzz import fuzz

from app.extensions import db
from app.models import Product
from app.services.text import match_key

# Stricter than reconciliation.py's label-to-order-item threshold (88):
# merging two genuinely different products into one catalog entry corrupts
# per-product history permanently, whereas a missed match just creates an
# extra (mergeable-later) catalog row.
#
# Calibrated as a 1-vs-921 retrieval over real Greek products from Open Food
# Facts (ODbL) — a much harder task than reconciliation's 1-vs-12, which is
# why it keeps a higher bar:
#
#                          right entry   WRONG merge   new entry   held-out
#                                                                   merged
#   token_set_ratio @ 85         88.3%         10.1%        1.6%      22.6%
#   WRatio          @ 90         94.3%          3.6%        2.1%      13.8%
#
# Both error columns matter and both improved. "WRONG merge" is a query
# attached to the wrong existing product; "held-out merged" is a genuinely new
# product swallowed into an existing entry rather than getting its own row.
# Under the old setting nearly a quarter of new products were absorbed that
# way, which is exactly the permanent corruption the paragraph above warns of.
#
# token_set_ratio returns 100 whenever one name's tokens are a subset of the
# other's, so "Caprice 115g" and "Caprice 400gr" — genuinely different SKUs —
# were indistinguishable at any threshold. See reconciliation.py for the full
# reasoning.

MATCH_THRESHOLD = 90


def resolve_product(product_name: str) -> Product:
    """Fuzzy-matches product_name against the existing catalog, creating a
    new Product if nothing matches closely enough.

    Does not commit — caller is expected to be building a larger transaction
    (e.g. an order + its line items) and commit once at the end. Uses
    session.flush() so the returned Product always has a real id, even when
    newly created, so callers can set a FK to it before that commit.
    """
    name = (product_name or "").strip() or "Unknown product"

    best_product = None
    best_score = 0.0
    for product in Product.query.all():
        score = fuzz.WRatio(name, product.canonical_name, processor=match_key)
        if score > best_score:
            best_score = score
            best_product = product

    if best_product is not None and best_score >= MATCH_THRESHOLD:
        return best_product

    new_product = Product(canonical_name=name)
    db.session.add(new_product)
    db.session.flush()
    return new_product
