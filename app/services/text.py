"""Text normalisation shared by the OCR parsers and the fuzzy matchers."""

import unicodedata

from rapidfuzz.utils import default_process


def fold_accents(text: str) -> str:
    """Strip accents, one character in, one character out.

    Folded per character rather than with a whole-string NFD pass so the result
    is always the same length as the input. That guarantees a span found in the
    folded text can be sliced straight out of the original — which is how
    document_intelligence recognises a Greek keyword without altering the lot
    number it returns.
    """
    return "".join(unicodedata.normalize("NFD", character)[:1] or character for character in text)


def match_key(text: str) -> str:
    """Processor for every rapidfuzz comparison in this app.

    rapidfuzz's `default_process` lowercases, strips punctuation and collapses
    whitespace, but leaves accents alone — and Greek drops its accents when set
    in capitals, which is how invoices are typically printed. So "Γάλα πλήρες"
    on a label and "ΓΑΛΑ ΠΛΗΡΕΣ" on the invoice for the same product scored
    only 78.6 and fell below the match threshold.

    Measured over 921 real Greek products (Open Food Facts): 27% of accented
    names failed to match their own all-caps rendering. With folding, none do.

    Final sigma is normalised for the same reason. Greek writes sigma as ς at
    the end of a word and σ elsewhere, but capital Σ lowercases to σ in both
    positions — so "πλήρες" and "ΠΛΗΡΕΣ" still differ by one character after
    everything else has been normalised. Mapping ς to σ closes that.
    """
    return default_process(fold_accents(text or "")).replace("ς", "σ")
