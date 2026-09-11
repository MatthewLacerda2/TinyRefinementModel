"""The attribution instrument computes what it claims (#230).

`paired.py` says whether two configurations differ; this says WHERE. The load-bearing
quantity is share-of-gain against share-of-tokens, because that ratio — not the
per-token mean — is what falsified three competing explanations of the code result in
one pass. A bucket holding 40% of tokens and 28% of the gain is under-represented
however large its mean looks.

So the arithmetic is pinned against tables whose answers are known by construction,
and the ordering is pinned too: a table sorted by per-token mean answers a different
question than the one being asked.
"""

import numpy as np
import pytest

from instruments.attribution import (
    attribute, by_confidence_decile, by_first_occurrence, by_token_category, render,
)


def test_shares_are_computed_against_the_totals_not_per_bucket():
    """Two buckets, equal counts, one carrying three times the gain."""
    rows = {b.name: b for b in attribute([1.0, 1.0, 3.0, 3.0], ["a", "a", "b", "b"])}

    assert rows["a"].share_of_tokens == pytest.approx(0.5)
    assert rows["b"].share_of_tokens == pytest.approx(0.5)
    assert rows["a"].share_of_gain == pytest.approx(0.25)
    assert rows["b"].share_of_gain == pytest.approx(0.75)


def test_the_ratio_is_what_marks_a_bucket_as_carrying_the_effect():
    """The whole trick. `common` has the larger total gain; `rare` carries far more
    per token it occupies. Only the ratio distinguishes them."""
    diffs = [1.0] * 90 + [3.0] * 10
    labels = ["common"] * 90 + ["rare"] * 10
    rows = {b.name: b for b in attribute(diffs, labels)}

    assert rows["common"].share_of_gain > rows["rare"].share_of_gain, "more total gain"
    assert rows["rare"].over_represented > rows["common"].over_represented
    assert rows["common"].over_represented < 1.0, "under-represented despite the bigger sum"


def test_a_bucket_moving_against_the_effect_shows_a_negative_share():
    """Buckets that push the other way must show as negative rather than quietly
    shrinking the denominator — word/identifier tokens did exactly this on code, and
    reading them as 'small positive' would have inverted the conclusion."""
    rows = {b.name: b for b in attribute([10.0, 10.0, -5.0], ["up", "up", "down"])}
    assert rows["down"].share_of_gain < 0
    assert rows["up"].share_of_gain > 1.0


def test_rows_are_ordered_by_share_of_gain_not_by_mean():
    """A table sorted by per-token mean answers a different question. `rare` has the
    higher mean; `common` explains more of the result and must lead."""
    diffs = [1.0] * 90 + [5.0] * 2
    labels = ["common"] * 90 + ["rare"] * 2
    assert [b.name for b in attribute(diffs, labels)] == ["common", "rare"]


def test_non_finite_differences_are_dropped_not_averaged():
    rows = {b.name: b for b in attribute([1.0, np.nan, 3.0, np.inf], ["a", "a", "b", "b"])}
    assert rows["a"].n == 1 and rows["b"].n == 1
    assert rows["a"].mean == pytest.approx(1.0)


def test_an_empty_input_returns_nothing_rather_than_dividing_by_zero():
    assert attribute([], []) == []
    assert attribute([np.nan, np.inf], ["a", "b"]) == []


# --- the bucketing rules ------------------------------------------------------

def _decode(ids):
    return {1: "  ", 2: "(", 3: ";", 4: "42", 5: "value", 6: "x1_", 7: "<<"}[ids[0]]


@pytest.mark.parametrize("tok,expected", [
    (1, "whitespace"), (2, "bracket"), (3, "punctuation"),
    (4, "number"), (5, "word/identifier"), (6, "word/identifier"), (7, "punctuation"),
])
def test_token_categories(tok, expected):
    assert by_token_category([tok], _decode) == [expected]


def test_first_occurrence_tracks_what_was_seen_before_it():
    assert by_first_occurrence([7, 7, 8, 7]) == [
        "first occurrence", "seen earlier", "first occurrence", "seen earlier"]


def test_confidence_deciles_cover_the_closed_unit_interval():
    """p = 1.0 must land in the top bucket, not in an eleventh one — an off-by-one
    here silently creates a bucket holding only perfectly-confident positions."""
    labels = by_confidence_decile([0.0, 0.05, 0.55, 0.999, 1.0])
    assert labels[0] == "p 0.0-0.1"
    assert labels[-1] == labels[-2] == "p 0.9-1.0"
    assert len(set(labels)) == 3


def test_the_rendered_table_carries_the_error_and_the_ratio():
    """Doctrine rule 1: report the spread, not just the point estimate."""
    text = render(attribute([1.0, 1.2, 3.0, 3.4], ["a", "a", "b", "b"]), "title")
    assert "se" in text and "% of gain" in text and "ratio" in text
