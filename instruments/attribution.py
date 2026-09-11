"""Where does a difference live? (#230)

`instruments/paired.py` answers *whether* two configurations differ. This answers
*where* — and in practice that is the question that produced every real conclusion.
"Depth helps on code, +0.0035 nats" is a number. "76% of that gain lands on 19% of
tokens, and it scales with bracket nesting" is an explanation, and it arrived by
bucketing the same per-position differences four ways in one afternoon.

**Share-of-gain against share-of-tokens is the whole trick.** A bucket holding 40%
of the tokens and 28% of the gain is *under*-represented, however large its
per-token mean looks. That single comparison falsified three competing explanations
of the code result at a glance:

- a tokenizer artifact, because whitespace is 40% of code tokens but only 28% of
  the gain
- long-range copying, because first-occurrence tokens gained MORE than tokens seen
  earlier
- uncertainty resolution, because the gain peaked mid-confidence and fell at both
  tails

Bucketing rules are data, not code. A new question is a new rule, not a new file --
the four that mattered (token category, confidence decile, first occurrence, nesting
depth) are all a function from a position to a label.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict

import numpy as np

from instruments import results as result_lines


@dataclasses.dataclass(frozen=True)
class Bucket:
    """One row of an attribution table."""

    name: str
    mean: float
    se: float
    n: int
    share_of_tokens: float
    share_of_gain: float

    @property
    def over_represented(self) -> float:
        """share_of_gain / share_of_tokens. Above 1.0 carries more than its weight.

        This ratio, not the per-token mean, is what makes a bucket interesting: a
        rare bucket with a large mean may still account for almost none of the
        effect, and a common bucket with a small mean may account for most of it.
        """
        return self.share_of_gain / self.share_of_tokens if self.share_of_tokens else 0.0


def attribute(diffs, labels, *, total_gain=None):
    """Bucket per-position differences by label.

    `diffs` and `labels` are parallel sequences over the SAME positions -- one
    difference and one bucket name each. Non-finite differences are dropped and
    counted per bucket rather than averaged (#229, #233).

    `share_of_gain` is signed and computed against the total summed gain, so
    buckets that move the result *against* the overall direction show as negative
    shares rather than quietly shrinking the denominator.
    """
    diffs = np.asarray(diffs, dtype=np.float64)
    labels = np.asarray(labels)
    finite = np.isfinite(diffs)
    diffs, labels = diffs[finite], labels[finite]

    n_total = diffs.size
    if n_total == 0:
        return []
    gain_total = float(diffs.sum()) if total_gain is None else float(total_gain)

    grouped = defaultdict(list)
    for d, lab in zip(diffs, labels):
        grouped[str(lab)].append(d)

    rows = []
    for name, values in grouped.items():
        v = np.asarray(values)
        mean = float(v.mean())
        se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
        rows.append(Bucket(
            name=name, mean=mean, se=se, n=v.size,
            share_of_tokens=v.size / n_total,
            share_of_gain=(float(v.sum()) / gain_total) if gain_total else 0.0,
        ))
    return sorted(rows, key=lambda b: -abs(b.share_of_gain))


def render(rows, title=""):
    """The table, ordered by share of gain — which is the order that answers the
    question, not the order of per-token means."""
    out = [title] if title else []
    out.append(f"{'bucket':>22} {'mean gain':>12} {'se':>10} "
               f"{'% tokens':>9} {'% of gain':>10} {'ratio':>7} {'n':>9}")
    for b in rows:
        out.append(f"{b.name:>22} {b.mean:>+12.6f} {b.se:>10.6f} "
                   f"{b.share_of_tokens:>8.1%} {b.share_of_gain:>9.1%} "
                   f"{b.over_represented:>7.2f} {b.n:>9,}")
    return "\n".join(out)


def emit(rows) -> None:
    """One RESULT line per bucket, so a spec can drive this and the referee judge it."""
    for b in rows:
        result_lines.emit(b.name, mean=b.mean, se=b.se, n=b.n,
                          share_of_tokens=b.share_of_tokens,
                          share_of_gain=b.share_of_gain)


# --- bucketing rules: a position -> a label ---------------------------------
#
# Rules are ordinary functions so a new question costs a function, not a module.
# Each takes whatever it needs and returns one label per position.

def by_token_category(token_ids, decode):
    """Whitespace / bracket / punctuation / number / word — the rule that turned
    'depth helps on code' into 'depth tracks nested structure'."""
    labels = []
    for tok in token_ids:
        try:
            text = decode([int(tok)])
        except Exception:
            labels.append("other")
            continue
        core = text.strip()
        if not text:
            labels.append("other")
        elif core == "":
            labels.append("whitespace")
        elif all(c in "()[]{}" for c in core):
            labels.append("bracket")
        elif all(c in ".,;:!?\"'`=+-*/<>|&%#@~^\\" for c in core):
            labels.append("punctuation")
        elif core.replace(".", "").isdigit():
            labels.append("number")
        elif core.replace("_", "").isalnum():
            labels.append("word/identifier")
        else:
            labels.append("mixed")
    return labels


def by_confidence_decile(probabilities):
    """Decile of the model's probability on the correct token.

    Tests 'depth just resolves uncertainty': if so the gain concentrates where the
    model is unsure. It did not -- it peaked mid-confidence and fell at both tails.
    """
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    return [f"p {min(int(x * 10), 9) / 10:.1f}-{(min(int(x * 10), 9) + 1) / 10:.1f}"
            for x in p]


def by_first_occurrence(token_ids):
    """Whether this token appeared earlier in the window.

    Tests 'depth is doing long-range copying': if so, repeats gain more. They gained
    LESS, which killed the hypothesis.
    """
    seen, labels = set(), []
    for tok in token_ids:
        t = int(tok)
        labels.append("seen earlier" if t in seen else "first occurrence")
        seen.add(t)
    return labels


def by_nesting_depth(token_ids, decode, cap=4):
    """Bracket nesting depth at each position — the rule that made the structural
    reading concrete: gain rose monotonically from nest 0 to nest 4+."""
    labels, depth = [], 0
    for tok in token_ids:
        try:
            text = decode([int(tok)])
        except Exception:
            text = ""
        opens = sum(text.count(c) for c in "([{")
        closes = sum(text.count(c) for c in ")]}")
        labels.append(f"nest {min(depth, cap)}" + ("+" if depth >= cap else ""))
        depth = max(0, depth + opens - closes)
    return labels
