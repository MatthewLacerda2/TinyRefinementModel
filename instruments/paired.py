"""Paired comparison: is one configuration really better than another? (#230)

Every depth result this project has came from a throwaway script. That is not a
discipline failure so much as a tooling one -- the reusable half was never
extracted, so each investigation rewrote it, and one of those rewrites got the
statistics wrong in a way that took a day to notice.

**The mistake worth encoding against.** Comparing one set of weights at two depths
on the same tokens is a PAIRED comparison. The noise floor from #17 -- seed-to-seed
variance across training runs, sigma ~ 0.03 nats -- is the wrong bar for it by about
two orders of magnitude, because it measures a different source of variation
entirely: training two models, not scoring one model twice. Reaching for it made a
real +0.0013 nat effect look like noise. The right test is the per-position
difference, whose standard error this module computes from the differences
themselves.

**Contrast is required, not optional.** `compare` takes at least two corpora and
refuses to run on one. Every real conclusion in the depth work came from a contrast
(code against prose); every wrong turn came from a pooled number with nothing to
compare it against. Making that structural is cheaper than remembering it.

    python -m instruments.paired --checkpoint runs/<run>/checkpoints \\
        --treatment-depth 8 --control-depth 1

The checkpoint path is the MANAGER ROOT -- the directory holding numerically-named
step directories -- not a step directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

from instruments import results as result_lines
from instruments._common import add_checkpoint_argument, load_env
from trm.config import MAX_SEQ_LEN

# ARCH-SPECIFIC: refiner/reasoner — it compares one model at two depths, and plain has no depth dial (#317).

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {
    "mean, se, t (RESULT)": ("sampled", "paired per-token CE difference over --rows rows; se is across those rows, not across seeds"),
}


@dataclasses.dataclass(frozen=True)
class Paired:
    """The difference between two arms, measured position by position.

    `mean` is in nats and signed so that POSITIVE means the treatment is better
    (lower cross-entropy) -- stated because a sign convention nobody wrote down is
    a sign convention somebody will misread.

    `nonfinite` counts positions dropped for being NaN or inf rather than folded
    into the mean. A checkpoint 3.20B tokens into the base run returns all-NaN
    logits on ~10% of code documents (#229), and one out-of-vocab id poisons a whole
    window (#233); a mean that swallows either is how a bad number becomes a
    finding.
    """

    corpus: str
    mean: float
    se: float
    n: int
    nonfinite: int

    @property
    def t(self) -> float:
        """mean / standard error. |t| >= 2 is the usual two-sigma reading.

        Zero variance is not zero signal. A difference that is IDENTICAL at every
        position is perfectly consistent -- maximally distinguishable, not
        undistinguishable -- so se == 0 with a non-zero mean is infinite t, not
        zero. Returning 0.0 there (the first version of this) would report the
        cleanest possible effect as noise.
        """
        if self.se > 0:
            return self.mean / self.se
        return 0.0 if self.mean == 0 else math.copysign(math.inf, self.mean)

    @property
    def verdict(self) -> str:
        if self.n < 2 or abs(self.t) < 2.0:
            return "indistinguishable"
        return "treatment better" if self.mean > 0 else "control better"

    def __str__(self) -> str:
        return (f"{self.corpus:<24} {self.mean:+.6f} nats  se {self.se:.6f}  "
                f"t {self.t:+7.1f}  n {self.n:,}"
                + (f"  [{self.nonfinite:,} non-finite dropped]" if self.nonfinite else "")
                + f"  -> {self.verdict}")


class _Accumulator:
    """Streaming sum and sum-of-squares, so a corpus never has to fit in memory.

    Deliberately not `np.mean` over a retained array: a real sweep is millions of
    positions, and holding them only to average once is the kind of cost that makes
    an instrument too expensive to reach for.
    """

    def __init__(self):
        self.n = self.s1 = self.s2 = 0
        self.nonfinite = 0

    def add(self, diffs: np.ndarray) -> None:
        finite = np.isfinite(diffs)
        self.nonfinite += int((~finite).sum())
        d = diffs[finite].astype(np.float64)
        self.n += d.size
        self.s1 += float(d.sum())
        self.s2 += float((d * d).sum())

    def result(self, corpus: str) -> Paired:
        if self.n < 2:
            return Paired(corpus, 0.0, 0.0, self.n, self.nonfinite)
        mean = self.s1 / self.n
        # Sample variance, n-1: with n in the millions the correction is
        # irrelevant, but a formula that is only right at large n is a trap for
        # whoever next points this at a small probe set.
        var = max((self.s2 - self.n * mean * mean) / (self.n - 1), 0.0)
        return Paired(corpus, mean, math.sqrt(var / self.n), self.n, self.nonfinite)


def ce_per_position(model, tokens, depth, pad_token_id):
    """Cross-entropy at every position that predicts a real token.

    Position i's logits predict token i+1, so the last position has no target and
    the first token has no prediction. Returned masked rather than pre-averaged
    because the whole point of this module is to look at the distribution.
    """
    logits = model(tokens, depth=depth, training=False, new_document=True).logits
    logits = jnp.asarray(logits[:, :-1, :], jnp.float32)
    targets = tokens[:, 1:]
    ce = -jax.nn.log_softmax(logits, axis=-1)
    ce = jnp.take_along_axis(ce, targets[..., None], axis=-1)[..., 0]
    return np.asarray(ce), np.asarray(targets != pad_token_id)


def compare(model, corpora, *, treatment_depth, control_depth, pad_token_id):
    """Paired difference between two depths, on each corpus.

    `corpora` maps a name to an iterable of token rows. **At least two are
    required** -- see the module docstring.
    """
    if len(corpora) < 2:
        raise ValueError(
            f"compare needs at least two corpora, got {list(corpora)}. A single "
            f"pooled number has nothing to be compared against, and every wrong "
            f"turn in the depth investigation came from one. Pass a control corpus "
            f"(prose against code is the contrast that has actually paid).")

    out = {}
    for name, rows in corpora.items():
        acc = _Accumulator()
        for row in rows:
            toks = jnp.asarray(np.asarray(row)[:, :MAX_SEQ_LEN])
            ce_t, mask = ce_per_position(model, toks, treatment_depth, pad_token_id)
            ce_c, _ = ce_per_position(model, toks, control_depth, pad_token_id)
            # control minus treatment, so positive = treatment better (lower CE)
            acc.add((ce_c - ce_t)[mask])
        out[name] = acc.result(name)
    return out


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_checkpoint_argument(ap, required=True, aliases=("--checkpoint",))
    ap.add_argument("--treatment-depth", type=int, default=8)
    ap.add_argument("--control-depth", type=int, default=1)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--corpora", default="pretrain/codeparrot,pretrain/fineweb-edu",
                    help="comma-separated; at least two")
    ap.add_argument("--seed", type=int, default=0,
                    help="selects a DIFFERENT held-out document sample, by offsetting "
                         "the skip. The model and the tokens are deterministic, so the "
                         "only honest source of variation across repeats is which "
                         "documents get scored — that is what a spec's seeds must vary "
                         "here, and it is what its sigma then means.")
    ap.add_argument("--skip", type=int, default=None,
                    help="rows to skip past trained-through data (default: the trainer's "
                         "VAL_SKIP_SAMPLES). Smaller corpora need a smaller value "
                         "(finemath has 19 chunks to the others' 30 and runs out before "
                         "the default)")
    args = ap.parse_args(argv)
    from trm.config import MODEL_ARCH
    if MODEL_ARCH == "plain":
        raise SystemExit("instruments.paired compares one model at two depths; MODEL_ARCH='plain' "
                         "ignores depth, so both arms would score the same forward pass. "
                         "Load a looped checkpoint with MODEL_ARCH=refiner or reasoner.")

    load_env()
    from trm.runtime.restore import load_eval_batches, restore_model
    from trm.train.validation import VAL_SKIP_SAMPLES

    model, _ = restore_model(args.checkpoint_path)
    names = [c.strip() for c in args.corpora.split(",") if c.strip()]
    # Each seed walks a disjoint block of documents: rows*2 apart, so two seeds
    # cannot overlap even at the largest --rows this is run with.
    skip = (VAL_SKIP_SAMPLES if args.skip is None else args.skip) + args.seed * args.rows * 2
    corpora = {n: load_eval_batches(n, num_rows=args.rows, skip=skip) for n in names}

    print(f"paired: depth {args.treatment_depth} (treatment) vs depth "
          f"{args.control_depth} (control); positive = treatment better\n")
    for res in compare(model, corpora,
                       treatment_depth=args.treatment_depth,
                       control_depth=args.control_depth,
                       pad_token_id=model.pad_token_id).values():
        print(res)
        result_lines.emit(res.corpus, mean=res.mean, se=res.se, t=res.t, n=res.n)


if __name__ == "__main__":
    _main()
