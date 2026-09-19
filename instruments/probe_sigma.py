"""How noisy is the validation probe itself? (#184)

    python -m instruments.probe_sigma --checkpoint-path runs/run_x/checkpoints [--rows 4 64] [--slices 6]

Scores ONE checkpoint on several disjoint held-out slices of the same size and
reports the spread. That spread is the probe's own sampling noise — what
"no change" looks like between two probe readings — and it is the number the
plateau detector's PLATEAU_MIN_DELTA must sit above. Measured, not assumed:
#17's 0.03 was seed-to-seed training noise on a 4-row probe, a different
quantity.

Slices are taken past VAL_SKIP_SAMPLES, each `rows` further on, so the trainer's
own slice (the first one) is included and the others never overlap it.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import argparse
import statistics

from instruments._common import add_checkpoint_argument, load_env
from trm.config import EVAL_ROWS, resolve_root
from trm.runtime.restore import restore_model
from trm.train.validation import VAL_SKIP_SAMPLES, ValidationProbe

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {
    "val CE per slice": ("sampled", "held-out CE on one disjoint slice of `rows` rows; the trainer's probe is slice 0"),
    "sigma across slices": ("measured", "sample sigma of those readings: the probe's own noise at this width"),
}

# Where this differs from production's environment, and why (#166).
ENV_DIVERGENCES = {"XLA_PYTHON_CLIENT_MEM_FRACTION": "an eval that may share the card with a training run"}

# Off-config defaults, on purpose (tests/apparatus/test_instrument_defaults.py).
CONFIG_DIVERGENCES = {}


def slice_offsets(rows, slices, skip=VAL_SKIP_SAMPLES):
    """Disjoint slices of `rows` rows starting at the trainer's own slice."""
    return [skip + i * rows for i in range(slices)]


def main(argv=None):
    load_env()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_checkpoint_argument(ap)
    ap.add_argument("--rows", type=int, nargs="+", default=[4, EVAL_ROWS], help="probe widths to compare")
    ap.add_argument("--slices", type=int, default=6, help="disjoint slices per width")
    args = ap.parse_args(argv)
    data_root = resolve_root(os.environ.get("DATA_ROOT", "runs/data"))
    model, step = restore_model(args.checkpoint_path)
    print(f"checkpoint step {step}; {args.slices} disjoint slices per width")
    for rows in args.rows:
        readings = []
        for skip in slice_offsets(rows, args.slices):
            ce = ValidationProbe(data_root, rows=rows, skip=skip).run(model)
            if ce is None:
                break
            readings.append(ce)
        sigma = statistics.stdev(readings) if len(readings) > 1 else float("nan")
        print(f"rows={rows:>3}: " + " ".join(f"{r:.4f}" for r in readings)
              + f"  | mean {statistics.mean(readings):.4f}  sigma {sigma:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
