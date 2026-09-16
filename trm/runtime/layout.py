"""How often a run writes, and where on disk it puts what it writes.

These constants are read by the jax-heavy trainer and checkpoint code and also by
tools that must stay light: `trm.runtime.launch` and `trm.runtime.rewind` run beside
a training job and import nothing but the standard library. Each tool used to keep
its own copy, with a test holding the copies together (#318). This module is that
single copy, and it imports only `os` so every one of them can read it.
"""

import os

# Cadences, in optimizer steps, both firing at the opt-step boundary — NOT nested
# in the logging block (nesting would multiply the interval by LOG_REAL_STEPS, the
# bug that hid the probe). The full-state checkpoint save blocks the loop, so it
# must stay rare.
# Env-overridable for one caller: the supervisor's fit gate (#168) sets both to 1,
# so a few-minute probe crosses a validation pass and a checkpoint write — where
# every observed launch OOM landed — instead of waiting 8,192 micro-steps for them.
VAL_EVERY_OPT_STEPS = int(os.environ.get("VAL_EVERY_OPT_STEPS", 64))
CHECKPOINT_EVERY_OPT_STEPS = int(os.environ.get("CHECKPOINT_EVERY_OPT_STEPS", 64))
# A metrics.csv row is written every LOG_REAL_STEPS opt steps. A validation probe's CE
# is held and written on the first such row at or after the probe, not on the probe's
# own opt step, so a reader aligning val CE with a checkpoint looks in the window
# [probe, probe + LOG_REAL_STEPS). Not recorded in run metadata (#351).
LOG_REAL_STEPS = 5

# The items every checkpoint step directory holds, in orbax's item order.
CHECKPOINT_ITEMS = ("model", "optimizer", "monitor_state", "step")
# Rolling-latest and best checkpoints each keep this many, newest first.
ROLLING_KEEP = 3

# Sibling subdir of the rolling-latest checkpoints holding the best held-out-CE
# checkpoints. Kept separate so best-retention never evicts the latest. Named for
# its criterion (#222): the old `best/` was selected on noisy train CE and went
# stale on two runs, and a new name keeps those archives from passing for these.
BEST_SUBDIR = "best_val_ce"

# Sibling subdir holding milestone checkpoints, which nothing evicts (#187).
MILESTONE_SUBDIR = "milestones"
