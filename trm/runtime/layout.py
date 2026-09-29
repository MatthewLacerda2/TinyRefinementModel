"""How often a run writes, and where on disk it puts what it writes.

These constants are read by the jax-heavy trainer and checkpoint code and also by
tools that must stay light: `trm.runtime.launch` and `trm.runtime.rewind` run beside
a training job and must never import jax. Each tool used to keep its own copy, with
a test holding the copies together (#318). This module is that single copy, and it
imports nothing, so every one of them can read it. The cadences and alarm bars a run
can set are knobs, fields of trm.settings.Config (#475).
"""

# A metrics.csv row is written every LOG_REAL_STEPS opt steps. A validation probe's CE
# is held and written on the first such row at or after the probe, not on the probe's
# own opt step, so a reader aligning val CE with a checkpoint looks in the window
# [probe, probe + LOG_REAL_STEPS). Not recorded in run metadata (#351).
LOG_REAL_STEPS = 5

# The items every checkpoint step directory holds, in orbax's item order.
CHECKPOINT_ITEMS = ("model", "optimizer", "monitor_state", "step")
# What a milestone step directory holds: the same, without the optimizer (#394).
MILESTONE_ITEMS = ("model", "monitor_state", "step")
# Rolling-latest and best checkpoints each keep this many, newest first.
ROLLING_KEEP = 3

# Sibling subdir of the rolling-latest checkpoints holding the best held-out-CE
# checkpoints. Kept separate so best-retention never evicts the latest. Named for
# its criterion (#222): the old `best/` was selected on noisy train CE and went
# stale on two runs, and a new name keeps those archives from passing for these.
BEST_SUBDIR = "best_val_ce"

# Sibling subdir holding milestone checkpoints, which nothing evicts (#187).
MILESTONE_SUBDIR = "milestones"

# The run's yardstick journal, one JSON line per scoring pass, appended by
# instruments.base_run and read by the supervisor to know what is already scored (#471).
YARDSTICK_JOURNAL = "yardstick.jsonl"
