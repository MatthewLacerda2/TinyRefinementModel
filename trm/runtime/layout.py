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
# The per-corpus probes (#363) fire on validation steps that are also multiples of
# this, so their readings share val_step. Rarer than the fineweb probe: each corpus
# costs one more probe's worth of forward passes.
VAL_BY_SOURCE_EVERY_OPT_STEPS = int(os.environ.get("VAL_BY_SOURCE_EVERY_OPT_STEPS", 128))
# A metrics.csv row is written every LOG_REAL_STEPS opt steps. A validation probe's CE
# is held and written on the first such row at or after the probe, not on the probe's
# own opt step, so a reader aligning val CE with a checkpoint looks in the window
# [probe, probe + LOG_REAL_STEPS). Not recorded in run metadata (#351).
LOG_REAL_STEPS = 5

# The f16 margins the supervisor watches during a run (#368). Crossing one is an
# alarm, announced and recorded, never a kill: a margin is a warning of the failure
# #235 found only after a 10-day run (the champion finished at 65,120 of f16's
# 65,504), not the failure itself. Whether any of them should stop a run is the
# owner's call.
#   act_max past a quarter of f16's ceiling: a block's output is climbing toward it.
ACT_MAX_ALARM = float(os.environ.get("ACT_MAX_ALARM", 65504 / 4))
#   the loss scale at or below this: the backward overflows with no scaling left (#199).
LOSS_SCALE_FLOOR_ALARM = float(os.environ.get("LOSS_SCALE_FLOOR_ALARM", 4))
#   the applied gradient's zero fraction at the underflow bar (#82's 0.05).
ZERO_GRAD_ALARM = float(os.environ.get("ZERO_GRAD_ALARM", 0.05))
#   arena headroom under this: no room left for the next allocation spike.
VRAM_HEADROOM_ALARM_MIB = float(os.environ.get("VRAM_HEADROOM_ALARM_MIB", 150))

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

# When a milestone is kept: at doubling token counts — 8M, 16M, 32M, … — so disk
# grows with log(run length) instead of with it (#394). The horizon is absolute
# tokens, not a fraction of a budget, because a run that stops on a criterion has
# no budget to take a fraction of. The old fixed 500M cadence would have written
# ~48 GB of full-state saves over a 10B-token run onto an SSD with 55 GB free, and
# kept nothing at all inside a 67M-token pair.
MILESTONE_FIRST_TOKENS = int(os.environ.get("MILESTONE_FIRST_TOKENS", 8_000_000))
MILESTONE_RATIO = float(os.environ.get("MILESTONE_RATIO", 2))
# The cap a runaway run stops at: 16 doublings from 8M is 262B tokens, far past
# anything this card can train.
MILESTONE_MAX_COUNT = int(os.environ.get("MILESTONE_MAX_COUNT", 16))
