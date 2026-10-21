---
name: training-runs
description: Launching, watching, resuming and storing a GPU training run on this machine — the base run above all — plus the pre-launch base-gate checklist, the model card, the cold tier, and the OOM and disk traps of a 6GB card. Use before any launch, when a run dies or OOMs, when the disk is tight, or when a run finishes and its weights need a home.
---

# Runs on the card

There is one RTX 2060 and one run at a time. A base run is the one job that waits for
the owner (≥48h, CLAUDE.md): surface it with its budget and wait. A base run is at
most **10 days** (owner, 2026-10-16), so the corpus and the budget are sized from the
measured seconds per opt step, never from an old target.

## Before a base run: the base-gate checklist

Something lands before the run when both answers are yes: *can it be judged before the
run?* and *does its benefit need the run to have been trained or logged with it?* That
covers anything that makes the run lighter or faster, anything that changes what the
weights learn, and every piece of telemetry, because a log the run didn't write can't
be recovered from its checkpoints. What the finished weights can use at any time (a KV
cache, an eval, a plot of existing logs) doesn't gate. Apply the `base-gate` label when
an issue passes the test. `gh issue list --label base-gate` is the launch checklist:
empty, or the leftovers waived by the owner by name.

## Launch

```
make launch SPEC=experiments/base/specs/<id>.toml BUDGET=4e9 [ISSUE=N]
```

It refuses without a committed spec or a budget, runs `make test-gpu`, and then the
supervisor runs the real trainer for ~5 minutes and refuses a config that doesn't
survive an apply, a probe and a checkpoint (#168). It derives the stop step, the
checkpoint path and the run directory itself (`trm/runtime/launch.py`). Never assemble
supervisor flags or a `systemd-run` line by hand.

- `make launch` and `make pair` detach at `oom_score_adj` 100. Anything started by
  hand from a Claude shell inherits Claude Code's 200 and becomes the kernel's first
  OOM victim.
- **Never as a Claude background task**: Claude Code kills its background tasks under
  low RAM, and the trainer dies with them.
- While a run lives: **don't pull the root checkout** (the supervisor relaunches a
  crash from root), and no session runs anything that builds or compiles a JAX model
  locally. Lint and pure-Python tests only; the rest goes to CI.
- Before launching: `df -h .`. A checkpoint write is what fails on a full disk.

## While it runs

The supervisor owns the trainer. It stops at the budget, kills on divergence or a
wedged process (alive but not advancing), guards the disk, relaunches crashes from a
retry budget, treats OOM as terminal, and heartbeats into the run's issue
(`trm/runtime/supervisor.py`). `make report RUN=run_…` gives the terminal view and
figures. Whether the model is still *learning* is the spec's `[stall]` rule, read by
`python -m instruments.stall --spec … --run runs/<run>`. A single stalled window is
noise; only a fired rule is news.

**Proof of life** is decreasing CE plus a completed (non `-tmp`) checkpoint directory,
not just finite loss.

**After a power cut or reboot:** `make resume` (`trm-base-run-resume.service` does it
at boot, #384). The resumed trainer continues from its newest checkpoint with an
*estimated* data position: fine for a base run, never for a pair arm.

## When it OOMs

- A deterministic run that OOMs at step N OOMs at step N again. Change a knob before
  relaunching.
- Read VRAM from the allocator (`memory_stats()` peak vs `bytes_limit`, 4883 MiB here;
  `make gate`, `instruments.mem_profile`). Under `cuda_async`, nvidia-smi shows the
  reserved pool (~5003 MiB) whatever the model.
- A BFC free-memory map like `***___*___` is fragmentation (enough free, nothing
  contiguous). `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` is the default
  (`trm/train/start.py`); `TF_GPU_ALLOCATOR` is silently ignored by JAX. Raising
  `MEM_FRACTION` only moves the OOM.
- A CUDA-graph instantiation OOM lives in driver memory outside the arena:
  `XLA_FLAGS=--xla_gpu_enable_command_buffer=` turns graphs off.

## When it finishes: where the weights go

- **The card**: a champion or a notable challenger gets one, from
  `docs/registry/MODEL_CARD_TEMPLATE.md` (`python -m instruments.base_run card` fills
  it for a base run). Rules in `docs/registry/README.md`.
- **The cold tier**: `/mnt/d_drive/TRM_cold/` is blob storage. Copy files there; don't
  train off it. A run launched with `COLD_ROOT` mirrors its milestones and periodic
  full-state copies itself and prunes mirrored milestones from the SSD
  (`trm/runtime/cold.py`). Claude may archive and delete weights it judges no longer
  needed without asking.
- **To free disk**, clear regenerable caches (`~/.cache/pip`, `~/.cache/uv`) and
  Claude's own dead runs. Never the owner's run dirs, and never the tokenized corpus.
