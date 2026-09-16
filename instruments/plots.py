"""Figures for a training run: one showable curve, two diagnostic sheets.

    python -m instruments.plots [--log runs/<run>/metrics.csv] [--out DIR]

Three images, split by audience (#179):

  training_curve.png      the hero — CE and held-out val CE against *tokens*,
                          perplexity on the right, and the LR anneal drawn
                          across the whole token budget so "how far in are we"
                          is one glance.
  throughput_progress.png tokens/sec from metrics.csv's wall_clock (#186) — or,
                          for runs older than that column, sampled from the
                          supervisor's heartbeats — progress against the budget, ETA.
  optimization_health.png gradient norm, arena-peak VRAM, zero-grad fraction,
                          logit health — plus sampled depth on the architectures
                          that have one.

Three rules this instrument exists to enforce:

1. **A panel whose data is absent is omitted, not drawn flat.** The previous
   version drew three panels of reasoner-only quantities (temporal drift,
   forget cost, diversity loss) on refiner runs, where those terms do not
   exist (#105) — a flat zero line reads as "we measured zero" when the truth
   is "there is nothing here to measure". The log decides what gets drawn;
   nothing is hardcoded on. See `available()` for the two ways data can be
   missing, and note that what was left out is always reported.
2. **The model is never instantiated.** Counting parameters by building a
   whole network — on the wrong architecture, while the card is busy — is the
   defect this rewrite removes. Parameter counts come analytically from
   `instruments.model_stats`.
3. **Everything the figure says about the run comes from the run** (#305).
   `trm.config` describes the interpreter drawing the picture — its
   environment, its defaults — not the run on disk, and reading the two as one
   is a whole class of bug: a 512-step arm crashed because this process's
   1000-step warmup left the cosine a negative horizon; a `plain` run was
   labelled `depth ≤8` and given the refiner's depth panel; every val CE line
   was annotated "every 64 optimizer steps" whatever the run used. `RunConfig`
   is the one place that resolves this, from the run's own metadata, and it
   says which values it had to fall back on.

Everything here is CPU-only and read-only: it reads a CSV and two log files
and writes PNGs. It is safe to run against a live run.
"""

import os

# This instrument must never touch the card: a training run owns it, and JAX
# would otherwise grab a GPU slice just to evaluate the LR schedule.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import datetime
import math
import pathlib
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from instruments._common import REPO_ROOT
from instruments.runlog import absence_reason, load, recorded_tokens_per_opt_step
from instruments.invariants import clean_column, suspect_rows
# Imported as a module, and used ONLY as RunConfig's fallback for runs that did
# not record a value: every constant in here describes this process (#305).
from trm import config as this_process
from trm.train.schedules import (
    PEAK_LR,
    WARMUP_STEPS,
    build_learning_schedule,
    resolve_decay_steps,
)

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {
    "training curves": ("measured", "metrics.csv as recorded, rows failing an invariant dropped"),
    "throughput (wall_clock)": ("measured", "Δtokens ÷ Δwall-clock between metrics.csv rows at least 30 min apart"),
    "throughput (older runs)": ("sampled", "tokens between supervisor heartbeats, for runs without wall_clock"),
    "arena peak VRAM": ("measured", "peak_bytes_in_use read from the allocator itself, logged on every metrics row"),
}

# The allocator's ceiling on this card, measured rather than guessed — and only the
# fallback: a run that logged its own `arena_limit_mib` is drawn against that. Production
# runs under cuda_async at XLA_PYTHON_CLIENT_MEM_FRACTION 0.85 (trm/train/start.py),
# where memory_stats() reports bytes_limit = 4,883 MiB on the 6GB RTX 2060 —
# instruments/vram_headroom_smoke reads it there, and trm/config.py's layer table
# is the same number from the other side (4,437 MiB peak + 446 MiB headroom at 9
# layers; 4,112 + 771 at 8). nvidia-smi cannot see this: cuda_async holds the whole
# pool, so it reads ~5,003 MiB whatever the trainer is actually using.
ARENA_LIMIT_MIB = 4883.0

# ── palette ──────────────────────────────────────────────────────────────────
# Dark surface (it screenshots well), but the steps are the validated dark
# categorical slots, not "bright on black": slots 1-3 clear the colour-vision
# and contrast gates as a set, so no panel uses more than three series.
SURFACE = "#1a1a19"
INK = "#ffffff"
INK_DIM = "#c3c2b7"
GRID = "#33332f"
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_DIM, "axes.titlecolor": INK,
    "axes.edgecolor": GRID, "xtick.color": INK_DIM, "ytick.color": INK_DIM,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.labelcolor": INK_DIM,
    "font.size": 10, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "figure.dpi": 130, "savefig.bbox": "tight",
}


# ── small helpers ────────────────────────────────────────────────────────────

def _clean(steps, values):
    """Drop points that cannot be drawn (None / NaN / inf) and any replayed
    step. A torn last CSV row and a resumed run that rewinds its step counter
    are both normal here, and neither should put a spike in the hero image."""
    out_s, out_v, last = [], [], -1
    for s, v in zip(steps, values):
        if s is None or v is None or s <= last:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fv):
            continue
        out_s.append(int(s))
        out_v.append(fv)
        last = int(s)
    return np.array(out_s, dtype=float), np.array(out_v, dtype=float)


def available(runlog, name):
    """Is this column worth an axis?

    Two ways it is not. It can be *absent* — the architecture never measured it,
    which since #105 is a blank cell. Or it can be present and constant at zero,
    which is what runs written before that convention put in the columns their
    architecture did not measure: `has()` is honestly True and the data is still
    not a measurement. Both cases must be omitted rather than drawn, because a
    flat zero line reads as "we measured zero" instead of "there is nothing
    here" — and the reader is told which of the two it was (see `why_omitted`).
    """
    return runlog.has(name) and not runlog.is_constant(name, 0.0)


def why_omitted(runlog, columns, absent=None):
    """None when at least one of these columns is worth drawing, else the reason
    the panel is being dropped — in the reader's words, not the code's. By default the
    reason comes from the run's recorded arch (`runlog.absence_reason`, shared with
    report.py): a plain run missing a column plain logs was not "not measured by this
    architecture". A column missing for a known other reason passes its own `absent`."""
    if any(available(runlog, column) for column in columns):
        return None
    if any(runlog.has(column) for column in columns):
        return "constant 0 throughout — logged, but not a measurement"
    if absent is not None:
        return absent
    cfg = RunConfig(getattr(runlog, "metadata", {}))
    return absence_reason(cfg.arch if cfg.recorded("MODEL_ARCH") else None, columns)


def series(runlog, name, cfg=None, suspect=None):
    """(tokens, values) for a column, cleaned. Empty arrays if it has no data.

    Steps become tokens at the *run's* tokens-per-opt-step, not this process's.
    Rows failing an invariant (#195) are dropped here rather than in each caller,
    so no figure can accidentally plot one. The resume artifact (#194) puts a
    ~40% downward spike in the curve otherwise — a dip the model never had.
    """
    if not available(runlog, name):
        return np.array([]), np.array([])
    if suspect is None:
        # Computed once per figure (RunConfig.of), not once per panel.
        suspect = cfg.suspect if cfg is not None and cfg.suspect is not None else suspect_rows(runlog)
    bad = suspect
    steps, values = _clean(*clean_column(runlog, name, bad))
    return steps * (cfg or RunConfig.of(runlog)).tokens_per_opt_step, values


def smooth(y, window):
    """Centred moving average, edge-padded so it spans the same x as the raw."""
    if window < 3 or len(y) < window:
        return np.asarray(y)
    kernel = np.ones(window) / window
    core = np.convolve(y, kernel, mode="valid")
    front = window // 2
    return np.pad(core, (front, len(y) - len(core) - front), mode="edge")


def smoothing_window(n):
    return 0 if n < 20 else int(min(101, max(5, n // 20)))


def fmt_tokens(n):
    n = float(n)
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= scale:
            return f"{n / scale:.4g}{suffix}"
    return f"{n:.0f}"


def _token_axis(ax, label="tokens trained"):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt_tokens(v)))
    ax.set_xlabel(label)


def _log_y(ax):
    """Log y, with the in-between ticks labelled *when they fit*.

    A CE curve spans less than one decade for most of a run, and a default log
    axis would then label a single tick — technically a log scale, practically
    unreadable. Over a wide range the same labels become a wall of text, so the
    minor formatter decides at draw time, once the limits are final."""
    ax.set_yscale("log")
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=tuple(range(2, 10))))

    def minor(value, _):
        low, high = ax.get_ylim()
        return f"{value:g}" if low > 0 and high / low <= 20 else ""

    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(minor))
    ax.tick_params(axis="y", which="minor", labelsize=8)


def _legend(ax, **kw):
    if ax.get_legend_handles_labels()[0]:
        ax.legend(**kw)


def _note(ax, text, size=8):
    """The caveat under a panel: what the number is, and what it is not. Placed
    a fixed number of points below the axes so tall and short panels agree, and
    wrapped to the panel's own width."""
    inches = ax.get_position().width * ax.figure.get_figwidth()
    ax.annotate(textwrap.fill(text, max(40, int(inches * 72 / (0.55 * size)))),
                xy=(0, 0), xycoords="axes fraction", xytext=(0, -46),
                textcoords="offset points", fontsize=size, color=INK_DIM, va="top")


# ── run context (metadata, supervisor heartbeats) ────────────────────────────

class RunConfig:
    """What the *plotted run* was configured as, read from its own metadata.

    `run_metadata.json`'s `parameters` block is the run's recipe as it was at
    launch. Anything it did not record falls back to this process's `trm.config`
    — the only option for runs written before the tracker recorded that key, and
    a guess, which is why `recorded()` exists: a panel that cannot trust a value
    can say so instead of drawing a confident wrong picture.
    """

    def __init__(self, metadata):
        params = (metadata or {}).get("parameters")
        self.params = params if isinstance(params, dict) else {}
        self.suspect = None  # rows failing an invariant; set by `of`, which has the rows

    @classmethod
    def of(cls, runlog):
        cfg = cls(getattr(runlog, "metadata", {}))
        cfg.suspect = suspect_rows(runlog)
        return cfg

    def recorded(self, key):
        """Did the run itself record this, or are we about to guess?"""
        return self.params.get(key) is not None

    def value(self, key, fallback, cast=None):
        value = self.params.get(key)
        if value is None:
            return fallback
        try:
            return cast(value) if cast else value
        except (TypeError, ValueError):  # a hand-edited metadata file
            return fallback

    # ── what model this was ──
    @property
    def arch(self):
        return self.value("MODEL_ARCH", this_process.MODEL_ARCH, str)

    @property
    def latent_dim(self):
        return self.value("LATENT_DIM", this_process.LATENT_DIM, int)

    @property
    def layers(self):
        return self.value("PLAIN_LAYERS", this_process.PLAIN_LAYERS, int)

    @property
    def max_depth(self):
        return self.value("MAX_STEPS_LIMIT", this_process.MAX_STEPS_LIMIT, int)

    @property
    def vocab_size(self):
        return self.value("VOCAB_SIZE", this_process.VOCAB_SIZE, int)

    @property
    def max_seq_len(self):
        return self.value("MAX_SEQ_LEN", this_process.MAX_SEQ_LEN, int)

    # ── how it was trained ──
    @property
    def optimizer(self):
        return self.value("TRM_OPTIMIZER", this_process.TRM_OPTIMIZER, str)

    @property
    def muon_lr_mult(self):
        return self.value("MUON_LR_MULT", this_process.MUON_LR_MULT, float)

    @property
    def warmup_steps(self):
        return self.value("WARMUP_STEPS", WARMUP_STEPS, int)

    @property
    def peak_lr(self):
        return self.value("PEAK_LR", PEAK_LR, float)

    @property
    def budget(self):
        """Planned token budget, or None when the run was launched without one."""
        return self.value("TRAIN_TOKEN_BUDGET", this_process.TRAIN_TOKEN_BUDGET)

    @property
    def tokens_per_opt_step(self):
        """Each micro-step scores two windows; accumulation × batch of them make
        one optimizer step. Derived from the run's own recipe when it recorded
        all three, because a run at a different batch recipe is mis-scaled by
        this process's constant."""
        recorded = recorded_tokens_per_opt_step(self.params)
        return this_process.TOKENS_PER_OPT_STEP if recorded is None else recorded

    @property
    def micro_steps(self):
        """Micro-steps folded into one optimizer step (accumulation × batch)."""
        return max(1, self.tokens_per_opt_step // (2 * self.max_seq_len))

    @property
    def decay_steps(self):
        """The LR anneal's horizon in optimizer steps, as the run resolved it."""
        if self.recorded("DECAY_STEPS"):
            return self.value("DECAY_STEPS", None, int)
        if self.budget:
            return max(1, round(self.budget / self.tokens_per_opt_step))
        return resolve_decay_steps(None)


def describe(cfg):
    """The one line that says which model this run trained — the identifying
    facts, per architecture. `depth ≤N` is one of them for the two looping
    arches and meaningless for the plain stack, which has N distinct blocks and
    no loop; the optimizer is one of them for every run (#26 trains matched arms
    that differ in nothing else).

    A fact the run did not record is named as missing rather than filled in from
    this process: PLAIN_LAYERS was 8 before 2026-09-13 and is 9 now, and printing
    an adamw run's label on a muon run is precisely the failure this whole issue
    is about."""
    parts = [cfg.arch, f"dim {cfg.latent_dim}"]
    if cfg.arch != "plain":
        parts.append(f"depth ≤{cfg.max_depth}")
    elif cfg.recorded("PLAIN_LAYERS"):
        parts.append(f"{cfg.layers} layers")
    else:
        parts.append("layer count not recorded")

    if not cfg.recorded("TRM_OPTIMIZER"):
        parts.append("optimizer not recorded")
    elif cfg.optimizer == "muon" and cfg.recorded("MUON_LR_MULT"):
        parts.append(f"muon (LR ×{cfg.muon_lr_mult:g} on matrices)")
    else:
        parts.append(cfg.optimizer)

    # The peak the schedule warms up to. Named only when the run recorded it and
    # it is not the historical 1e-4, so every existing label is unchanged and the
    # #287 arms — which differ in nothing else — cannot be confused for each other.
    if cfg.recorded("PEAK_LR") and cfg.peak_lr != PEAK_LR:
        parts.append(f"peak LR {cfg.peak_lr:g}")
    return " · ".join(parts)


def lr_schedule(cfg):
    """(schedule, None), or (None, why the LR panel cannot be drawn).

    Rebuilding the anneal needs the run's warmup, and runs written before #305
    did not record it. Substituting this process's `WARMUP_STEPS` is what made
    the plotter *crash* on every short arm: a 512-step horizon minus a 1000-step
    warmup is a negative cosine, and optax refuses it. So when the warmup we hold
    does not fit the run's horizon, the panel is omitted with the reason said out
    loud — the same rule the diagnostic sheets follow for a column nobody
    measured."""
    if cfg.warmup_steps < cfg.decay_steps:
        return build_learning_schedule(cfg.decay_steps, warmup_steps=cfg.warmup_steps,
                                       peak_lr=cfg.peak_lr), None
    if cfg.recorded("WARMUP_STEPS"):
        return None, (f"the run's warmup ({cfg.warmup_steps:,} steps) covers its whole "
                      f"{cfg.decay_steps:,}-step horizon — there is no anneal to draw")
    return None, (f"this run did not record its warmup, and this process's "
                  f"WARMUP_STEPS={WARMUP_STEPS:,} does not fit its {cfg.decay_steps:,}-step "
                  f"horizon — the schedule cannot be rebuilt, only guessed at "
                  f"(re-run with WARMUP_STEPS set to the run's own to draw it)")


def val_cadence(runlog, cfg):
    """(optimizer steps between held-out probes, how we know), or (None, None).

    The run's own knob when it recorded one, else read off the val_ce rows —
    which is still a measurement of this run. Never this process's default:
    `VAL_EVERY_OPT_STEPS` is an env knob, and the note that used to read "every
    64 optimizer steps" was wrong for every ablation arm (they run 16).

    The observed figure is the mean gap, not the median. The probe fires on its
    own cadence but its value is only written on the next *logged* row, so a
    probe every 16 steps logged every 5 lands at 20, 35, 50, 65, 80, 100 — gaps
    of 15, 15, 15, 15, 20, whose median is 15 and whose mean is exactly 16."""
    if cfg.recorded("VAL_EVERY_OPT_STEPS"):
        return cfg.value("VAL_EVERY_OPT_STEPS", None, int), "recorded"
    steps, _ = runlog.column("val_ce")
    if len(steps) < 2:
        return None, None
    return int(round((steps[-1] - steps[0]) / (len(steps) - 1))), "observed"


_HEARTBEAT = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s+\w+:\s*step\s+(\d+)/")


def read_heartbeats(runlog):
    """[(timestamp, opt step)] from the supervisor's hourly status lines.

    This is the only wall-clock a run records against its progress — metrics.csv
    has no timestamps at all — so throughput here is *sampled hourly*, never
    measured per step."""
    run_id = str(runlog.run_id)
    run_dir = pathlib.Path(getattr(runlog, "run_dir", "") or REPO_ROOT / "runs" / run_id)
    candidates = [
        run_dir.parent / f"{run_id}.supervisor.log",   # how the launcher names it
        run_dir / "supervisor.log",
        REPO_ROOT / "runs" / f"{run_id}.supervisor.log",
    ]

    beats, last = [], -1
    for path in candidates:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            match = _HEARTBEAT.match(line)
            if not match:
                continue
            step = int(match.group(2))
            if step <= last:  # a relaunch replays steps; keep the advancing ones
                continue
            beats.append((datetime.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"), step))
            last = step
        if beats:
            break
    return beats


# ── figure 1: the hero ───────────────────────────────────────────────────────

def _perplexity_axis(ax):
    """Perplexity ticks on the right. Not a second measure on a second scale —
    the same CE curve, relabelled exp(CE), because perplexity is the number
    that compares to published models."""
    low, high = ax.get_ylim()
    twin = ax.twinx()
    twin.set_yscale(ax.get_yscale())
    twin.set_ylim(low, high)
    twin.grid(False)
    twin.spines["right"].set_visible(True)
    ticks = [p for p in (2, 3, 5, 10, 20, 50, 100, 200, 500, 1e3, 5e3, 1e4, 5e4)
             if low <= math.log(p) <= high]
    twin.set_yticks([math.log(p) for p in ticks])
    twin.set_yticklabels([f"{p:,.0f}" for p in ticks])
    twin.minorticks_off()
    twin.set_ylabel("perplexity", color=INK_DIM)
    return twin


def _param_count(cfg):
    """Headline parameter count, analytic, for the shape the *run* trained.
    Guarded: the figures are worth drawing even if the stats module is
    unavailable — but the model is never built to get this number."""
    try:
        from instruments.model_stats import total_params

        shape = {"dim": cfg.latent_dim, "vocab_size": cfg.vocab_size,
                 "max_seq_len": cfg.max_seq_len}
        if cfg.arch == "plain":
            shape["num_layers"] = cfg.layers
        else:
            shape["max_depth"] = cfg.max_depth
        return int(total_params(cfg.arch, **shape))
    except Exception:
        return None


def training_curve(runlog, outdir):
    cfg = RunConfig.of(runlog)
    tokens, ce = series(runlog, "ce", cfg)
    if len(ce) == 0:
        print(f"training_curve: skipped — CE is {why_omitted(runlog, ['ce'])}.")
        return None

    val_tokens, val_ce = series(runlog, "val_ce", cfg)
    if not len(val_ce):
        print(f"training_curve: no val CE line — {why_omitted(runlog, ['val_ce'])}.")
    budget, decay_steps = cfg.budget, cfg.decay_steps
    tokens_per_step = cfg.tokens_per_opt_step
    horizon = budget if budget else decay_steps * tokens_per_step
    done = tokens[-1]

    # The LR panel is conditional like every other: a schedule we cannot rebuild
    # is omitted with its reason, never approximated.
    schedule, no_schedule = lr_schedule(cfg)
    if no_schedule:
        print(f"training_curve: no LR panel — {no_schedule}.")

    fig = plt.figure(figsize=(13.5, 8.6 if schedule else 6.6))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.55)
    ax = fig.add_subplot(grid[0] if schedule else grid[:])
    window = smoothing_window(len(ce))

    ax.plot(tokens, ce, color=BLUE, alpha=0.22, linewidth=1.0)
    ax.plot(tokens, smooth(ce, window), color=BLUE, linewidth=2.0,
            label="train CE (window 2)" + (f", {window}-point mean" if window else ""))
    if len(val_ce):
        ax.plot(val_tokens, val_ce, color=ORANGE, linewidth=1.8, marker="o",
                markersize=3, label="held-out val CE")

    uniform = math.log(cfg.vocab_size)
    if ce.max() > uniform * 0.9:
        ax.axhline(uniform, color=INK_DIM, linewidth=0.9, alpha=0.5)
        ax.text(tokens[0], uniform, f" uniform over {cfg.vocab_size:,} vocab ({uniform:.2f} nats)",
                fontsize=8, color=INK_DIM, va="bottom")

    _log_y(ax)
    ax.set_ylabel("cross-entropy (nats)")
    _token_axis(ax)
    ax.set_xlim(0, done * 1.06)

    steps_axis = ax.secondary_xaxis("top", functions=(lambda t: t / tokens_per_step,
                                                      lambda s: s * tokens_per_step))
    steps_axis.set_xlabel("optimizer steps", fontsize=9)

    # Direct labels on the two endpoints; the axis carries everything else.
    ax.annotate(f"{ce[-1]:.3f}", (tokens[-1], smooth(ce, window)[-1]), color=BLUE,
                fontsize=9, fontweight="bold", xytext=(6, -2), textcoords="offset points")
    if len(val_ce):
        ax.annotate(f"{val_ce[-1]:.3f}", (val_tokens[-1], val_ce[-1]), color=ORANGE,
                    fontsize=9, fontweight="bold", xytext=(6, -2), textcoords="offset points")

    _perplexity_axis(ax)
    _legend(ax, loc="upper right", bbox_to_anchor=(1.0, 0.93))

    params = _param_count(cfg)
    subtitle = describe(cfg)
    if params:
        # A count built partly from this process's config is marked as such: a
        # plain block is ~10.6M params, so an unrecorded layer count moves this
        # number by more than its own precision.
        shape_recorded = cfg.recorded("PLAIN_LAYERS" if cfg.arch == "plain" else "MAX_STEPS_LIMIT")
        subtitle += f" · {'' if shape_recorded else '~'}{params / 1e6:.1f}M params"
    subtitle += f"  |  {fmt_tokens(done)} tokens"
    if budget:
        subtitle += f" of {fmt_tokens(budget)} ({100 * done / budget:.1f}% of budget)"
    ax.set_title(f"{runlog.run_id} — training curve\n{subtitle}", loc="left", linespacing=1.6)
    cadence, known = val_cadence(runlog, cfg)
    if known == "recorded":
        measured = f"measured every {cadence:,} optimizer steps."
    elif known == "observed":
        measured = (f"measured about every {cadence:,} optimizer steps — read off its own rows, "
                    "since the run did not record the cadence.")
    else:
        measured = "measured on the run's own cadence."
    _note(ax, "train CE is the second window of each document (more context than the first); "
              f"val CE is a held-out probe, {measured}")

    # ── LR panel: the whole horizon, so the anneal is visible against progress
    if schedule:
        ax_lr = fig.add_subplot(grid[1])
        sched_steps = np.linspace(0, decay_steps, 400)
        ax_lr.plot(sched_steps * tokens_per_step, [float(schedule(s)) for s in sched_steps],
                   color=AQUA, linewidth=1.8)
        # A finished run can overshoot its horizon by a few steps; keep the marker
        # inside the panel so its label does not float off the edge.
        now = min(done, horizon)
        ax_lr.axvspan(0, now, color=INK, alpha=0.06)
        ax_lr.axvline(now, color=INK_DIM, linewidth=1.0)
        ax_lr.annotate(f"now · {fmt_tokens(done)}", (now, 1.0), xycoords=("data", "axes fraction"),
                       xytext=(-5 if now > 0.85 * horizon else 5, -10), textcoords="offset points",
                       fontsize=8.5, color=INK, ha="right" if now > 0.85 * horizon else "left")
        ax_lr.set_yscale("log")
        ax_lr.set_ylabel("learning rate")
        ax_lr.set_xlim(0, horizon)
        _token_axis(ax_lr, "tokens — the full planned budget" if budget
                    else f"tokens — LR horizon ({decay_steps:,} steps; no budget recorded)")
        ax_lr.set_title(f"LR schedule over the run's horizon — {cfg.warmup_steps:,}-step warmup, "
                        f"cosine to {decay_steps:,} (shaded = done)", loc="left")

    path = pathlib.Path(outdir) / "training_curve.png"
    fig.savefig(path)
    plt.close(fig)
    omitted = [] if len(val_ce) else [("val_ce", why_omitted(runlog, ["val_ce"]))]
    if no_schedule:
        omitted.append(("lr", no_schedule))
    return {
        "path": str(path),
        "panels": ["ce"] + (["lr"] if schedule else []) + (["val_ce"] if len(val_ce) else []),
        "omitted": omitted,
    }


# ── figure 2: throughput & progress ──────────────────────────────────────────

# How much longer than the usual beat spacing an interval may be before we stop
# believing it measures throughput. Heartbeats land on a fixed poll schedule, so
# a much longer gap means beats went *missing* — and the wall-clock inside it
# counts time the trainer was not running.
GAP_FACTOR = 2.5


def usable_intervals(hours, tokens):
    """(rate, rate_hours, dropped, cadence) — throughput over the intervals that
    actually measure throughput.

    Δtokens ÷ Δwall-clock is only a rate if the trainer ran for the whole
    interval. Across a logging gap — a crash, a manual stop, an eval borrowing
    the card — the same ratio measures *duty cycle*, and it renders on the plot
    as a throughput collapse that never happened (run_20260813_214725 showed a
    fake dip to 1,000 tok/s and a fake decline to 2,900 against a true, flat
    ~4,500). Drop those intervals and hand back the count so the caller can say
    what went, rather than silently narrowing the data.
    """
    d_hours, d_tokens = np.diff(hours), np.diff(tokens)
    positive = d_hours > 0
    if not positive.any():
        return np.array([]), np.array([]), 0, 0.0
    cadence = float(np.median(d_hours[positive]))
    clean = positive & (d_hours <= GAP_FACTOR * cadence)
    dropped = int((positive & ~clean).sum())
    if not clean.any():
        return np.array([]), np.array([]), dropped, cadence
    return (d_tokens[clean] / (d_hours[clean] * 3600.0),
            hours[1:][clean], dropped, cadence)


THROUGHPUT_SPACING_S = 1800  # thin logged rows to intervals at least this long


def clock_samples(runlog, min_spacing_s=THROUGHPUT_SPACING_S):
    """([(timestamp, opt step)], source): the run's own clock when it has one.

    metrics.csv carries wall_clock since #186, on every logged row, so throughput
    is measured against the file that is the authority on progress. Rows are
    thinned to intervals of at least `min_spacing_s` so each point is an end-to-end
    rate (checkpoints and probes included), not five opt steps of noise. Runs
    older than the column fall back to supervisor heartbeats, which are sampled.
    """
    stamped = [(row["wall_clock"], row["step"]) for row in runlog.metrics if row.get("wall_clock")]
    if len(stamped) >= 2:
        thinned = [stamped[0]]
        for stamp, step in stamped[1:]:
            if (stamp - thinned[-1][0]).total_seconds() >= min_spacing_s:
                thinned.append((stamp, step))
        if thinned[-1] != stamped[-1]:
            thinned.append(stamped[-1])
        if len(thinned) >= 2:
            return thinned, "metrics"
    return read_heartbeats(runlog), "heartbeats"


def throughput_progress(runlog, outdir):
    cfg = RunConfig.of(runlog)
    beats, source = clock_samples(runlog)
    measured = source == "metrics"
    budget = cfg.budget
    tokens_per_step = cfg.tokens_per_opt_step
    done_tokens = runlog.last_step * tokens_per_step

    if len(beats) < 2:
        print("throughput_progress: no wall_clock in metrics.csv and fewer than two supervisor "
              "heartbeats — skipped (throughput needs a clock).")
        return None

    times = [t for t, _ in beats]
    steps = np.array([s for _, s in beats], dtype=float)
    hours = np.array([(t - times[0]).total_seconds() / 3600.0 for t in times])
    tokens = steps * tokens_per_step

    # How much of the run the heartbeats actually witnessed. They are a separate
    # stream from metrics.csv — the supervisor's stdout — so they stop the moment
    # a relaunch redirects that stdout or lengthens --heartbeat-hours, while the
    # run itself carries happily on. run_20260813_214725 lost them at step 13,890
    # of 30,520 and this figure drew 45% of a run as though it were the whole
    # thing, with a "recent mean" computed from six-day-old data. metrics.csv is
    # the authority on how far the run got; compare against it and say so.
    last_beat_step = int(steps[-1])
    coverage = last_beat_step / runlog.last_step if runlog.last_step else 0.0
    stale = coverage < 0.98

    rate, rate_hours, dropped, cadence = usable_intervals(hours, tokens)
    if rate.size == 0:
        print("throughput_progress: every heartbeat interval spans a logging gap "
              "— skipped (no interval measures throughput rather than downtime).")
        return None
    recent = float(np.mean(rate[-6:]))

    fig, (ax_rate, ax_prog) = plt.subplots(2, 1, figsize=(12.5, 8.4))
    fig.subplots_adjust(hspace=0.55)

    ax_rate.plot(rate_hours, rate, color=BLUE, linewidth=1.6, marker="o", markersize=3,
                 label="tokens/sec (per interval)")
    ax_rate.axhline(recent, color=ORANGE, linewidth=1.2, linestyle="--",
                    label=f"recent mean {recent:,.0f} tok/s")
    ax_rate.set_ylim(0, max(rate.max(), recent) * 1.25)
    ax_rate.set_ylabel("tokens / second")
    ax_rate.set_xlabel("hours since the run's first " + ("logged row" if measured else "heartbeat"))
    title = ("Throughput — measured from metrics.csv wall_clock" if measured
             else "Throughput — sampled from heartbeats, not measured")
    if stale:
        title += (f"  ⚠ heartbeats cover only {100 * coverage:.0f}% of the run "
                  f"(to step {last_beat_step:,} of {runlog.last_step:,})")
    ax_rate.set_title(title, loc="left", **({"color": ORANGE} if stale else {}))
    _legend(ax_rate, loc="lower right")
    note = (f"each point is one {'interval between logged rows' if measured else 'supervisor heartbeat interval'} "
            f"(~{cadence:.1f}h apart): "
            f"Δsteps × {tokens_per_step:,} tokens ÷ Δwall-clock. "
            "It includes checkpointing and validation, so it is the honest end-to-end rate.")
    if dropped:
        note += (f"\n{dropped} interval(s) spanning a logging gap were dropped — across a gap this "
                 "ratio measures duty cycle (downtime included), not throughput.")
    if stale:
        note += (f"\n⚠ the supervisor stopped logging at step {last_beat_step:,}; the run reached "
                 f"{runlog.last_step:,}. Everything after that is UNPLOTTED, not flat — this panel "
                 "describes the first part of the run only.")
    _note(ax_rate, note)

    ax_prog.plot(hours, tokens, color=BLUE, linewidth=2.0, label="tokens trained")
    eta_text = None
    # No projection off stale heartbeats. The ETA is anchored to times[-1], so a
    # heartbeat stream that died days ago yields a confidently-wrong finish date
    # — worse than no date at all.
    if budget and recent > 0 and done_tokens < budget and not stale:
        # Against the CSV's position, not the last heartbeat's: the heartbeat is
        # up to an hour stale, and the hero image counts from the CSV. Two
        # figures disagreeing about "how far in are we" is worse than an ETA
        # that is an hour optimistic.
        remaining_h = (budget - done_tokens) / (recent * 3600.0)
        finish_h = hours[-1] + remaining_h
        ax_prog.plot([hours[-1], finish_h], [tokens[-1], budget], color=BLUE,
                     linewidth=1.4, linestyle="--", alpha=0.6,
                     label="projection at the recent mean rate")
        ax_prog.axhline(budget, color=ORANGE, linewidth=1.2)
        ax_prog.text(finish_h, budget, f"budget {fmt_tokens(budget)} ", color=ORANGE,
                     fontsize=9, va="bottom", ha="right")
        eta = times[-1] + datetime.timedelta(hours=remaining_h)
        eta_text = (f"{100 * done_tokens / budget:.1f}% done · {remaining_h / 24:.1f} days left · "
                    f"ETA {eta:%Y-%m-%d %H:%M}")
        ax_prog.set_xlim(0, finish_h * 1.02)
        ax_prog.set_ylim(0, budget * 1.12)
    ax_prog.set_ylabel("tokens")
    ax_prog.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
    ax_prog.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt_tokens(v)))
    ax_prog.set_xlabel("hours since the run's first " + ("logged row" if measured else "heartbeat"))
    ax_prog.set_title("Progress against the token budget"
                      + (f" — {eta_text}" if eta_text else ""), loc="left")
    _legend(ax_prog, loc="lower right")
    prog_note = f"{fmt_tokens(done_tokens)} tokens at optimizer step {runlog.last_step:,}. "
    if stale:
        # The curve is drawn from heartbeats and therefore stops early, while this
        # note quotes metrics.csv. Saying so beats letting the reader assume the
        # run stalled where the line ends.
        prog_note += (f"The curve stops at step {last_beat_step:,} because the heartbeats do; "
                      "the run continued past it. No ETA is drawn — projecting from a dead "
                      "heartbeat stream gives a confident wrong answer.")
    else:
        prog_note += ("The ETA extrapolates the recent mean rate; it is an estimate, and it "
                      "assumes no crash-relaunch and no throughput drift.")
    _note(ax_prog, prog_note)

    # Loud on stdout too: whoever regenerates the plots should learn the panel is
    # partial without having to notice orange text inside the image.
    if stale:
        print(f"⚠ throughput_progress: supervisor heartbeats stop at step {last_beat_step:,} "
              f"of {runlog.last_step:,} ({100 * coverage:.0f}% of the run) — the throughput panel "
              "describes only that portion, and no ETA was drawn.")
    if dropped:
        print(f"  throughput_progress: dropped {dropped} heartbeat interval(s) spanning a logging "
              "gap (they measure duty cycle, not throughput).")

    path = pathlib.Path(outdir) / "throughput_progress.png"
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "panels": ["throughput", "progress"],
            "omitted": [], "coverage": coverage, "dropped_intervals": dropped}


# ── figure 3: optimization health ────────────────────────────────────────────

def _panel_grad_norm(ax, runlog, cfg):
    if runlog.has("applied_grad_norm"):
        # The norm the clip actually sees (#180): read directly against the line.
        from trm.train.optimizers import CLIP_NORM
        tokens, values = series(runlog, "applied_grad_norm", cfg)
        window = smoothing_window(len(values))
        ax.plot(tokens, values, color=BLUE, alpha=0.22, linewidth=1.0)
        ax.plot(tokens, smooth(values, window), color=BLUE, linewidth=1.8, label="applied (window mean)")
        ax.axhline(CLIP_NORM, color=ORANGE, linewidth=1.2, linestyle="--", label=f"clip at {CLIP_NORM:g}")
        clipped = float(np.mean(values > CLIP_NORM)) if len(values) else 0.0
        _log_y(ax)
        ax.set_ylabel("‖g‖₂")
        ax.set_title(f"Gradient norm the clip sees — clipped on {clipped:.0%} of logged steps", loc="left")
        _legend(ax, loc="best", fontsize=8)
        _note(ax, "the norm of the accumulated gradient the optimizer applies. Above the dashed line "
                  "clip_by_global_norm, not the LR schedule, is setting the step size.")
        return
    tokens, values = series(runlog, "grad_norm_avg", cfg)
    window = smoothing_window(len(values))
    ax.plot(tokens, values, color=BLUE, alpha=0.22, linewidth=1.0)
    ax.plot(tokens, smooth(values, window), color=BLUE, linewidth=1.8)
    _log_y(ax)
    ax.set_ylabel("‖g‖₂")
    ax.set_title("Gradient norm (raw, pre-clip)", loc="left")
    _note(ax, "the RAW per-micro-step norm. clip_by_global_norm(1.0) applies to the mean over "
              f"the {cfg.micro_steps:,}-micro-step accumulation, which is not logged — "
              "these two numbers are not comparable, so do not read this against the 1.0 clip (#180).")


def _panel_depth(ax, runlog, cfg):
    """Only for an architecture that HAS a depth (see `not_applicable`): the
    refiner loops its shared block, the reasoner scans, and the realised mean of
    the draw is worth a check. The plain stack takes the same argument and drops
    it, so there the panel is the sampler's dice roll and nothing else."""
    tokens, values = series(runlog, "depth_avg", cfg)
    expected = (1 + cfg.max_depth) / 2
    ax.plot(tokens, values, color=AQUA, alpha=0.25, linewidth=1.0)
    ax.plot(tokens, smooth(values, smoothing_window(len(values))), color=AQUA, linewidth=1.8)
    ax.axhline(expected, color=INK_DIM, linewidth=0.9, alpha=0.6)
    ax.text(tokens[-1] if len(tokens) else 0, expected, f"uniform mean {expected:.1f} ",
            fontsize=8, color=INK_DIM, va="bottom", ha="right")
    ax.set_ylabel("mean sampled depth")
    ax.set_title(f"Sampled reasoning depth (of ≤{cfg.max_depth})", loc="left")
    _note(ax, "depth is drawn uniformly per micro-step; this is the sampler's realised mean, "
              "a check that the draw is unbiased — not something the model learns.")


def _panel_vram(ax, runlog, cfg):
    """How close the run came to the allocator's ceiling (#168, #305). Logged on
    every row and, until now, never drawn — the one number that says whether the
    next layer or the next batch fits."""
    tokens, values = series(runlog, "arena_peak_mib", cfg)
    peak = float(values.max()) if len(values) else 0.0
    limit, recorded = arena_limit_mib(runlog)
    ax.plot(tokens, values, color=BLUE, linewidth=1.8, label="arena peak (high-water mark)")
    ax.axhline(limit, color=ORANGE, linewidth=1.2, linestyle="--",
               label=f"arena limit {limit:,.0f} MiB" + ("" if recorded else " (assumed)"))
    ax.set_ylim(0, max(limit, peak) * 1.08)
    ax.set_ylabel("MiB")
    ax.set_title(f"VRAM — peak {peak:,.0f} MiB, {limit - peak:,.0f} MiB spare "
                 f"({peak / limit:.0%} of the arena)", loc="left")
    _legend(ax, loc="lower right", fontsize=8)
    _note(ax, "peak_bytes_in_use, read from the allocator itself — not nvidia-smi, which reports "
              "the whole preallocated cuda_async pool (~5,003 MiB) whatever is in use. It is a "
              "high-water mark within a session, so it only ever climbs."
              + ("" if recorded else f" The limit is ASSUMED: this run did not log its own, so the "
                                     f"line is the {ARENA_LIMIT_MIB:,.0f} MiB measured on the 6GB RTX 2060 "
                                     f"at MEM_FRACTION 0.85, and is wrong for any other card or fraction."))


def arena_limit_mib(runlog):
    """(limit, recorded): the run's own logged `arena_limit_mib` when it has one, else
    the RTX 2060 measurement, flagged as assumed so the panel can say so."""
    _, limits = runlog.column("arena_limit_mib")
    if limits:
        return float(limits[-1]), True
    return ARENA_LIMIT_MIB, False


def _panel_zero_grad(ax, runlog, cfg):
    # The applied (window-mean) gradient when the run recorded it (#191); older runs
    # only have one micro-step's, which is labelled as such below.
    applied = runlog.has("applied_zero_frac_dense_max")
    tokens, values = series(runlog, "applied_zero_frac_dense_max" if applied else "zero_frac_dense_max", cfg)
    # Dots, not a line: this series is spiky by nature (an occasional micro-step
    # underflows, most do not), and joining the spikes draws a solid wall that
    # hides both the floor and how often the spikes happen.
    ax.plot(tokens, values, color=ORANGE, alpha=0.45, linestyle="none", marker=".",
            markersize=2.5, label="per logged step")
    window = smoothing_window(len(values))
    if window:
        ax.plot(tokens, smooth(values, window), color=AQUA, linewidth=1.4,
                label=f"{window}-point mean")
    if len(values) and values.min() > 0:
        _log_y(ax)
    ax.set_ylabel("fraction of zero entries")
    ax.set_title("Zero-gradient fraction (worst dense tensor, "
                 + ("applied gradient)" if applied else "one micro-step)"), loc="left")
    # "best" earns its keep here: the spikes and the floor move around, so the
    # free band between them is not always the same corner.
    _legend(ax, loc="best", fontsize=8)
    _note(ax, "the largest zero fraction over the dense parameter tensors at that step. "
              "Spikes are a live signal of an f16 gradient that underflowed to zero.")


def _panel_logits(ax, runlog, cfg):
    for name, colour, label in (("out_entropy", BLUE, "output entropy H"),
                                ("logz_mean", ORANGE, "mean log Z"),
                                ("max_abs_logit", AQUA, "max |logit|")):
        tokens, values = series(runlog, name, cfg)
        if not len(values):
            continue
        ax.plot(tokens, values, color=colour, alpha=0.22, linewidth=1.0)
        ax.plot(tokens, smooth(values, smoothing_window(len(values))), color=colour,
                linewidth=1.8, label=label)
    ax.set_ylabel("nats")
    ax.set_title("Logit health", loc="left")
    _legend(ax, loc="lower left", fontsize=9)
    _note(ax, "all three live on the same log-space scale, so they share one axis. Entropy "
              "falling toward 0 with |logit| climbing is the confident-collapse shape to watch for.")


HEALTH_PANELS = (
    ("grad_norm", ("grad_norm_avg", "applied_grad_norm"), _panel_grad_norm),
    ("depth", ("depth_avg",), _panel_depth),
    ("vram", ("arena_peak_mib",), _panel_vram),
    ("zero_grad", ("zero_frac_dense_max",), _panel_zero_grad),
    ("logits", ("out_entropy", "logz_mean", "max_abs_logit"), _panel_logits),
)


# Why a panel's column can be missing for a reason that is not the architecture.
ABSENT_REASON = {
    "vram": "not logged by this run — the column arrived with #168, and a CPU run "
            "has no allocator statistics to report",
}


def not_applicable(key, cfg):
    """Why this run's ARCHITECTURE makes a panel meaningless, whatever the CSV
    holds — the second way a panel can be wrong to draw, beside missing data.

    `depth_avg` measures something only for the depth-dialled arches (refiner,
    reasoner). Plain runs before #316 logged a sampled depth_avg that `PlainTransformer`
    took and ignored (trm/model/plain.py), so on such a run the panel plots the dice,
    not the model (#305); from #316 on, plain leaves it blank. Either way the panel is
    refused for plain. Only a run that *recorded* its arch is judged here — for one that
    did not, drawing is the lesser error.
    """
    if key == "depth" and cfg.recorded("MODEL_ARCH") and cfg.arch == "plain":
        return ("PlainTransformer ignores the depth argument — this column is the "
                "sampler's own draw, not a property of the model")
    return None


def optimization_health(runlog, outdir):
    """Every panel here is conditional on its data. A run of an architecture
    that does not produce one of these quantities simply gets a smaller sheet —
    never an axis of zeros standing in for a measurement that never happened.
    What was left out, and why, is said out loud rather than silently dropped."""
    cfg = RunConfig.of(runlog)
    drawn, omitted = [], []
    for key, columns, draw in HEALTH_PANELS:
        reason = not_applicable(key, cfg) or why_omitted(runlog, columns, ABSENT_REASON.get(key))
        (omitted.append((key, reason)) if reason else drawn.append((key, draw)))

    for key, reason in omitted:
        print(f"optimization_health: omitted {key} — {reason}.")
    if not drawn:
        print("optimization_health: nothing left to draw — skipped.")
        return None

    cols = 1 if len(drawn) == 1 else 2
    rows = math.ceil(len(drawn) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.6 * cols, 4.9 * rows), squeeze=False)
    fig.subplots_adjust(hspace=0.75, wspace=0.28)
    flat = list(axes.flat)

    for ax, (_, draw) in zip(flat, drawn):
        draw(ax, runlog, cfg)
        _token_axis(ax)
    for ax in flat[len(drawn):]:
        ax.axis("off")

    fig.suptitle(f"{runlog.run_id} — optimization health", x=0.09, ha="left",
                 fontsize=14, fontweight="bold")
    if omitted:
        fig.text(0.09, 0.008, "omitted — " + "; ".join(f"{k}: {r}" for k, r in omitted),
                 fontsize=8, color=INK_DIM)
    path = pathlib.Path(outdir) / "optimization_health.png"
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "panels": [key for key, _ in drawn],
            "omitted": omitted}


# ── entry point ──────────────────────────────────────────────────────────────

def build(log=None, outdir="."):
    """Write every figure the run has data for; return what was written."""
    runlog = load(log)
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(STYLE):
        figures = [
            training_curve(runlog, outdir),
            throughput_progress(runlog, outdir),
            optimization_health(runlog, outdir),
        ]
    figures = [f for f in figures if f]

    print(f"run {runlog.run_id}: step {runlog.last_step:,}, {fmt_tokens(runlog.tokens)} tokens")
    for figure in figures:
        print(f"  wrote {figure['path']}  [{', '.join(figure['panels'])}]")
    if not figures:
        print("  nothing written — the log has no plottable data yet.")
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", default=None,
                        help="metrics.csv or a run directory (default: the latest run)")
    parser.add_argument("--out", default=".", help="directory for the PNGs (default: repo root)")
    args = parser.parse_args()
    try:
        build(log=args.log, outdir=args.out)
    except FileNotFoundError as missing:
        raise SystemExit(str(missing))


if __name__ == "__main__":
    main()
