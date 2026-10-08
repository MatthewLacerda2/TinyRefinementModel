"""The run's knobs: one frozen `Config`, read from the environment once (#475).

`Config.from_env` is the only function in `trm/` that reads a knob from the
environment; `tests/core/test_config_read_once.py` holds that. pydantic-settings
converts each value to its field's type and runs the fail-closed checks, and
`model_dump()` is what a run records, so every knob is recorded by construction
(#358). An unset environment resolves to exactly the values every run so far used.

Jax-free on purpose: the supervisor, `trm.runtime.launch` and `trm.runtime.rewind`
run beside a training job and read their cadences from here. The compute policy
that needs jax (COMPUTE_DTYPE, the compilation cache) stays in `trm/config.py`.

Field name = environment variable = the key in run_metadata.json.
"""

import os
import pathlib
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, PositiveInt, ValidationError, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _optional(parse):
    """Unset or empty means None, anything else goes through `parse`."""
    return BeforeValidator(lambda v: None if v is None or v == "" else parse(v))


# Scientific notation is accepted ("2e9").
_TokenCount = Annotated[int | None, _optional(lambda v: int(float(v)))]
_OptSteps = Annotated[int | None, _optional(int)]
# Set to any non-empty value means on, as it always has (see FORCE_F32_COMPUTE).
_SetMeansOn = Annotated[bool, BeforeValidator(lambda v: bool(v))]

# The default mixture: the ramp every run so far trained with (see DATA_MIXTURE).
DEFAULT_DATA_MIXTURE = ("pretrain/fineweb-edu=0.85:0.35,"
                        "pretrain/codeparrot=0.10:0.40,"
                        "pretrain/finemath=0.05:0.25")


class Config(BaseSettings):
    # init_settings is the only source: `from_env` hands over the environment as a
    # mapping, so a test or an instrument can build the Config of any environment
    # and nothing reads os.environ behind its back. validate_default runs the
    # defaults through the same conversion as an environment value (a float knob's
    # default is a float).
    model_config = SettingsConfigDict(frozen=True, case_sensitive=True, validate_default=True,
                                      extra="forbid")

    @classmethod
    def settings_customise_sources(cls, *_, init_settings, **__):
        return (init_settings,)

    @classmethod
    def from_env(cls, environ=os.environ):
        """The Config this environment asks for, or a refusal naming every bad knob.

        Refuses with SystemExit, before a model is built: a typo must not silently
        train the wrong run (#104)."""
        # Strings from the environment; pydantic converts each to its field's type.
        given: dict[str, Any] = {name: environ[name] for name in cls.model_fields if name in environ}
        try:
            return cls(**given)
        except ValidationError as error:
            raise SystemExit("\n".join(
                f"{'.'.join(map(str, e['loc']))}={e['input']!r}: {e['msg']}" for e in error.errors())) from None

    # ── Compute ───────────────────────────────────────────────────────────────
    # A test-only escape hatch from the f16 policy (trm/config.py): CPU XLA cannot
    # lower the f16-with-f32-accumulation matmuls inside the rematerialized scan, so
    # the test suite (which defaults to CPU while the GPU trains) sets it. Never set
    # it for training.
    FORCE_F32_COMPUTE: _SetMeansOn = False

    # ── Architecture ──────────────────────────────────────────────────────────
    # v1 base-run width: 960 is the widest that fits the 6GB card with bf16 optimizer
    # moments (#18) — ~5.3GB peak at depth-8/batch-1, ~0.7GB margin; dim1024 OOMs. Not a
    # power of 2, but a clean multiple of 64, and the big VRAM lines (the 50304×dim
    # embedding/LM head, the FFN) don't care. What matters is head_dim — see NUM_HEADS.
    # Env-overridable only so tests/apparatus/test_trainer_end_to_end.py can run the real
    # trainer at toy size on CPU; both shape the param tree, so a resume checks them (#317).
    LATENT_DIM: int = 960
    MAX_SEQ_LEN: int = 512
    # 15 heads → head_dim = 960/15 = 64, the tensor-core-clean size on Turing f16.
    # (16 heads would give head_dim 60, not a multiple of 8 → XLA pads to 64: you pay
    # near-1024 attention cost for 960 of width. Avoid.) The attention asserts
    # dim%heads==0 and head_dim even for RoPE.
    NUM_HEADS: int = 15

    # Normalize each residual BRANCH's output before it is added back ("sandwich" /
    # post-norm, as in Gemma 2). The pre-norms bound what goes INTO attention and the
    # MLP; nothing bounds what comes out, and on the 4B champion that is measurably a
    # problem: encoder block 7's SwiGLU product reaches 39,104 on code from a norm2
    # input of 6 (SwiGLU multiplies two projections, each ~5x larger on code-shaped
    # input), and down_proj carries it to 64,896 -- 99.1% of f16's 65,504 ceiling.
    # That is the root cause of #229's whole-window NaN. See #235.
    #
    # OFF: the 2,000-step pair (PR #551) ended INCONCLUSIVE: -0.015 val CE at the cap,
    # 2% fewer tokens to 3.70 (bar 5%), +1.6% wall-clock; the 512-step lead (-0.056, PR
    # #538) mostly came from the start. It holds act_max near 75 where the control's
    # grows past 600. Adopting it for that is #494's call.
    POST_NORM: bool = False

    # PlainTransformer depth. It was 9 between 2026-09-13 and 2026-09-20, on the
    # allocator numbers for batch 1 (8 layers left 771 MiB of arena headroom, 9 left
    # 446, 10 left 121).
    #
    # Back to 8 for the base run (#385, owner 2026-09-20): the depth pays for itself
    # only if the card cannot spend the memory better, and it can. Measured on the real
    # trainer, 40 opt steps each with checkpoints every 16 and the validation probe
    # running (instruments/bench_train_step never ran either, which is why #24's batch-2
    # win was real and irrelevant):
    #   9 layers, batch 1   4,424 tok/s   arena peak 4058 MiB, 825 MiB headroom
    #   8 layers, batch 2   6,149 tok/s   arena peak 4502 MiB, 381 MiB headroom  <- this
    # +39% throughput, which is 531M tokens a day against 382M. The ninth layer costs
    # more than a day of every four. The fallback if a long run OOMs is 8 layers at
    # batch 1, which is both cheaper in memory than what shipped before and faster.
    PLAIN_LAYERS: int = 8

    # The residual stream's dtype on the plain stack (#357): the value every block adds
    # into, so an accumulator under the dtype policy (trm/config.py), and f32 as in
    # standard mixed precision. In f16 a block's O(1) output rounds to nothing once the
    # stream passes ~4k, and the f16 base run's last block was at ~4,000 by opt step
    # 19,800 (#357). The matmuls, and the norms' outputs that feed them, stay f16. Never
    # narrower than the compute dtype. float16 reproduces every run before #357.
    RESIDUAL_DTYPE: Literal["float16", "float32"] = "float32"

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Optimizer selector (#26), failing closed on an unknown name (#104):
    #   muon  — THE DEFAULT since 2026-09-19 (#388): orthogonalized momentum for the 2-D
    #           weight matrices, AdamW for the token embedding, norms and biases. It
    #           reaches a fixed held-out CE in 1.32x fewer tokens than AdamW at AdamW's
    #           own best peak (docs/findings/2026-09-18-muon-holds-1-32x-over-adamw-at-its-best-lr.md).
    #   adamw — the recipe every run before that used (trm/train/optimizers.py).
    # The embedding is 2-D but is a lookup table, not a linear map, so Muon must NOT
    # orthogonalize it; the partition lives in optimizers.muon_partition and
    # tests/core/test_muon_partition.py holds it, because a wrong partition trains
    # happily and silently.
    TRM_OPTIMIZER: Literal["adamw", "muon"] = "muon"
    # Muon's update has RMS ~1 per element after Newton-Schulz and the sqrt(rows/cols)
    # factor, so it wants a much larger LR than Adam's. The matrix partition runs the
    # shared schedule times this multiplier; the embedding/norm partition keeps the
    # schedule as is. 16.667 on the 6e-4 peak puts the matrices at 1e-2, the value the
    # #26 sweep chose (x25/x50/x100/x200 on 1e-4) and #382 confirmed at the new peak.
    MUON_LR_MULT: float = 16.666667
    # The optimizer's remaining knobs, named so that none is a library default nobody can
    # read (#358): an optax upgrade that moved one would have changed the recipe silently.
    # Each is what every run so far trained with, so every existing config resolves the same.
    #   ADAM_B2 is the memory of Adam's variance estimate, 1/(1-b2) opt steps: ~1000 at
    #   0.999, ~20 at 0.95. 0.95 since #359, the value GPT-3, LLaMA, PaLM and nanoGPT
    #   train with: a gradient spike then holds down the Adam partition's step (the tied
    #   embedding, the norms) for ~20 opt steps instead of ~1000. 0.999 reproduces every
    #   run before #359.
    ADAM_B1: float = 0.9
    ADAM_B2: float = 0.95
    ADAM_EPS: float = 1e-8
    # Weight decay on the >=2-D params, multiplied by the LR (so coupled to it): under muon the
    # matrices only, under adamw every >=2-D param, the embedding included.
    WEIGHT_DECAY: float = 1e-2
    # The token embedding's decay under muon (#360): per opt step at the peak LR, then following
    # the schedule's shape but not its scale (lr/PEAK_LR, Wortsman et al. 2023), so a PEAK_LR
    # change does not move it. 6e-5 is nanoGPT/GPT-3's 0.1 x their 6e-4 peak; WEIGHT_DECAY x LR
    # gave the table 6e-6, ten times less. 6e-6 restores the recipe before #360 at a 6e-4 peak.
    # It is Adam's partition that takes it: the >=2-D leaves Muon leaves to Adam, which are the
    # lookup tables; norms and biases stay undecayed.
    EMBED_WEIGHT_DECAY: float = 6e-5
    # The global-norm clip on the accumulation window's MEAN gradient. The trainer logs that
    # norm (applied_grad_norm, #180) and whether the clip bit (clip_active). 1.0 is a
    # measured choice since #447's pair: removing it cost +0.046 val CE at 512 steps (PR #560).
    CLIP_NORM: float = 1.0
    # Muon's own (optax.contrib.scale_by_muon): momentum, Newton-Schulz iterations, the
    # normalization epsilon, Nesterov.
    MUON_BETA: float = 0.95
    MUON_NS_STEPS: int = 5
    # Which Newton-Schulz coefficients (#375): "keller", Keller Jordan's 2024 quintic
    # reused at every step (every run so far), or "polar_express", a minimax-optimal
    # quintic per step (trm/train/polar_express.py). polar_express since #375's pair
    # (KEEP: -0.023 val CE at 512 steps, ahead from step 72 on, same tok/s; PR #538).
    MUON_NS_COEFFS: Literal["keller", "polar_express"] = "polar_express"
    MUON_EPS: float = 1e-8
    MUON_NESTEROV: bool = True
    # Clean micro-steps before the f16 loss scaler tries a larger S (#199). Each probe that
    # overflows discards a micro-step: ~200 of 65,536 in a 512-step pair. PyTorch's default
    # is 2,000; #368 asks whether ours should move. Named here, value unchanged.
    LOSS_SCALE_GROWTH_INTERVAL: int = 256

    # ── Loss ──────────────────────────────────────────────────────────────────
    # PaLM's z-loss (#369): Z_LOSS_WEIGHT * (log Z)^2 per scored position is added to the
    # gradient, which pins the log-partition of the tied head near 0 instead of letting
    # it drift (the 4B champion's went from 11 to 16-17). 1e-4 is PaLM's and OLMo 2's
    # value. The CE every run reports never includes it, so it stays comparable across
    # the change. 0 turns it off and reproduces every run before #369.
    Z_LOSS_WEIGHT: float = 1e-4

    # ── Batching ──────────────────────────────────────────────────────────────
    # BATCH_SIZE x ACCUMULATION_STEPS stays 128, so tokens per optimizer step, the LR
    # schedule and the learning dynamics are the same at any split; fatter micro-steps
    # only amortize the flat per-micro-step optimizer cost (#24).
    #
    # 2 is the widest that fits at dim 960 / 8 layers: a real trainer launch measured
    # 6,149 tok/s and 381 MiB of arena headroom on 2026-09-20 (#385's smoke: 382), and
    # batch 4 OOMs (#548). The headroom is thin, so the supervisor's fit gate (#168)
    # watches this knob; if a long run OOMs, drop to batch 1 rather than adding a layer
    # back. bench_train_step misranks this knob (#561): judge it with the real trainer.
    BATCH_SIZE: int = 2

    @field_validator("BATCH_SIZE")
    @classmethod
    def _batch_divides_128(cls, value):
        if value < 1 or 128 % value:
            raise ValueError("must divide 128, so tokens per opt step stay fixed")
        return value

    @computed_field
    @property
    def ACCUMULATION_STEPS(self) -> int:
        return 128 // self.BATCH_SIZE

    # Target tokens consumed per optimizer step: each micro-step scores two
    # MAX_SEQ_LEN prediction windows, ACCUMULATION_STEPS micro-steps make one opt step.
    @computed_field
    @property
    def TOKENS_PER_OPT_STEP(self) -> int:
        return self.ACCUMULATION_STEPS * self.BATCH_SIZE * 2 * self.MAX_SEQ_LEN

    # ── Held-out evaluation ───────────────────────────────────────────────────
    # Held-out evaluation reads a fixed number of *rows* (prediction-window pairs) and
    # scores them one row at a time, deliberately ignoring BATCH_SIZE: a throughput knob
    # must never redefine what "val CE" means.
    #
    # 64 since 2026-09-15 (#184); 4 rows were noisier than the plateau detector's bar.
    # Val CE recorded before that change is NOT comparable to val CE after it. The
    # probe's own sigma is measured by instruments/probe_sigma.py.
    EVAL_ROWS: int = 64
    # Where the fineweb validation slice starts: far past any plausible training
    # consumption (an 8k-opt-step run consumes under 1M fineweb samples; fineweb holds
    # 4.3M) so the slice stays held out.
    VAL_SKIP_SAMPLES: int = 3_000_000

    # The plateau detector reads HELD-OUT CE (#184): train CE is moved by the
    # curriculum underneath it (the #157 run's train CE rose 3.20 -> 3.36 while the
    # model improved). A plateau is "the windowed val CE has not improved by
    # PLATEAU_MIN_DELTA for PLATEAU_PATIENCE opt steps". The bar must sit above the
    # probe's own noise or the detector is a coin flip. Measured 2026-09-15
    # (instruments/probe_sigma.py, one plain checkpoint, 6 disjoint slices):
    #   4 rows  mean 4.74  sigma 0.195   (readings 4.49 .. 5.06 — which rows you got)
    #   64 rows mean 4.78  sigma 0.097
    # That is the level error between slices; documents are heavy-tailed, so 16x
    # the rows only halves it. The detector watches ONE fixed slice, whose
    # probe-to-probe jitter during training measured 0.011 nats at 4 rows (#184);
    # 0.01 is set at that jitter, above what 64 rows should show, and 2x the old
    # 0.005 that sat below it.
    PLATEAU_MIN_DELTA: float = 0.01
    PLATEAU_PATIENCE: int = 400

    # ── Schedules (built in trm/train/schedules.py) ───────────────────────────
    # Planned token budget for the run (#83), recorded in run_metadata.json. Drives
    # Schedules.decay_steps so the LR anneal bottoms out when training ends instead of
    # at a constant chosen for past short runs (15000 opt steps ≈ 2.0B tokens — an
    # anneal that would sit frozen at the floor for most of a longer run). Accepts plain
    # ints or scientific notation ("2e9"). Unset → the historical 15000-step horizon, so
    # existing configs and the golden run resolve unchanged.
    TRAIN_TOKEN_BUDGET: _TokenCount = None
    # Warmup is absolute: it stabilizes the optimizer's first moments, a fixed-cost
    # phase that does not grow with the run. Overridable for one purpose: a short
    # matched pair (#26 stage 2, ~500 opt steps) cannot spend 1000 of them warming
    # up. A base run leaves it alone.
    WARMUP_STEPS: int = 1000
    # The peak the schedule warms up to. 6e-4 since 2026-09-19 (#388): 1e-4 was chosen once
    # and never compared against anything; #287 measured 3e-4 at 2.25x fewer tokens and
    # 6e-4 at 1.24x fewer again (docs/findings/2026-09-18-the-lr-curve-flattens-6e-4-beats-3e-4-by-1-24x.md).
    # The golden run has its own optimizer and LR, so it does not move with this.
    PEAK_LR: float = 6e-4
    # The LR schedule's shape (#386): wsd (the default) or cosine, and where WSD's final
    # decay starts. What each one means is written beside the schedule, in
    # trm/train/schedules.py.
    LR_SCHEDULE: Literal["cosine", "wsd"] = "wsd"
    WSD_DECAY_FRACTION: float = 0.2
    WSD_DECAY_START: _OptSteps = None

    # ── Data ──────────────────────────────────────────────────────────────────
    # The id masked out of attention keys and loss targets. It used to be EOT_TOKEN_ID
    # (50256), and that plumbed the document separator into the pad sink (#373): EOT was
    # never a target (no learned "the document ends here" — a model that cannot stop
    # talking) and never a key (tokens after a boundary attended into the previous
    # document with no marker that one passed). 50257 is the first id past r50k's real
    # vocabulary, inside the padded table and never written by prefill, so as the pad it
    # leaves EOT an ordinary token. Adopted as the default for the base run (owner,
    # 2026-09-20) without its matched pair: #373 is a bug, not a hypothesis — a base run
    # that never learns to end a document is broken whatever the pair would have said.
    # Set PAD_TOKEN_ID=50256 to reproduce a run recorded under the old convention.
    PAD_TOKEN_ID: int = 50257

    @field_validator("PAD_TOKEN_ID")
    @classmethod
    def _pad_is_eot_or_past_the_vocab(cls, value):
        if value not in (50256, 50257):
            raise ValueError("use 50256 (historical, the EOT id) or 50257 (#373)")
        return value

    # Seeds (#17: the seed-variance noise floor needs same-config runs differing ONLY
    # in seed). Both are recorded in run_metadata.json so every run stays reproducible.
    #   DATA_SEED  — data-pipeline randomness: the start offset into each source
    #                (under 1,025 tokens), the mixture draws.
    #                NOT the document order. Every source is read front to back, so two
    #                seeds see nearly the same documents in the same order, and a pair's
    #                seed spread is init variance, not data variance (#378).
    #   MODEL_SEED — parameter initialization (the nnx.Rngs the trainer builds
    #                the model with).
    DATA_SEED: int = 42
    MODEL_SEED: int = 42

    # The training mixture (#439): which buckets under DATA_ROOT a run reads, and how
    # their weights move. One `bucket=start:end` per source, comma-separated, in mixer
    # order; the weights ramp linearly from the start column to the end column over the
    # mixture ramp, then hold. `bucket=w` holds w the whole run. Each column must sum to
    # 1 — a renormalization nobody wrote down is a hidden default on the hot path.
    # Parsed and checked by trm.train.schedules.parse_mixture.
    #
    # Chosen in the spec, before the run, never after a number is seen (CLAUDE.md rule
    # 3), so two arms that differ only here are a matched pair on the mixture. A pair
    # that informs a base run inherits that run's mixture.
    #
    # The default is the ramp every run so far trained with, web-heavy to a code/math
    # blend (#362), so an unset environment resolves bit-for-bit to what it always did.
    DATA_MIXTURE: str = DEFAULT_DATA_MIXTURE
    # Where the ramp ends, as a fraction of the LR schedule's decay horizon (#362): a
    # third of the way in, the shape the 4B champion trained with (10,000 of 30,518
    # steps). Budget-relative, so a short pair inherits the shape of the run it informs.
    MIXTURE_RAMP_FRACTION: float = 10000 / 30518

    # Branching a checkpoint onto another mixture (#489). An exact resume refuses a data
    # state written for a different DATA_MIXTURE (#439); a branch keeps the weights, the
    # optimizer and the step, and rebuilds the stream instead: every bucket the checkpoint
    # read continues where it stopped (nothing repeats), a bucket it never read starts at
    # its beginning, the mixer's draws restart from DATA_SEED, and every reader then skips
    # DATA_SEED x DATA_BRANCH_SEED_STRIDE rows. A branch starts every seed from the same
    # weights, so without that skip two seeds would read the same rows and their spread
    # would measure almost nothing. The stride is 80k rows (~82M tokens per bucket): more
    # than a bucket at 25% reads in a 2,048-step branch (65k rows), and small enough that
    # three seeds fit the 254M-token general-web bucket #489 reads. Off: a resume is exact.
    DATA_BRANCH: bool = False
    DATA_BRANCH_SEED_STRIDE: int = 80_000

    # ── Run cadence (in optimizer steps) ──────────────────────────────────────
    # Both fire at the opt-step boundary — NOT nested in the logging block (nesting
    # would multiply the interval by LOG_REAL_STEPS, the bug that hid the probe). The
    # full-state checkpoint save blocks the loop, so it must stay rare. Overridable for
    # one caller: the supervisor's fit gate (#168) sets both to 1, so a few-minute probe
    # crosses a validation pass and a checkpoint write — where every observed launch OOM
    # landed — instead of waiting 8,192 micro-steps for them.
    VAL_EVERY_OPT_STEPS: int = 64
    CHECKPOINT_EVERY_OPT_STEPS: int = 64
    # The per-corpus probes (#363) fire on validation steps that are also multiples of
    # this, so their readings share val_step. Rarer than the fineweb probe: each corpus
    # costs one more probe's worth of forward passes.
    VAL_BY_SOURCE_EVERY_OPT_STEPS: int = 128

    # Milestone checkpoints are kept at doubling token counts — 8M, 16M, 32M, … — so disk
    # grows with log(run length) instead of with it (#394). The horizon is absolute
    # tokens, not a fraction of a budget, because a run that stops on a criterion has
    # no budget to take a fraction of. The old fixed 500M cadence would have written
    # ~48 GB of full-state saves over a 10B-token run onto an SSD with 55 GB free, and
    # kept nothing at all inside a 67M-token pair.
    MILESTONE_FIRST_TOKENS: int = 8_000_000
    MILESTONE_RATIO: float = 2.0
    # The cap a runaway run stops at: 16 doublings from 8M is 262B tokens, far past
    # anything this card can train.
    MILESTONE_MAX_COUNT: int = 16

    # ── The cold tier (the supervisor's, #458; trm/runtime/cold.py) ───────────
    # With COLD_ROOT set, the SSD is kept at least this free by removing milestones
    # whose byte-identical copy is on the HDD; the owner's "margin of error, not a
    # rule" (2026-09-20). It sits well above the supervisor's mid-run KILLED_DISK
    # floor (the next checkpoint write plus 2GB, ~4.6GB at dim 960), so pruning acts
    # long before that stop would, and equals the launch precheck's min_free_gb.
    SSD_KEEP_FREE_GB: float = 12.0

    # ── f16 margin alarms (the supervisor's, #368) ────────────────────────────
    # Crossing one is an alarm, announced and recorded, never a kill: a margin is a
    # warning of the failure #235 found only after a 10-day run (the champion finished
    # at 65,120 of f16's 65,504), not the failure itself. Whether any of them should
    # stop a run is the owner's call.
    #   branch_max (each branch's f16 output, #536; the stream while it is f16) past a
    #   quarter of f16's ceiling: a block's output is climbing toward it.
    ACT_MAX_ALARM: float = 65504 / 4
    #   the loss scale at or below this: the backward overflows with no scaling left (#199).
    LOSS_SCALE_FLOOR_ALARM: float = 4.0
    #   the applied gradient's zero fraction at the underflow bar (#82's 0.05).
    ZERO_GRAD_ALARM: float = 0.05
    #   arena headroom under this: no room left for the next allocation spike.
    VRAM_HEADROOM_ALARM_MIB: float = 150.0

    # ── Milestone scoring (the supervisor's, #471) ────────────────────────────
    # CPU LAMBADA scorers alive at once on a run, a predecessor supervisor's counted
    # through their claims (#506); the rest wait for a later poll. One scorer is
    # ~1.5 GB of RAM and ~40 min of CPU on a 15 GB box the trainer also needs; nine
    # at once is the OOM that killed a run (#471).
    MILESTONE_SCORERS: PositiveInt = 1


# Where things are, not how the run trains: machine-local paths, usually from the
# repo's .env, which the entry points load at run time (after this module's import),
# so they are read when asked for rather than frozen into CONFIG.
# COLD_ROOT is the HDD directory runs mirror to (trm/runtime/cold.py); unset, nothing
# leaves the SSD.
LOCATIONS = ("DATA_ROOT", "CHECKPOINT_ROOT", "COLD_ROOT")


def location(name, default=None, environ=os.environ):
    """A path from the environment, read now; `default` when unset."""
    assert name in LOCATIONS, f"{name} is not a location; a knob belongs on Config"
    return environ.get(name, default)


# The checkout this code runs from: its .env is the only one ever read (#541).
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def env_file(root=REPO_ROOT):
    """`root/.env` as a dict, a relative location resolved against `root`; {} without one.

    Never searches upward: python-dotenv's default walks to parent directories, so a
    worktree nested inside the repo picked up the root's .env and resolved its relative
    DATA_ROOT inside the worktree, where there is no corpus (#541)."""
    from dotenv import dotenv_values  # here, not at the top: the jaxfree CI jobs lack it
    values = {k: v for k, v in dotenv_values(pathlib.Path(root) / ".env").items() if v is not None}
    for name in LOCATIONS:
        value = values.get(name)
        if value and "://" not in value:
            values[name] = str((pathlib.Path(root) / value).resolve())
    return values


def load_env(root=REPO_ROOT, environ=os.environ):
    """Load `env_file(root)` into the environment; a value already set there wins."""
    for name, value in env_file(root).items():
        environ.setdefault(name, value)


# The launching process's Config, read when this module is first imported. Only entry
# points read it (the trainer's start, the supervisor, launch, rewind, infer, prefill,
# and trm/config.py for the compute dtype) and hand it down; library code takes the
# Config it is given (#475, tests/core/test_config_read_once.py).
CONFIG = Config.from_env()
