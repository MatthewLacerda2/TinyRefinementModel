"""What a training run is built from: the model and its optimizer, and the data
pipeline that feeds the loop (the loop itself is loop.py). Held-out scoring lives in
validation.py, the optimizer chains in optimizers.py, schedules and mixture policies
in schedules.py.

Every knob read here comes from the Config it is handed (#475):
trm.train.start builds it, with a resumed run's own budget, and passes it down."""

import queue
import threading

import jax
import numpy as np
from flax import nnx

from trm.config import resolve_root
from trm.data.loaders import DataMixer, TextDataGenerator
from trm.model import build_model
from trm.settings import DEFAULT_DATA_MIXTURE, load_env, location
from trm.train.optimizers import optimizer_chain
from trm.train.schedules import Schedules, mixture_label, parse_mixture

load_env()

PREFETCH_SIZE = 128

DATA_ROOT = location("DATA_ROOT", "")
if DATA_ROOT:
    DATA_ROOT = resolve_root(DATA_ROOT)


def _param_count(model):
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))


def split_samples(total_samples, weights):
    """Split a sample count across sources by mixture weight.

    The unit that resume correctness hangs on: TextDataGenerator's skip_count is
    in SAMPLES, while every step counter in this trainer is in MICRO-STEPS, and
    each micro-step draws BATCH_SIZE samples across the mixer (#24). Callers pass
    the SAMPLE total — read from the checkpoint, not re-derived from steps — so
    a run that resumes under a different batch size than it was trained at still
    seeks to the right place. Getting this wrong is silent either way: too small
    and the model re-trains on data it already saw, too large and it skips a
    slice of corpus it never read. No crash, no warning, just a bad run.

    Per-source truncation is deliberate: skip_count is an integer sample offset,
    so the total can fall short by at most one sample per source.
    """
    return [int(total_samples * w) for w in weights]


def samples_from_micro_steps(micro_steps, weights, batch_size):
    """split_samples for the case with no recorded sample count — a pre-#24
    checkpoint, or a fresh run. Converts micro-steps at the given batch size."""
    return split_samples(micro_steps * batch_size, weights)


def init_model_and_optimizer(config):
    print(f"🚀 Initializing PlainTransformer (Dim={config.LATENT_DIM}, layers={config.PLAIN_LAYERS})...")
    model = build_model(config, nnx.Rngs(config.MODEL_SEED))

    print(f"📐 {_param_count(model) / 1e6:.1f}M parameters "
          f"(MODEL_SEED={config.MODEL_SEED}, DATA_SEED={config.DATA_SEED})")
    schedules = Schedules.of(config)
    # The resolved LR horizon must be visible at launch (#83): an anneal that
    # bottoms out before the budget ends is undertraining masquerading as an
    # architecture problem.
    budget = config.TRAIN_TOKEN_BUDGET
    budget_note = (f"TRAIN_TOKEN_BUDGET={budget:,}" if budget is not None
                   else "TRAIN_TOKEN_BUDGET unset — historical default")
    print(f"🗓️ LR horizon: DECAY_STEPS={schedules.decay_steps:,} opt steps "
          f"(warmup {config.WARMUP_STEPS:,}) ≈ "
          f"{schedules.decay_steps * config.TOKENS_PER_OPT_STEP / 1e9:.2f}B "
          f"target tokens ({budget_note})")
    # Every schedule with its horizon and what it scales with (#362).
    print("🗓️ Schedules: " + " | ".join(
        f"{name} {steps:,} opt steps ({kind})" for name, (kind, steps) in schedules.horizons.items()))
    # The mixture at both ends of its ramp, by bucket (#439): what this run reads.
    print(f"🥣 Mixture: {mixture_label(schedules.sources, schedules.start_weights)} -> "
          f"{mixture_label(schedules.sources, schedules.end_weights)}")
    # The whole optimizer at launch, every knob named (#358).
    muon = (f"muon on the matrices (LR x{config.MUON_LR_MULT:g}, beta {config.MUON_BETA:g}, "
            f"{config.MUON_NS_STEPS} Newton-Schulz steps, {config.MUON_NS_COEFFS}), adamw on the rest"
            if config.TRM_OPTIMIZER == "muon" else "adamw")
    print(f"🎛️ Optimizer: {muon} | adam b1 {config.ADAM_B1:g} b2 {config.ADAM_B2:g} "
          f"eps {config.ADAM_EPS:g} | weight decay {config.WEIGHT_DECAY:g} x LR"
          + (f", embedding {config.EMBED_WEIGHT_DECAY:g}/step at peak" if config.TRM_OPTIMIZER == "muon" else "")
          + f" | clip {config.CLIP_NORM:g}")
    # What the loss and the stream are, beside the optimizer (#357, #369).
    print(f"🧮 Loss: z-loss {config.Z_LOSS_WEIGHT:g} x log^2 Z | residual stream "
          f"{config.RESIDUAL_DTYPE}")
    optimizer = nnx.Optimizer(model, optimizer_chain(config, schedules.learning_rate), wrt=nnx.Param)  # pyright: ignore[reportArgumentType] -- duck-typed: MultiSteps has init/update but is no GradientTransformation

    return model, optimizer

def setup_data_pipeline(config, start_step, samples_seen=None, data_state=None):
    # Warned here, where data is first needed, not at import: every importer of this
    # module (instruments that never load data included) used to print it.
    if not DATA_ROOT:
        print("⚠️ Warning: DATA_ROOT is not set. Data loading will fail unless provided via environment.")
    print("🚀 Initializing Dynamic Data Phases...")
    schedules = Schedules.of(config)
    # Every reader and the mixer seeded from DATA_SEED, each its own stream.
    pretrain_sources = [TextDataGenerator(f"{DATA_ROOT}/{path}", max_seq_len=config.MAX_SEQ_LEN,
                                          rng=np.random.default_rng(config.DATA_SEED))
                        for path in schedules.sources]
    pretrain_mixer = DataMixer(pretrain_sources, schedules.start_weights,
                               rng=np.random.default_rng(config.DATA_SEED), names=schedules.sources)

    if config.DATA_BRANCH and start_step == 0:
        # A branch that found no checkpoint would train a fresh model on the arm's
        # mixture and report it as the branch.
        raise SystemExit("DATA_BRANCH=1 but no checkpoint was restored: a branch starts from one (#489)")
    if start_step > 0 and config.DATA_BRANCH:
        # A branch onto this run's mixture (#489): the weights continue, the stream
        # is rebuilt, and each seed reads its own rows.
        if data_state is None:
            raise SystemExit("DATA_BRANCH=1 needs a checkpoint that saved its data state (#424)")
        pretrain_mixer.branch_state(data_state, parse_mixture(DEFAULT_DATA_MIXTURE)[0],
                                    skip=config.DATA_SEED * config.DATA_BRANCH_SEED_STRIDE)
        print(f"🌿 Data stream branched onto {', '.join(schedules.sources)}: seed {config.DATA_SEED} "
              f"skips {config.DATA_SEED * config.DATA_BRANCH_SEED_STRIDE:,} rows per bucket (#489)")
    elif start_step > 0 and data_state is not None:
        # Exact (#424): the reader and mixer state saved with the last batch the
        # checkpointed run consumed, so the next row is the one it would have read.
        pretrain_mixer.load_state(data_state)
        print("📍 Data stream restored exactly from the checkpoint (#424)")
    elif start_step > 0:
        # A checkpoint from before #424: the position is estimated, and the stream
        # after it is not the one the run would have read.
        print("⚠️ Data position estimated from the sample count: this checkpoint predates "
              "the saved data state (#424)")
        start_opt_step = start_step // config.ACCUMULATION_STEPS
        # Prefer the recorded sample count over re-deriving it from micro-steps
        # (#24): only the recorded figure survives a change in BATCH_SIZE between
        # the run that wrote the checkpoint and the one resuming it.
        avg_weights = schedules.average_curriculum_weights(start_opt_step)
        skips = (split_samples(samples_seen, avg_weights) if samples_seen is not None
                 # Pre-#24 checkpoints come from runs that counted from 1 (#355).
                 else samples_from_micro_steps(start_step - 1, avg_weights, config.BATCH_SIZE))
        for gen, skip in zip(pretrain_sources, skips, strict=True):
            gen.skip_count = skip

    data_queue = queue.Queue(maxsize=PREFETCH_SIZE)

    def data_wrapper():
        loader_step = start_step
        while True:
            loader_opt_step = loader_step // config.ACCUMULATION_STEPS
            pretrain_mixer.set_weights(schedules.curriculum_weights(loader_opt_step))
            batch = pretrain_mixer.get_batch(config.BATCH_SIZE)

            if batch is None:
                data_queue.put((None, None, None))
                break

            # The state AFTER this batch travels with it, so the trainer can save
            # the one for the last batch it actually consumed, not the prefetched ones.
            data_queue.put((batch, pretrain_mixer.last_source, pretrain_mixer.state()))
            loader_step += 1

    threading.Thread(target=data_wrapper, daemon=True).start()
    return data_queue
