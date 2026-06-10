# Refactor Plan — Code Quality Remediation

Date: 2026-06-09. Branch context: `feat/curriculum-learning`.

This plan addresses every issue found in the full-repo review. The root cause of most of
them is the same: features were patched on top of each other as the model grew in
sophistication, without ever establishing repo-wide patterns. So this plan does two things:
first it states the patterns, then it maps every concrete issue to a fix that conforms to
them. The model architecture itself is NOT being changed — this is engineering remediation
only.

Hardware constraint that shapes several decisions: training runs on an RTX 2060 (6 GB VRAM,
Turing). Turing has no bfloat16 support, so float16 compute is a deliberate, permanent
policy for this repo — the fix is to make that policy explicit and centralized, not to
change it.

---

## Part 1 — Patterns to establish (the rules everything else follows)

1. **One source of truth for configuration.** All architecture, training, and dtype
   constants live in one `config.py`. No module ever imports config from `layers.py`.
   Experiment knobs (loss weights, curriculum breakpoints) are named constants there or in
   `schedules.py` — never inline literals at the point of use.

2. **Modules are named for their responsibility, and have exactly one.** A reader
   navigating by filename must land in the right place on the first try.

3. **Fail loudly.** No bare `except:`, no `except Exception: pass`, no silently zeroing
   NaN losses. Anything that swallows an error must log what it swallowed and why that is
   safe. Diagnostics that go NaN must crash or warn, never get written to CSV for 1000
   steps unnoticed.

4. **One concept, one name.** A flag or quantity keeps the same name across every layer it
   passes through (data loader → train loop → grad step → model).

5. **Every loss term is provably wired.** The forget-cost incident (the cost was computed
   and logged but never added to the total loss, so the model never learned to forget)
   must be impossible to repeat: a regression test asserts that each loss component
   actually influences the total loss gradient.

6. **Comments explain why, and die with the code they describe.** The existing
   stop-gradient and ACT-accumulation comments are the standard to keep. Contradictory or
   stale comments are treated as bugs.

7. **Derived numbers are computed, not transcribed.** Parameter counts, byte footprints,
   and dataset names are derived from the actual model/config, so they cannot drift.

8. **The environment is reproducible.** Pinned dependencies, recorded seeds, and resume
   behavior that does not corrupt logs.

---

## Part 2 — Target file layout

```
config.py            # all constants: arch dims, training, PAD token, dtype policy
layers.py            # nn modules only (RotaryAttention, blocks, BlockStack)
losses.py            # pure loss fns: slot stability (InfoNCE), CE-with-mask helper
model.py             # UniversalReasoner (unchanged math)
schedules.py         # optax schedules + curriculum weight/step functions + loss weights
data_loaders.py      # TextDataGenerator, DataMixer (fixed API, see B1)
grad_step.py         # jitted compute_grad_step / apply_grads (was train_local.py)
trainer.py           # train loop + optimizer factories (extracted from start_training.py)
monitor.py           # LossMonitor (moved out of metrics_logger.py)
metrics_logger.py    # MetricsLogger only (CSV + console)
checkpoint_utils.py  # unchanged role
run_tracker.py       # unchanged role
start_training.py    # CLI entry point ONLY: argparse + wiring + run/resume resolution
prefill.py           # decomposed (see C5)
plot_history.py      # plotting; param stats derived from the real model (see B4)
infer_local.py       # CLI inference (KV-cache decision, see B3)
tests/               # real unit tests (pytest); data-curation scripts move to tools/
tools/data_curation/ # the existing inspection scripts, renamed out of tests/
pyproject.toml       # project metadata, pinned deps, pytest + ruff config
```

The repo stays flat — at ~2.4k lines a package hierarchy would be ceremony. The win is
that names stop lying.

---

## Part 3 — Issue-by-issue fixes

### A. Structure & organization

**A1. Config constants live in `layers.py`.**
Create `config.py` holding: `LATENT_DIM`, `NUM_BLOCKS`, `SHARED_SLOTS`, `MAX_SEQ_LEN`,
`VOCAB_SIZE`, `MAX_STEPS_LIMIT`, `BATCH_SIZE`, `ACCUMULATION_STEPS`, `PAD_TOKEN_ID`,
`NUM_HEADS`, `NUM_GROUPS`, plus the new dtype policy (A2). All imports updated.
`layers.py` keeps zero config definitions. This fixes the inverted dependency where
`run_tracker` and `plot_history` import a neural-net module just to read constants.

**A2. Dtype policy is implicit and the `dtype` argument lies.**
Add to `config.py`:
- `COMPUTE_DTYPE = jnp.float16` with a comment stating the hardware reason (RTX 2060 /
  Turing — no bf16; f16 compute is the deliberate policy, params stay f32).
- `PARAM_DTYPE = jnp.float32`.
`RotaryAttention` and the MLP projections use `COMPUTE_DTYPE` instead of hardcoded
`jnp.float16`. The ignored `dtype` parameter on `RotaryAttention.__init__` is removed
(a parameter that is accepted and ignored is worse than no parameter). Norms stay f32
explicitly. Note for the future: if f16 underflow in gradients ever becomes a problem,
the fix is optax loss scaling — documented here so it is not rediscovered.

**A3. Misleading module names / `start_training.py` mixes five concerns.**
- `train_local.py` → `grad_step.py` (it contains the jitted gradient step, nothing else).
- Train loop, `init_model_and_optimizer`, `create_sft_optimizer`, and the data-pipeline
  thread move from `start_training.py` to `trainer.py`.
- Curriculum functions (`get_curriculum_weights`, `get_average_curriculum_weights`,
  `get_curriculum_steps`) move to `schedules.py` — that is where a reader looks for
  anything that varies with training step.
- `start_training.py` shrinks to: argparse, run/resume discovery, wiring, and the call
  into `trainer.train_loop`. Target ≤ ~120 lines.
- `LossMonitor` moves to `monitor.py`; `metrics_logger.py` keeps only `MetricsLogger`.

**A4. `tests/` contains no tests; `__init__.py` ambiguity.**
- Move `tests/data_curation/*` to `tools/data_curation/` — they are inspection scripts,
  not tests, and their current location makes "we have tests" look true when it is not.
- Delete the empty root `__init__.py` (the repo is a script collection, not a package).
- Add real tests (see D1).

### B. Correctness & reliability

**B1. Curriculum/DataMixer weight-length mismatch (real bug).**
`data_wrapper` assigns a fresh length-3 weight list to `pretrain_mixer.weights` every
batch, while `DataMixer` removes exhausted sources and renormalizes to fewer entries.
After any source exhausts, `zip(sources, counts)` silently truncates and draws can return
undersized or empty batches.
Fix: stop mutating mixer internals from outside. `DataMixer.get_batch` gains an optional
`weights=` argument; the mixer maps incoming weights to its *surviving* sources by index
(it remembers which original indices are still alive) and renormalizes internally. A unit
test covers: 3 sources, middle one exhausts, curriculum weights keep arriving, batches
stay correctly sized and the dead source's probability mass is redistributed.

**B2. NaN masking hides divergence; silent exception swallowing.**
- Remove `total_loss = jnp.where(isfinite, total_loss, 0.0)`. Replace with explicit
  handling in the train loop: if the returned loss is non-finite, log a loud warning with
  the step number, skip the optimizer update for that micro-step, and count consecutive
  non-finite steps — abort the run after a threshold (e.g. 50). Divergence becomes a
  visible event, not a dip in the loss curve.
- `metrics_logger.py` bare `except:` → catch the specific filesystem errors and print
  the exception.
- `run_tracker.py` three `except Exception: pass` blocks → keep the "metadata must never
  kill training" intent, but print a one-line warning with the exception so failures are
  visible. `_check_compatibility`'s blanket pass is narrowed to JSON/IO errors only — a
  bug in the compatibility check itself must not silently disable the check.
- `checkpoint_utils.discover_latest_checkpoint_run` `except Exception: pass` → same
  treatment: warn and continue scanning.

**B3. Dead code: KV cache, `effective_is_causal`, dropped accumulators, unused imports.**
Decision: **delete the KV-cache machinery** (`k_cache`, `v_cache`, `cache_index`,
`reset_state`, the `use_cache` parameter threaded through every signature). Rationale:
it has never been exercised (`use_cache=True` appears nowhere), so it is untested code
pretending to be a feature, and this architecture re-runs the reasoning loop over slots
every token anyway — a correct incremental-decode path is a real feature with its own
design work (hunch-cache interaction, slot positions), not a flag flip. It belongs in
`docs/PLAN.md` as future work, implemented fresh when needed. Removing it also deletes
the `reset_state()` calls at the top of `model.__call__`.
Also delete: the `effective_is_causal` variable (always False — `jax.nn.dot_product_attention`
is always called with explicit masks), `accum_forget_cost` (accumulated, never logged),
the unused `checkpoint_run_id_from_meta` unpacking, and unused imports (`json`,
`datetime`, `subprocess`, `sys` in `start_training.py`; `LATENT_DIM` in `model.py`).
`load_or_create_checkpoint` stops returning `optimizer` (it returns it unmodified) and
stops returning the unused run id — the return shape matches what callers use.

**B4. Hand-transcribed parameter math in `plot_history.py` (and it is already wrong).**
The byte-footprint math assumes projection weights are stored in f16; in Flax NNX,
`dtype=` is the *computation* dtype — params are stored f32. So the printed VRAM estimate
is currently wrong, which is the inevitable end state of transcribed math.
Fix: `print_model_stats` builds the real model (`UniversalReasoner(LATENT_DIM,
nnx.Rngs(0))` on CPU via `jax.default_device`), walks `nnx.state(model)` leaves, and sums
counts/bytes from actual shapes and dtypes, grouped by path prefix (encoder / decoder /
reasoning / embeddings / heads). The duplicated `hidden_dim` formula disappears. The
hand-math version is deleted, not commented out.

**B5. `temporal_drift` NaN for the entire 1-step curriculum phase (found in
`runs/run_20260604_011134/metrics.csv`).**
Cause: with `max_steps=1` the trajectory diff `states[1:] - states[:-1]` is empty and the
mean of an empty array is NaN.
Fix: in `model.py`, guard the drift computation — when the trajectory has fewer than two
steps, report 0.0. Additionally (pattern 3), `MetricsLogger.log` checks every diagnostic
value with `math.isfinite` and prints a loud `⚠️ non-finite metric: <name>` the first
time one appears, so a broken diagnostic can never again ship silently for 1000 steps.

**B6. `metrics.csv` accumulates overlapping step ranges on resume (found in the same run:
step regressions like 3905 → 3595).**
Cause: checkpoints restore to the last *best* step, but the CSV already has rows beyond
it; every resume appends a replayed range. Plots silently double-draw those segments.
Fix: on resume, `MetricsLogger.__init__` receives `start_step`; if the CSV's last row has
`step >= start_step`, it rewrites the file truncated to rows `< start_step` before
appending. The plotter additionally drops non-monotonic rows defensively (existing CSVs
like this run's stay plottable). A note is appended to `docs/CHANGES.md` since historical
CSVs contain these artifacts.

**B7. Hidden coupling: `hunch_cache` sized by global `BATCH_SIZE`; undeclared
`sft_start_step`.**
- `UniversalReasoner.__init__` gains an explicit `batch_size` parameter used to size
  `hunch_cache`, defaulting to `config.BATCH_SIZE`. The coupling still exists (the cache
  is stateful by design) but becomes visible at the constructor instead of hidden in a
  global. A shape assert in `__call__` produces a clear error if tokens don't match the
  cache batch size, instead of a cryptic broadcast failure.
- `LossMonitor.__init__` declares `self.sft_start_step = None`. All
  `getattr(monitor, "sft_start_step", None)` guards become plain attribute access.

**B8. Loss-wiring regression test (the forget_cost incident).**
New test: build a tiny model (dim 32, 2 slots, vocab 64), run `compute_grad_step`'s loss
function, and for each loss component (forget cost, diversity, ponder, refinement) assert
that perturbing its lambda changes the total loss when the component is nonzero. This
makes "computed and logged but never added to the loss" a CI failure instead of something
discovered mid-run from a flat forgetting curve.

### C. Readability & naming

**C1. Contradictory remat comments.**
`layers.py` says the reasoning stack should not double-checkpoint inside
`jax.checkpoint(scan_step)`; `model.py` passes `use_remat=True` for it and argues the
opposite. Resolution: the code's current behavior (remat everywhere) is what trained the
existing runs and fits in 6 GB, so the code stands and the *stale comment in `layers.py`
is rewritten* to describe reality: every stack uses per-block remat; the reasoning stack
is additionally inside the scan-level checkpoint, trading recompute FLOPs for fitting the
backward pass in 6 GB. If a future memory benchmark shows the double checkpoint is
wasteful, that experiment goes in `docs/` first.

**C2. One name per concept for the document-boundary flag.**
`reset_mask` → `should_truncate` → `should_refresh` is one concept with three names.
Unified name: `doc_boundary` from the data loader through the queue and grad step;
`should_refresh` is kept only inside `model.__call__` where the meaning genuinely shifts
(boundary → "start from fresh slots"). The grad-step parameter is renamed accordingly.

**C3. Misleading strings and comments.**
- `"Memory-mapping ... into VRAM"` → `"Memory-mapping ... (lazy host-RAM paging)"`.
- `prefill.py` `"Use 75% of cores"` comment → matches the actual `cpu_count() - 1`.
- `create_sft_optimizer` `"10x LR penalty"` → `"reducing LR to 10% for SFT"`.
- README dataset list: "Python-Edu" → "CodeParrot-clean" (match `prefill.MIXTURE`).
  README is also updated for every rename in this plan (file names, commands).

**C4. Inline magic numbers and the import inside the jitted loss.**
- `0.08` (refinement weight) and `0.03` (anchor CE weight) become
  `REFINEMENT_LOSS_WEIGHT` and `ANCHOR_CE_WEIGHT` in `schedules.py`, next to the lambda
  schedules they belong with.
- Curriculum breakpoints (`1000/4000/8000`, `CURRICULUM_STEPS = 10000`) and the weight
  ramp endpoints become named module-level constants in `schedules.py` when the
  curriculum functions move there (A3).
- `from schedules import ...` moves from inside `loss_fn` to the top of `grad_step.py`.
- The repeated `1e9` mask constant becomes `MASK_NEG = -1e9` (one definition, used as
  `(mask - 1) * -MASK_NEG`-style helpers) in `layers.py` with a one-line helper
  `padding_bias(mask)` replacing the four copies of
  `(mask.astype(f32) - 1.0) * 1e9` + reshape.

**C5. `prefill.py` is a 220-line monolith.**
Decompose `run_prefill` into focused units, same behavior:
- `load_progress(save_path)` / `save_progress(...)` — status.json + chunk-recovery scan.
- `extract_text(item, dataset_alias)` — the per-dataset field/format logic (fineweb score
  filter, ultrachat message flattening, fallbacks).
- `stream_with_retries(ds_cfg, start_offset, queue, stop_event)` — the producer thread
  with reconnect logic.
- `write_chunk(token_acc, save_path, file_idx)` — stride-aligned chunk save.
- `run_prefill()` becomes the ~50-line orchestration loop.
`extract_text` gets unit tests (it is pure and encodes real, easy-to-break dataset
assumptions).

### D. Tooling, tests, reproducibility

**D1. Real test suite (pytest).** Initial set, all CPU-fast:
- `test_curriculum.py` — `get_curriculum_weights` sums to 1 and hits documented
  endpoints; `get_average_curriculum_weights` is consistent with the running average of
  `get_curriculum_weights` (numerically, against a brute-force mean); `get_curriculum_steps`
  boundaries.
- `test_data_loaders.py` — `TextDataGenerator` over tmpdir .npy files: batch shapes,
  doc-boundary mask on file rollover, exact `skip_count` resume; `DataMixer` exhaustion +
  external-weights redistribution (B1).
- `test_losses.py` — slot stability loss: shape, zero for identical normalized states up
  to the InfoNCE floor, decreases as slots decorrelate.
- `test_model_smoke.py` — tiny-config forward pass: logits shape, finite outputs,
  `should_refresh` true/false both run, drift guard at max_steps=1 returns 0 not NaN (B5).
- `test_loss_wiring.py` — B8.
- `test_monitor.py` — LossMonitor plateau detection and best-tracking.

**D2. Packaging and pins.**
Add `pyproject.toml` with project metadata and tool config (pytest paths, ruff line
length + rules). Pin exact working versions in `requirements.txt` (read them from the
current venv with `pip freeze` filtered to direct deps) — JAX/Flax NNX API churn makes
unpinned installs a guaranteed future breakage. Add `ruff` as the single lint/format
tool; run it once over the repo and commit the (mechanical) result separately from any
logic change.

**D3. Seeding.**
`TextDataGenerator`'s random start-offset augmentation uses an explicit
`np.random.default_rng(seed)` passed in from config, and the seed is recorded in
`run_metadata.json` by `RunTracker.get_hyperparameters`. Resume token-accounting stays
approximate (that is inherent to the averaged-curriculum estimate and is now stated in a
comment where `skip_count` is computed), but the randomness itself becomes reproducible.

---

## Part 4 — Execution order

Each phase leaves the repo runnable; nothing in a later phase blocks training between
phases. Verification for every phase: `python -c "import trainer, grad_step, model"`-level
smoke + the test suite once it exists + a short real training smoke run (a few hundred
micro-steps) before merging the branch.

1. **Phase 1 — config + renames (pure motion, no logic).** A1, A3, A4 moves, C2 renames,
   C3 strings, README updates. Done with `git mv` where possible so history follows.
2. **Phase 2 — delete dead code.** B3, plus the C1 comment fix (same files).
3. **Phase 3 — correctness fixes.** B1 (mixer API), B2 (fail-loudly), B5 (drift guard +
   metric finiteness check), B6 (CSV resume truncation), B7 (constructor batch size,
   monitor attribute), C4 (constants + import hoist), A2 (dtype policy).
4. **Phase 4 — tests + tooling.** D1, D2, D3, B8. From here on, CI-by-habit: run pytest
   before every commit.
5. **Phase 5 — derived stats + prefill decomposition.** B4, C5. These touch code that is
   not on the training hot path, so they go last.

Phases 1–3 are the "one focused refactor day". Phases 4–5 are the second day.

## Part 5 — Acceptance criteria

- Zero `from layers import` lines that import a constant.
- `grep -rn "except:" "except Exception: pass"` returns nothing unexplained.
- `pytest` green; loss-wiring test fails if any cost term is dropped from the total loss.
- A fresh `pip install -r requirements.txt` into a clean venv runs the smoke test.
- A resume produces a `metrics.csv` with strictly increasing steps.
- `plot_history.py` prints parameter/byte stats computed from real `nnx.state` shapes.
- A reader can answer "where is the train loop / the loss / the curriculum / the config"
  correctly from filenames alone.

---

## Part 6 — Decisions & status log (append-only)

**2026-06-09 — scope decisions after user review:**
- B5 framing corrected: temporal_drift NaN was not a broken computation — the metric
  (state movement between consecutive reasoning steps) is genuinely *undefined* for a
  1-step trajectory. Only the handling is fixed: report 0.0 when there are fewer than two
  steps, and the logger warns loudly on any non-finite metric.
- D1 test suite trimmed by user decision: only two tests are kept, chosen because each
  guards a bug this repo actually produced — `test_loss_wiring.py` (the forget_cost
  incident) and the DataMixer weight-redistribution test (B1). The broader suite is
  dropped. (`tests/data_curation/` was created by Google Antigravity, not by intent.)
- The docs-folder placement of this plan follows the existing `AGENTS.md` rule (all .md
  files except auxmd/README/AGENTS live in `docs/`, append-only).

**2026-06-09 — Phase 1 complete.** `config.py` created; `train_local.py` → `grad_step.py`;
`trainer.py` extracted (train loop, optimizer factories, data pipeline); curriculum
functions moved to `schedules.py`; `LossMonitor` → `monitor.py` (with `sft_start_step`
declared); `start_training.py` reduced to CLI wiring (~100 lines); `tests/data_curation/`
→ `tools/data_curation/`; root `__init__.py` removed; `doc_boundary` naming unified;
misleading strings fixed (VRAM print, cores comment, SFT LR message, README dataset
name). Verified: all modules import, and a real `compute_grad_step` on the GPU returns a
finite loss/grad-norm.
- New observation for Phase 3: `os.path.abspath(DATA_ROOT)` mangles `gs://` URLs
  (prepends cwd, collapses `//`). Harmless for local paths, but the `.env.example`
  advertises a GCS path that would break. Fix alongside B-items: only abspath when the
  path has no URL scheme.

**2026-06-09 — Phase 2 complete.** Deleted: KV-cache machinery in `RotaryAttention`
(`k_cache`/`v_cache`/`cache_index`, `reset_state`, the `use_cache` parameter threaded
through every attention signature, and the three `reset_state()` calls in
`model.__call__`); the always-False `effective_is_causal`; the never-logged
`accum_forget_cost` in the train loop; unused `jnp` import in `checkpoint_utils`.
`load_or_create_checkpoint` now returns `(mngr, monitor, start_step)` — it no longer
returns the optimizer it never modified, nor a run id nobody read. The stale remat
comment in `layers.py` was rewritten to describe actual behavior (per-block remat
everywhere; the reasoning stack is additionally inside the scan-level checkpoint, a
deliberate FLOPs-for-VRAM trade on 6 GB). Incremental decoding is logged as designed
future work in `docs/PLAN.md`.
- **Breaking note:** removing the `nnx.Cache` variables changes the model's state tree,
  so checkpoints saved before Phase 2 (e.g. `run_20260604_011134`) are no longer
  restorable. Accepted: that run trained on code with the unwired forget cost and is
  superseded.
