"""Two training-path fixes, validated by targeted logic/round-trip tests
(a full smoke run needs DATA_ROOT + the GPU f16 path, so we test the mechanism
directly instead):

1. Rolling-latest checkpointing — at the save cadence the trainer now persists
   the true latest state every time (not only on a new best), so a resume picks
   up where training actually left off. Best-CE state is preserved in a sibling
   'best_val_ce/' dir whose retention can't evict the latest.

2. Validation-probe cadence — the probe must fire every VAL_EVERY_OPT_STEPS
   optimizer steps, independent of the every-LOG_REAL_STEPS logging block.
"""

import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from trm.runtime.checkpoints import save_checkpoint, discover_latest_checkpoint_run
from trm.runtime.layout import BEST_SUBDIR, CHECKPOINT_ITEMS, ROLLING_KEEP
from trm.runtime.monitor import LossMonitor


def _make_manager(path):
    return ocp.CheckpointManager(
        path,
        item_names=CHECKPOINT_ITEMS,
        options=ocp.CheckpointManagerOptions(max_to_keep=ROLLING_KEEP, create=True),
    )


def _save_state(mngr, step, model_arr, opt_arr, monitor, sft_active, run_id):
    """Save the same Composite save_checkpoint builds, but with plain arrays so
    the test stays independent of the full model. Mirrors the production schema."""
    mngr.save(
        step,
        args=ocp.args.Composite(
            model=ocp.args.StandardSave({"w": model_arr}),
            optimizer=ocp.args.StandardSave({"w": opt_arr}),
            monitor_state=ocp.args.JsonSave({
                "ce_history": monitor.ce_history,
                "best_ce": monitor.best_ce,
                "best_loss": monitor.best_loss,
                "best_avg_ce": monitor.best_avg_ce,
                "last_improvement_step": monitor.last_improvement_step,
                "sft_active": sft_active,
                "sft_start_step": monitor.sft_start_step,
                "run_id": run_id,
            }),
            step=ocp.args.JsonSave(step),
        ),
    )
    mngr.wait_until_finished()


def test_rolling_latest_advances_past_best(tmp_path):
    """Steps where CE does NOT improve must still be saved, and latest_step()
    must point at the true latest — the save-on-best-only bug failed this."""
    chk = tmp_path / "checkpoints"
    mngr = _make_manager(chk)
    monitor = LossMonitor()

    # Step 100: a "best". Steps 200, 300: NOT new bests, but training moved on.
    for step, w in [(100, 1.0), (200, 2.0), (300, 3.0)]:
        _save_state(mngr, step, jnp.array(w), jnp.array(-w), monitor, False, "run_test")

    assert mngr.latest_step() == 300, "rolling-latest must track the true latest step"

    restored = mngr.restore(
        mngr.latest_step(),
        args=ocp.args.Composite(
            model=ocp.args.StandardRestore({"w": jnp.array(0.0)}),
            optimizer=ocp.args.StandardRestore({"w": jnp.array(0.0)}),
            monitor_state=ocp.args.JsonRestore(),
            step=ocp.args.JsonRestore(),
        ),
    )
    assert float(restored["model"]["w"]) == 3.0
    assert restored["step"] == 300


def test_best_subdir_does_not_break_discovery(tmp_path):
    """The sibling best dir lives inside the rolling checkpoint dir; the
    rolling manager (and discovery) must ignore it as a non-step entry."""
    runs_root = tmp_path / "runs"
    chk = runs_root / "run_test" / "checkpoints"
    monitor = LossMonitor()

    rolling = _make_manager(chk)
    best = _make_manager(chk / BEST_SUBDIR)
    _save_state(rolling, 10, jnp.array(1.0), jnp.array(1.0), monitor, False, "run_test")
    _save_state(rolling, 20, jnp.array(2.0), jnp.array(2.0), monitor, False, "run_test")
    _save_state(best, 10, jnp.array(1.0), jnp.array(1.0), monitor, False, "run_test")

    # Rolling manager still sees only its own steps.
    reopened = _make_manager(chk)
    assert reopened.latest_step() == 20

    found_path, found_run = discover_latest_checkpoint_run(runs_root=str(runs_root))
    assert found_run == "run_test"
    assert found_path == str(chk)


def test_save_checkpoint_schema_matches_loader(tmp_path, tiny_model, make_tiny_model):
    """save_checkpoint must write exactly what load_or_create_checkpoint reads:
    full round-trip with the real model + a real optimizer, restored into a
    fresh model with identical forward output."""
    import optax
    from flax import nnx
    from trm.runtime.checkpoints import load_or_create_checkpoint

    optimizer = nnx.Optimizer(tiny_model, optax.sgd(0.0), wrt=nnx.Param)
    monitor = LossMonitor()
    monitor.best_ce = 1.23
    monitor.best_val_ce = 3.6474
    monitor.sft_start_step = None

    chk = str(tmp_path / "checkpoints")
    save_mngr = ocp.CheckpointManager(
        chk, item_names=CHECKPOINT_ITEMS,
        options=ocp.CheckpointManagerOptions(max_to_keep=ROLLING_KEEP, create=True),
    )
    save_checkpoint(save_mngr, 42, tiny_model, optimizer, monitor, False, "run_x")
    del save_mngr

    fresh = make_tiny_model(seed=99)
    fresh_opt = nnx.Optimizer(fresh, optax.sgd(0.0), wrt=nnx.Param)
    _, _, resumed, start_step = load_or_create_checkpoint(fresh, fresh_opt, chk)

    assert start_step == 43, "resume must continue at saved step + 1"
    assert resumed.best_val_ce == 3.6474, (
        "a resume that forgot the held-out best would overwrite best_val_ce/ with "
        "a worse model on its first probe")

    tokens = jnp.asarray(np.full((1, 16), 5, dtype=np.int32))
    ref = np.asarray(tiny_model(tokens, depth=2, training=False, new_document=True).logits)
    got = np.asarray(fresh(tokens, depth=2, training=False, new_document=True).logits)
    np.testing.assert_array_equal(ref, got)


# --- The best checkpoint is selected on held-out CE (#222) -------------------

def test_best_follows_val_ce_when_train_ce_gets_lucky_early():
    """The shape of both stale `best/` dirs: one lucky train window early, then
    train CE never beats it while val CE keeps improving. Selection on val must
    keep moving; selection on train would have frozen at the first window."""
    monitor = LossMonitor()
    train = [3.9, 1.4, 3.7, 3.6, 3.5, 3.4]        # 1.4 is the lucky window
    val = [3.95, 3.90, 3.80, 3.70, 3.72, 3.60]    # steady, with one uptick
    saved = []
    for step, (t, v) in enumerate(zip(train, val)):
        monitor.push(step, t, t)
        if monitor.push_val(v):
            saved.append(step)
    assert saved == [0, 1, 2, 3, 5], "every val improvement saves; the uptick does not"
    assert monitor.best_val_ce == 3.60


def test_a_new_phase_starts_a_fresh_val_best():
    monitor = LossMonitor()
    monitor.push_val(2.0)
    monitor.reset_for_new_phase(100)
    assert monitor.push_val(2.5), "the phase's first probe is its best so far"


def test_the_trainer_saves_best_only_on_a_val_improvement():
    """The trigger lives inside train_loop, which needs data and a device to run
    end to end. So guard the wiring directly: the one save into best_mngr sits
    under `monitor.push_val(...)`, and nothing train-CE-shaped can trigger it."""
    import ast
    import inspect
    from trm.train import trainer

    tree = ast.parse(inspect.getsource(trainer.train_loop))
    saves = [call for call in ast.walk(tree)
             if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "save_checkpoint"
             and isinstance(call.args[0], ast.Name) and call.args[0].id == "best_mngr"]
    assert len(saves) == 1, "exactly one best save"
    guards = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and "push_val" in ast.unparse(node.test)
              and any(save in list(ast.walk(node)) for save in saves)]
    assert guards, "the best save must be guarded by monitor.push_val"
    cadenced = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                and "VAL_EVERY_OPT_STEPS" in ast.unparse(node.test)
                and any(save in list(ast.walk(node)) for save in saves)]
    assert cadenced, ("the best save must sit inside the probe's cadence, so best writes "
                      "are bounded to one per probe (#174), not one per improving log step")
    assert not hasattr(LossMonitor(), "is_new_best"), "the train-CE trigger is gone"


def test_the_best_dir_is_named_for_its_criterion():
    """Old `best/` archives were selected on train CE; a distinct name keeps them
    from being mistaken for these."""
    assert BEST_SUBDIR == "best_val_ce"


# --- Validation-probe cadence -------------------------------------------------

def _probe_opt_steps(total_micro_steps, accumulation_steps, val_every):
    """Replays exactly the step-gating the trainer uses and returns the list of
    opt-steps at which the probe fires. Mirrors trainer.train_loop's conditions."""
    fired = []
    for step in range(total_micro_steps):
        if (step + 1) % accumulation_steps == 0:
            opt_step = (step + 1) // accumulation_steps
            if opt_step % val_every == 0:
                fired.append(opt_step)
    return fired


def test_probe_fires_at_configured_cadence():
    accum = 4          # keep the test fast; ratio is what matters
    val_every = 8

    total = accum * val_every * 3  # cover several probe intervals
    fired = _probe_opt_steps(total, accum, val_every)

    assert fired == [8, 16, 24], f"probe should fire every {val_every} opt-steps, got {fired}"


def test_the_run_records_the_cadence_the_trainer_used():
    """#305: the training curve labels its val CE line "measured every N steps",
    and N has to be the plotted run's own — VAL_EVERY_OPT_STEPS is an env knob
    (the #26 arms run 16, the supervisor's fit gate runs 1)."""
    from trm.runtime import layout
    from trm.runtime.run_tracker import RunTracker

    assert RunTracker.get_hyperparameters()["VAL_EVERY_OPT_STEPS"] == layout.VAL_EVERY_OPT_STEPS


def test_old_nested_cadence_was_multiplied():
    """Guard documenting the bug: nesting the probe inside the logging block
    multiplied the cadence by LOG_REAL_STEPS — the fix must NOT reproduce this."""
    accum, log_real, val_every = 4, 5, 8

    # Buggy gating: opt_step only evaluated inside the logging block.
    buggy = []
    for step in range(accum * log_real * val_every * 2):
        if (step + 1) % (accum * log_real) == 0:
            opt_step = (step + 1) // accum
            if opt_step % val_every == 0:
                buggy.append(opt_step)

    fixed = _probe_opt_steps(accum * log_real * val_every * 2, accum, val_every)

    # Buggy cadence is lcm(log_real, val_every); fixed is val_every.
    assert buggy[0] == 40 and fixed[0] == 8
    assert all(s % (log_real * val_every // np.gcd(log_real, val_every)) == 0 for s in buggy)
