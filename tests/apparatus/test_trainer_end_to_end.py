"""The real trainer, end to end, at toy size on CPU (#325).

`python -m trm.train.start` trains a 2-block, 32-wide plain model on a synthetic
corpus: to a checkpoint, killed with TERM the way every pair harness and the
supervisor stop it, then resumed from that checkpoint. The assertions read what the
run wrote (metrics.csv, the checkpoint, the log), never the trainer's source text.
A string check that `train_loop` mentions `applied_gradient_stats` passes for code
that never calls it; a filled `applied_grad_norm` column does not.

Slow (two JAX processes, ~1-2 min), so it runs in CI's pytest job and not beside a
trainer on the card.
"""

import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

from instruments.runlog import load

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEQ = 16
STRIDE = 2 * SEQ + 1
# Opt steps per leg. Rows are logged every LOG_REAL_STEPS = 5 opt steps.
FIRST_LEG, SECOND_LEG = 10, 15
ENV = {
    "JAX_PLATFORMS": "cpu", "FORCE_F32_COMPUTE": "1",
    "LATENT_DIM": "32", "NUM_HEADS": "4", "MAX_SEQ_LEN": str(SEQ), "PLAIN_LAYERS": "2",
    "MODEL_ARCH": "plain", "EVAL_ROWS": "2", "VAL_SKIP_SAMPLES": "0",
    "VAL_EVERY_OPT_STEPS": "5", "VAL_BY_SOURCE_EVERY_OPT_STEPS": "5",
    "CHECKPOINT_EVERY_OPT_STEPS": "5", "MILESTONE_FIRST_TOKENS": "0",
    "WARMUP_STEPS": "2", "TRAIN_TOKEN_BUDGET": str(40 * 128 * 2 * SEQ),
    "MODEL_SEED": "0", "DATA_SEED": "0",
}


def _corpus(root):
    rng = np.random.default_rng(0)
    for source in ("fineweb-edu", "codeparrot", "finemath"):
        d = root / "data" / "pretrain" / source
        d.mkdir(parents=True)
        for i in range(2):
            np.save(d / f"chunk_{i}.npy", rng.integers(0, 50000, 3000 * STRIDE, dtype=np.int32))
    return root / "data"


def _last_step(metrics):
    try:
        steps = [int(line.split(",")[0]) for line in metrics.read_text().splitlines()[1:]
                 if line.split(",")[0].isdigit()]
    except OSError:
        return 0
    return max(steps, default=0)


def _train_until(tmp_path, run_dir, opt_steps, timeout=900):
    env = {**os.environ, **ENV, "PYTHONPATH": REPO, "DATA_ROOT": str(tmp_path / "data"),
           "JAX_COMPILATION_CACHE_DIR": str(tmp_path / "jax-cache")}
    log = open(run_dir / "leg.log", "a")
    proc = subprocess.Popen([sys.executable, "-m", "trm.train.start", "--checkpoint-path",
                             str(run_dir / "checkpoints")], cwd=tmp_path, env=env,
                            stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + timeout
    try:
        while proc.poll() is None and _last_step(run_dir / "metrics.csv") < opt_steps:
            assert time.time() < deadline, (run_dir / "leg.log").read_text()[-3000:]
            time.sleep(2)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=180)
    return (run_dir / "leg.log").read_text()


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("e2e")
    _corpus(tmp_path)
    run_dir = tmp_path / "runs" / "run_e2e"
    run_dir.mkdir(parents=True)
    first = _train_until(tmp_path, run_dir, FIRST_LEG)
    checkpoint = run_dir / "checkpoints" / str(FIRST_LEG * 128 - 1)
    second = _train_until(tmp_path, run_dir, SECOND_LEG)
    return {"dir": run_dir, "first": first, "second": second, "checkpoint": checkpoint}


def test_a_logged_row_carries_every_measurement_the_loop_is_wired_to_make(run):
    rows = {r["step"]: r for r in load(str(run["dir"])).metrics}
    assert FIRST_LEG in rows, sorted(rows)
    row = rows[FIRST_LEG]
    for column in ("ce", "applied_grad_norm", "applied_zero_frac_dense_max",
                   "zero_frac_dense_max", "val_ce", "val_step", "clip_active",
                   "loss_scale", "act_max"):
        assert row.get(column) is not None, f"{column} empty on the row at opt step {FIRST_LEG}"
    assert row["val_step"] == FIRST_LEG
    assert set(row["val_by_source"]) == {"codeparrot", "finemath"}


def test_the_checkpoint_records_what_was_consumed_and_where_the_data_stands(run):
    state = json.loads((run["checkpoint"] / "monitor_state" / "metadata").read_text())
    # One row per micro-step at BATCH_SIZE 1, counted as served (#24): a full window
    # per opt step, so the checkpoint sits on the optimizer's boundary (#355).
    assert state["samples_seen"] == FIRST_LEG * 128
    assert state["data_state"]["sources"], "the exact data position (#424)"


def test_the_checkpoint_is_taken_between_optimizer_windows(run):
    """#355: the trainer's opt-step boundary is the optimizer's. A checkpoint taken
    one micro-step early holds 127 gradients in the accumulator and N-1 updates."""
    import orbax.checkpoint as ocp

    leaves = dict(_leaves(ocp.StandardCheckpointer().restore(run["checkpoint"] / "optimizer")))
    assert int(leaves["/opt_state/mini_step/value"]) == 0
    assert int(leaves["/opt_state/gradient_step/value"]) == FIRST_LEG


def _leaves(tree, prefix=""):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _leaves(value, f"{prefix}/{key}")
    elif isinstance(tree, (list, tuple)):
        for i, value in enumerate(tree):
            yield from _leaves(value, f"{prefix}/{i}")
    else:
        yield prefix, tree


def test_the_resume_continues_from_the_checkpoint_and_restores_the_data_exactly(run):
    assert f"Resuming from step {FIRST_LEG * 128}" in run["second"]
    assert "Data stream restored exactly" in run["second"]
    steps = [r["step"] for r in load(str(run["dir"])).metrics]
    assert steps == sorted(set(steps)) and SECOND_LEG in steps
