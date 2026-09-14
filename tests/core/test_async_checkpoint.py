"""Checkpoint writes no longer stop training, and cannot tear (#218).

The blocking write held the GPU at 0% for ~12s per checkpoint. The danger in
removing it is a torn checkpoint — state read while training mutates it — which
is far worse than 12 seconds. So the gate is a round trip, not a timing.
"""

import os
import signal
import subprocess
import sys
import textwrap

import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

from trm.runtime import checkpoints as ck
from trm.runtime.monitor import LossMonitor


def _leaves(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


def test_an_async_write_restores_the_state_at_save_time_while_training_mutates_it(tmp_path):
    from instruments.arch import build

    model = build("plain", dim=60, num_layers=2)
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
    monitor = LossMonitor()
    monitor.ce_history = [3.0, 2.9]
    snapshot = _leaves(nnx.state(model))

    mngr = ocp.CheckpointManager(str(tmp_path), item_names=ck.CHECKPOINT_ITEMS,
                                 options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True))
    ck.save_checkpoint(mngr, 10, model, optimizer, monitor, False, "run_x", wait=False)
    # Train on during the write: every weight moves, and the history grows.
    nnx.update(model, jax.tree_util.tree_map(lambda x: x + 1.0, nnx.state(model, nnx.Param)))
    monitor.ce_history.append(2.8)
    ck.wait_for_pending_saves()

    fresh = build("plain", dim=60, num_layers=2, seed=5)
    restored = mngr.restore(10, args=ocp.args.Composite(
        model=ocp.args.StandardRestore(nnx.state(fresh)),
        optimizer=ocp.args.StandardRestore(nnx.state(nnx.Optimizer(fresh, optax.adam(1e-2), wrt=nnx.Param))),
        monitor_state=ocp.args.JsonRestore(), step=ocp.args.JsonRestore()))
    assert all(np.array_equal(a, b) for a, b in zip(snapshot, _leaves(restored["model"])))
    assert restored["monitor_state"]["ce_history"] == [3.0, 2.9], "history as of the save, not after"


class _Manager:
    def __init__(self, log, name):
        self.log, self.name, self.directory = log, name, name

    def save(self, step, args):
        self.log.append(f"{self.name}.save")
        return True

    def wait_until_finished(self):
        self.log.append(f"{self.name}.wait")

    def latest_step(self):
        return None


def test_only_one_write_is_in_flight_so_host_ram_holds_one_copy(tmp_path, monkeypatch, tiny_model):
    """Rolling, best and milestone can all save at one step. Each holds a ~1.7GB host
    copy until written, so each new save first waits for the previous write."""
    optimizer = nnx.Optimizer(tiny_model, optax.sgd(0.0), wrt=nnx.Param)
    log = []
    rolling, best = _Manager(log, "rolling"), _Manager(log, "best")
    ck.save_checkpoint(rolling, 1, tiny_model, optimizer, LossMonitor(), False, "r", wait=False)
    ck.save_checkpoint(best, 1, tiny_model, optimizer, LossMonitor(), False, "r", wait=False)
    ck.wait_for_pending_saves()
    assert log == ["rolling.save", "rolling.wait", "best.save", "best.wait"]


def test_the_trainer_saves_asynchronously_and_waits_on_the_way_out():
    import ast
    import inspect
    from trm.train import trainer

    tree = ast.parse(inspect.getsource(trainer.train_loop))
    saves = [c for c in ast.walk(tree) if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "save_checkpoint"]
    assert saves and all(any(k.arg == "wait" and k.value.value is False for k in c.keywords) for c in saves)
    finals = [n for n in ast.walk(tree) if isinstance(n, ast.Try) and n.finalbody]
    assert any("wait_for_pending_saves" in ast.unparse(f) for n in finals for f in n.finalbody)


def test_sigterm_unwinds_so_a_pending_write_can_finish(tmp_path):
    """The supervisor's budget stop is a TERM. Without this, the process dies
    without running `finally`, taking an in-flight final checkpoint with it."""
    marker = tmp_path / "finally_ran"
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(f"""
        import os, signal, time
        from trm.runtime.checkpoints import exit_cleanly_on_sigterm
        exit_cleanly_on_sigterm()
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(30)
        finally:
            open({str(marker)!r}, "w").write("yes")
    """))
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run([sys.executable, str(script)], cwd=root, timeout=120,
                          env={**os.environ, "PYTHONPATH": root}, capture_output=True, text=True)
    assert marker.exists(), "finally must run on SIGTERM"
    assert proc.returncode == 128 + signal.SIGTERM
    assert "received SIGTERM" in proc.stdout, "a TERM must leave a trace in the log, not end it mid-stream"
