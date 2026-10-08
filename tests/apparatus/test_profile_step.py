"""The micro-step profiler (#473) traces the real trainer and reads its own trace.

The run half drives `instruments.profile_step` at its default window on a toy plain
model and a synthetic corpus, on CPU: what it proves is that the trainer's named
spans land in a trace and the summary reads a whole window from them. What the
trace *says* about the card needs the card (the PR's resume protocol).
"""

import glob
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from instruments.profile_step import covered, merged, read_trace, summarize

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEQ = 16
# BATCH_SIZE 32 makes ACCUMULATION_STEPS 4, so the default window (4 micro-steps
# after 6 of warmup) holds an optimizer update, as it does at the shipped size.
ENV = {"JAX_PLATFORMS": "cpu", "FORCE_F32_COMPUTE": "1", "LATENT_DIM": "32", "NUM_HEADS": "4",
       "MAX_SEQ_LEN": str(SEQ), "PLAIN_LAYERS": "2", "BATCH_SIZE": "32",
       "MODEL_SEED": "0", "DATA_SEED": "0"}
SPANS = {"data_get", "step_scalars", "grad_step", "loss_readback", "guard", "loss_scale",
         "apply_grads", "token_loss_readback"}


def test_the_cover_counts_overlapping_device_work_once():
    cover = merged([(0, 10), (5, 15), (20, 30)])
    assert cover == [[0, 15], [20, 30]]
    assert covered(cover, 10, 25) == 5 + 5


def test_the_window_runs_between_boundaries_and_charges_idle_to_the_span_it_sits_in():
    # Two micro-steps of 100 ns: the device is busy through each grad_step's call and
    # idle while the host reads the loss back.
    spans = [("data_get", 0, 10), ("grad_step", 10, 40), ("loss_readback", 40, 100),
             ("data_get", 100, 110), ("grad_step", 110, 140), ("loss_readback", 140, 200),
             ("data_get", 200, 210)]
    dispatch = [(15, 40), (115, 140)]
    device = [(20, 60), (120, 160)]
    s = summarize(spans, dispatch, device)
    assert (s["micro_steps"], s["wall"], s["busy"]) == (2, 200, 80)
    assert s["spans"]["loss_readback"] == (120, 120 - 40, 120)
    # The Python around the call: 5 ns of each grad_step span holds no XLA call.
    assert s["spans"]["grad_step"][2] == 10
    # The last boundary's own span lies outside the window.
    assert s["spans"]["data_get"][0] == 20


def test_a_trace_without_two_boundaries_is_refused():
    with pytest.raises(SystemExit, match="no whole micro-step"):
        summarize([("data_get", 0, 10)], [], [])


def _corpus(root):
    rng = np.random.default_rng(0)
    for source in ("fineweb-edu", "codeparrot", "finemath"):
        d = root / "data" / "pretrain" / source
        d.mkdir(parents=True)
        np.save(d / "chunk_0.npy", rng.integers(0, 50000, 400 * (2 * SEQ + 1), dtype=np.int32))


@pytest.fixture(scope="module")
def profile(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("profile")
    _corpus(tmp)
    env = {**os.environ, **ENV, "PYTHONPATH": REPO, "DATA_ROOT": str(tmp / "data"),
           "JAX_COMPILATION_CACHE_DIR": str(tmp / "jax-cache")}
    out = subprocess.run([sys.executable, "-m", "instruments.profile_step", "--out", str(tmp / "profile")],
                         cwd=tmp, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, (out.stdout + out.stderr)[-4000:]
    (run_dir,) = glob.glob(str(tmp / "profile" / "run_*"))
    return run_dir, out.stdout


def test_the_trace_holds_every_named_span_of_the_default_window(profile):
    run_dir, _ = profile
    (xplane,) = glob.glob(os.path.join(run_dir, "trace", "**", "*.xplane.pb"), recursive=True)
    spans, dispatch, _, _ = read_trace(xplane)
    s = summarize(spans, dispatch, [])
    assert s["micro_steps"] == 4
    assert set(s["spans"]) == SPANS
    assert dispatch, "the XLA calls on the trainer's thread are what the nnx.jit wrapper is measured against"
    assert all(wrapper >= 0 for _, _, wrapper in s["spans"].values())


def test_the_summary_is_written_beside_a_perfetto_trace(profile):
    run_dir, stdout = profile
    summary = pathlib.Path(run_dir, "summary.txt").read_text()
    assert "profile: 4 micro-steps" in summary and summary.strip() in stdout
    assert all(f"  {name} " in summary for name in SPANS)
    assert "#474 gate" in summary
    assert glob.glob(os.path.join(run_dir, "trace", "**", "perfetto_trace.json.gz"), recursive=True)


def test_pace_times_the_real_loop_without_a_trace(tmp_path):
    """#561: the throughput reading for any knob that changes the micro-step count runs
    the trainer's own loop, unprofiled. It counts every micro-step between the two
    settles: the window's micro-steps plus the two that bracket it."""
    _corpus(tmp_path)
    env = {**os.environ, **ENV, "PYTHONPATH": REPO, "DATA_ROOT": str(tmp_path / "data"),
           "JAX_COMPILATION_CACHE_DIR": str(tmp_path / "jax-cache")}
    out = subprocess.run([sys.executable, "-m", "instruments.profile_step", "--pace", "--out",
                          str(tmp_path / "profile")], cwd=tmp_path, env=env, capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, (out.stdout + out.stderr)[-4000:]
    assert "pace: batch 32 x 4 micro-steps per opt step, 6 micro-steps timed" in out.stdout
    assert "s/opt step" in out.stdout and "tok/s" in out.stdout
    (run_dir,) = glob.glob(str(tmp_path / "profile" / "run_*"))
    assert not os.path.exists(os.path.join(run_dir, "trace")), "--pace runs no profiler"
