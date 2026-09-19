"""Capture the latent trajectory — every state a forward pass walks through:
the refiner's refine loop (#225), or the plain model's blocks (#391).

States alone are cheap: a whole trajectory at dim 960 and seq 512 is
9 x 512 x 960 x 2 bytes = 8.8 MB. Why the production model could not hand one back
before, and the depth-inertness it would have caught earlier: #225.

This module is the one API every downstream depth instrument uses (#227's
readout probe, #228's visualiser). It ships the measurement and makes no claim
about what the trajectories will show — the readouts below are descriptive.

    python -m instruments.latents --checkpoint runs/<run>/checkpoints [--step N]

The plain model ignores `--depth`: its trajectory is its N blocks. The figures
drawn from these trajectories live in `instruments/trajectory_figures.py`.

**The checkpoint path is the MANAGER ROOT** — the directory that *contains*
numerically-named step directories — not the step directory itself. Orbax
discovers checkpoints by scanning for numeric subdirectory names, so passing
`checkpoints/3899391` fails with "No checkpoint found" while `checkpoints/`
works. This trap silently made four HDD archives unloadable once already.
"""

from __future__ import annotations

import argparse
import dataclasses

import jax.numpy as jnp
import numpy as np

from trm.config import MAX_SEQ_LEN, MAX_STEPS_LIMIT

from instruments import results as result_lines
from instruments._common import add_checkpoint_argument, load_env

# A trajectory is whatever stack of states the architecture walks: the refiner's passes,
# the plain model's blocks. `LanguageModel.capture_trajectory` refuses for any other (#317).

# The architectures that implement LanguageModel.capture_trajectory. Checked before a
# checkpoint loads, so asking for another is refused in a second, not after a restore.
TRAJECTORY_ARCHES = ("plain", "refiner")

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {
    "trajectory metrics (RESULT)": ("sampled", "geometry of the latent trajectory over the captured rows only"),
}


@dataclasses.dataclass(frozen=True)
class Trajectory:
    """Every state one forward pass passed through.

    `states` is [depth+1, b, s, dim] in f32 — index 0 is the state before the
    first pass or block, which no other return path exposes. Without it
    there is no first step to measure and the trajectory has no origin.

    `gates` is the per-pass mean gate openness [depth], already computed inside
    the refine loop and previously discarded, or None for a model with no gate.

    `nonfinite` counts non-finite entries. It is a field rather than an
    exception because a trajectory that blew up is itself a measurement (#229,
    #233) — but every readout below refuses to average over one, because a mean
    that silently swallows a NaN is worse than no number.
    """

    states: np.ndarray
    gates: np.ndarray | None
    depth: int
    nonfinite: int = 0

    @property
    def ok(self) -> bool:
        return self.nonfinite == 0

    def _require_finite(self, what):
        if not self.ok:
            raise ValueError(
                f"{what}: the trajectory holds {self.nonfinite} non-finite values, "
                f"so this readout would be meaningless. See #229 (f16 overflow on "
                f"some checkpoints) and #233 (one out-of-vocab token id poisons the "
                f"whole window)."
            )

    def _steps(self):
        """The per-pass displacement vectors, [depth, b, s, dim]."""
        return np.diff(self.states, axis=0)

    def step_sizes(self) -> np.ndarray:
        """‖z_k − z_{k−1}‖ per pass, [depth].

        Geometric decay means the loop is converging to a fixed point — later
        passes are barely moving and the depth is going spare. Flat means it is
        still doing work at the last pass we run.
        """
        self._require_finite("step_sizes")
        s = self._steps().reshape(self.depth, -1)
        return np.linalg.norm(s, axis=1)

    def turning_angles(self) -> np.ndarray:
        """cos∠(step_k, step_{k−1}), [depth−1].

        Near +1: walking a straight line, so the passes are one long stride
        chopped up, and the same displacement might be reachable in fewer.
        Near −1: oscillating, undoing the previous pass.
        Near 0: wandering — each pass is doing something unrelated to the last.
        """
        self._require_finite("turning_angles")
        s = self._steps().reshape(self.depth, -1)
        norms = np.linalg.norm(s, axis=1)
        norms[norms == 0.0] = 1.0
        unit = s / norms[:, None]
        return np.einsum("kd,kd->k", unit[1:], unit[:-1])

    def distance_to_final(self) -> np.ndarray:
        """‖z_k − z_K‖ for every k, [depth+1] — how far each state still is from
        where the pass ends up. Tells you how many passes it took to arrive,
        which is not the same question as how big each step was."""
        self._require_finite("distance_to_final")
        d = (self.states - self.states[-1]).reshape(self.depth + 1, -1)
        return np.linalg.norm(d, axis=1)

    def emit_results(self) -> None:
        """One `RESULT` line per pass, so a spec can drive this and
        `instruments/verdict.py` can judge it (`instruments/results.py`)."""
        sizes, dists = self.step_sizes(), self.distance_to_final()
        angles = self.turning_angles()
        for k in range(1, self.depth + 1):
            metrics = {"step_size": sizes[k - 1], "distance_to_final": dists[k]}
            if k >= 2:
                metrics["turning_angle"] = angles[k - 2]
            if self.gates is not None:
                metrics["gate_openness"] = self.gates[k - 1]
            result_lines.emit(f"d{k}", **metrics)


def capture(model, tokens, depth) -> Trajectory:
    """Run one forward pass and keep every state it passed through.

    `model` is an already-restored model, not a path — capture does no I/O, so a
    caller sweeping depths or documents restores once instead of per call.
    """
    states, gates = model.capture_trajectory(tokens, depth=depth)
    states = np.asarray(states, dtype=np.float32)
    nonfinite = int((~np.isfinite(states)).sum())
    return Trajectory(
        states=states,
        gates=None if gates is None else np.asarray(gates, dtype=np.float32),
        depth=int(states.shape[0] - 1),
        nonfinite=nonfinite,
    )


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_checkpoint_argument(ap, required=True, aliases=("--checkpoint",))
    ap.add_argument("--depth", type=int, default=MAX_STEPS_LIMIT)
    ap.add_argument("--source", default="pretrain/fineweb-edu")
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step to restore (default: the newest the manager holds)")
    args = ap.parse_args(argv)
    from trm.config import MODEL_ARCH
    if MODEL_ARCH not in TRAJECTORY_ARCHES:
        raise SystemExit(f"instruments.latents reads a trajectory; MODEL_ARCH={MODEL_ARCH!r} "
                         f"has none to capture. Use one of {', '.join(TRAJECTORY_ARCHES)}.")

    # DATA_ROOT lives in .env and the held-out loader reads it from the
    # environment. Loading it here rather than making every caller export it
    # by hand, the way trm/infer.py already does.
    load_env()

    from trm.runtime.restore import load_eval_batches, restore_arch

    model, _ = restore_arch(MODEL_ARCH, args.checkpoint_path, step=args.step)
    # load_eval_batches yields input rows, not (input, target) pairs.
    for i, row in enumerate(load_eval_batches(args.source, num_rows=args.rows)):
        traj = capture(model, jnp.asarray(row[:, :MAX_SEQ_LEN]), args.depth)
        if not traj.ok:
            print(f"row {i}: {traj.nonfinite} non-finite values — skipped (#229/#233)")
            continue
        print(f"row {i}: states {traj.states.shape}")
        print("  step size        ", np.round(traj.step_sizes(), 4))
        print("  turning angle    ", np.round(traj.turning_angles(), 4))
        print("  distance to final", np.round(traj.distance_to_final(), 4))
        if traj.gates is not None:
            print("  gate openness    ", np.round(traj.gates, 4))
        traj.emit_results()


if __name__ == "__main__":
    _main()
