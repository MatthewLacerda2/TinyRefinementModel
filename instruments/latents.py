"""Capture the latent trajectory — every state a forward pass walks through, block
by block (#225, #391).

States alone are cheap: a whole trajectory at dim 960 and seq 512 is
9 x 512 x 960 x 4 bytes = 17.7 MB.

This module is the one API every downstream trajectory instrument uses. It ships
the measurement and makes no claim about what the trajectories will show — the
readouts below are descriptive.

    python -m instruments.latents --checkpoint runs/<run>/checkpoints [--step N]

The figures drawn from these trajectories live in `instruments/trajectory_figures.py`.

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

from instruments import results as result_lines
from instruments._common import add_checkpoint_argument, load_env
from trm.settings import CONFIG

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {
    "trajectory metrics (RESULT)": ("sampled", "geometry of the latent trajectory over the captured rows only"),
}


@dataclasses.dataclass(frozen=True)
class Trajectory:
    """Every state one forward pass passed through.

    `states` is [blocks+1, b, s, dim] in f32 — index 0 is the embedding, the state
    before the first block, which no other return path exposes. Without it there
    is no first step to measure and the trajectory has no origin.

    `nonfinite` counts non-finite entries. It is a field rather than an
    exception because a trajectory that blew up is itself a measurement (#229,
    #233) — but every readout below refuses to average over one, because a mean
    that silently swallows a NaN is worse than no number.
    """

    states: np.ndarray
    nonfinite: int = 0

    @property
    def blocks(self) -> int:
        return int(self.states.shape[0] - 1)

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
        """The per-block displacement vectors, [blocks, b, s, dim]."""
        return np.diff(self.states, axis=0)

    def step_sizes(self) -> np.ndarray:
        """‖z_k − z_{k−1}‖ per block, [blocks].

        Geometric decay means the stack is converging — later blocks are barely
        moving the stream. Flat means it is still doing work at the last block.
        """
        self._require_finite("step_sizes")
        s = self._steps().reshape(self.blocks, -1)
        return np.linalg.norm(s, axis=1)

    def turning_angles(self) -> np.ndarray:
        """cos∠(step_k, step_{k−1}), [blocks−1].

        Near +1: walking a straight line, so the blocks are one long stride
        chopped up. Near −1: oscillating, undoing the previous block.
        Near 0: wandering — each block is doing something unrelated to the last.
        """
        self._require_finite("turning_angles")
        s = self._steps().reshape(self.blocks, -1)
        norms = np.linalg.norm(s, axis=1)
        norms[norms == 0.0] = 1.0
        unit = s / norms[:, None]
        return np.einsum("kd,kd->k", unit[1:], unit[:-1])

    def distance_to_final(self) -> np.ndarray:
        """‖z_k − z_K‖ for every k, [blocks+1] — how far each state still is from
        where the stream ends up. Tells you how many blocks it took to arrive,
        which is not the same question as how big each step was."""
        self._require_finite("distance_to_final")
        d = (self.states - self.states[-1]).reshape(self.blocks + 1, -1)
        return np.linalg.norm(d, axis=1)

    def emit_results(self) -> None:
        """One `RESULT` line per block, so a spec can drive this and
        `instruments/verdict.py` can judge it (`instruments/results.py`)."""
        sizes, dists = self.step_sizes(), self.distance_to_final()
        angles = self.turning_angles()
        for k in range(1, self.blocks + 1):
            metrics = {"step_size": sizes[k - 1], "distance_to_final": dists[k]}
            if k > 1:  # an angle needs the step before this one
                metrics["turning_angle"] = angles[k - 2]
            result_lines.emit(f"block{k}", **metrics)


def capture(model, tokens) -> Trajectory:
    """Run one forward pass and keep every state it passed through.

    `model` is an already-restored model, not a path — capture does no I/O, so a
    caller sweeping documents restores once instead of per call.
    """
    states = np.asarray(model.capture_trajectory(tokens), dtype=np.float32)
    return Trajectory(states=states, nonfinite=int((~np.isfinite(states)).sum()))


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_checkpoint_argument(ap, required=True, aliases=("--checkpoint",))
    ap.add_argument("--source", default="pretrain/fineweb-edu")
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step to restore (default: the newest the manager holds)")
    args = ap.parse_args(argv)

    # DATA_ROOT lives in .env and the held-out loader reads it from the
    # environment. Loading it here rather than making every caller export it
    # by hand, the way trm/infer.py already does.
    load_env()

    from trm.runtime.restore import load_eval_batches, restore_model

    model, _ = restore_model(CONFIG, args.checkpoint_path, step=args.step)
    # load_eval_batches yields input rows, not (input, target) pairs.
    for i, row in enumerate(load_eval_batches(CONFIG, args.source, num_rows=args.rows)):
        traj = capture(model, jnp.asarray(row[:, :CONFIG.MAX_SEQ_LEN]))
        if not traj.ok:
            print(f"row {i}: {traj.nonfinite} non-finite values — skipped (#229/#233)")
            continue
        print(f"row {i}: states {traj.states.shape}")
        print("  step size        ", np.round(traj.step_sizes(), 4))
        print("  turning angle    ", np.round(traj.turning_angles(), 4))
        print("  distance to final", np.round(traj.distance_to_final(), 4))
        traj.emit_results()


if __name__ == "__main__":
    _main()
