"""Capture the refinement trajectory — every latent state the model passes
through while it thinks (#225).

Depth inertness was found at the *end* of a 10-day run by eyeballing eight
sampled prompts, because the production model could not hand back a single
intermediate state: `return_all_states` existed only on the toy `CausalRefiner`,
and even there it was entangled with `return_all_iters`, which drags the
`[depth, b, s, vocab]` logit tensor along. States alone were never expensive.
At the live config a whole trajectory is

    9 states x 1 x 512 x 960 x 2 bytes = 8.8 MB

so this has been affordable the entire time; the coupling is what kept it out of
reach.

This module is the one API every downstream depth instrument uses (#227's
readout probe, #228's visualiser). It ships the measurement and makes no claim
about what the trajectories will show — the readouts below are descriptive.

    python -m instruments.latents --checkpoint runs/<run>/checkpoints --depth 8

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

# ARCH-SPECIFIC: refiner — a trajectory is the refine loop's states, and only the refiner loops (#317).

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {
    "trajectory metrics (RESULT)": ("sampled", "geometry of the latent trajectory over the captured rows only"),
}


@dataclasses.dataclass(frozen=True)
class Trajectory:
    """Every state one forward pass passed through.

    `states` is [depth+1, b, s, dim] in f32 — index 0 is the encoder output,
    *before* any refinement, which no other return path exposes. Without it
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
    ap.add_argument("--checkpoint", required=True,
                    help="checkpoint MANAGER ROOT (the dir holding numeric step dirs), "
                         "not a step dir")
    ap.add_argument("--depth", type=int, default=MAX_STEPS_LIMIT)
    ap.add_argument("--source", default="pretrain/fineweb-edu")
    ap.add_argument("--rows", type=int, default=1)
    args = ap.parse_args(argv)
    from trm.config import MODEL_ARCH
    if MODEL_ARCH != "refiner":
        raise SystemExit(f"instruments.latents reads the refiner's refinement trajectory; "
                         f"MODEL_ARCH={MODEL_ARCH!r} has no refine loop to capture. "
                         f"Load a refiner checkpoint with MODEL_ARCH=refiner.")

    # DATA_ROOT lives in .env and the held-out loader reads it from the
    # environment. Loading it here rather than making every caller export it
    # by hand, the way trm/infer.py already does.
    from dotenv import load_dotenv
    load_dotenv()

    from trm.runtime.restore import load_eval_batches, restore_model

    model, _ = restore_model(args.checkpoint)
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
