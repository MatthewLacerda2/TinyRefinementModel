"""Build the architecture a run would actually train — one place, for every smoke.

Every smoke takes `--arch` defaulting to `MODEL_ARCH` and builds through here. Why a
class name in an instrument is refused, and the four incidents behind it:
`tests/core/test_instruments_know_their_architecture.py`, which enforces it.

The network itself comes from `trm.model.build_model`, the factory the trainer and
the restore path use, so a smoke cannot build a model the trainer would not. Keyword
overrides are passed through per architecture, because the arches do not share a
signature: `num_layers` is the plain stack's depth knob, `encoder_layers` is the
refiner's, and the reasoner takes neither.
"""

from flax import nnx

from trm.config import LATENT_DIM, MODEL_ARCH
from trm.model import build_model

ARCHES = ("plain", "refiner", "reasoner")


def build(arch=None, *, dim=None, seed=0, **overrides):
    """The model `arch` names, or `MODEL_ARCH` when it names nothing.

    `overrides` are forwarded only where they apply — passing `num_layers` to the
    refiner is a caller error worth failing on, not something to silently drop.
    """
    arch = MODEL_ARCH if arch is None else arch
    if arch not in ARCHES:
        raise SystemExit(f"unknown --arch {arch!r}; use one of {', '.join(ARCHES)}")
    return build_model(arch, LATENT_DIM if dim is None else dim, nnx.Rngs(seed), **overrides)


def add_arch_argument(parser):
    """The flag every smoke should take, worded the same way in each."""
    parser.add_argument(
        "--arch", default=MODEL_ARCH, choices=ARCHES,
        help="architecture to exercise; defaults to MODEL_ARCH, i.e. whatever a run "
             "launched right now would actually train")
    return parser
