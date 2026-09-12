"""Build the architecture a run would actually train — one place, for every smoke.

Four checks were found aimed to one side of what they guard, in one week: the
pre-launch overfit gate built the CONTROL architecture, the GPU numerical smoke and
the VRAM sizer both hardcoded the refiner, and the memory profiler did too. None was
noticed until the default changed and they all kept pointing at the past.

The failure is not that any one of them was wrong to name an architecture. It is
that naming one was invisible — a constructor call in the middle of a file, with no
statement that a choice had been made. So the choice lives here, every smoke takes
`--arch` defaulting to `MODEL_ARCH`, and
`tests/core/test_instruments_know_their_architecture.py` fails any instrument that
builds a model without either reading MODEL_ARCH or declaring itself arch-specific
in one line.

Keyword overrides are passed through per architecture, because the arches do not
share a signature: `num_layers` is the plain stack's depth knob, `encoder_layers` is
the refiner's, and the reasoner takes neither.
"""

from flax import nnx

from trm.config import LATENT_DIM, MODEL_ARCH

ARCHES = ("plain", "refiner", "reasoner")


def build(arch=None, *, dim=None, seed=0, **overrides):
    """The model `arch` names, or `MODEL_ARCH` when it names nothing.

    `overrides` are forwarded only where they apply — passing `num_layers` to the
    refiner is a caller error worth failing on, not something to silently drop.
    """
    arch = MODEL_ARCH if arch is None else arch
    if arch not in ARCHES:
        raise SystemExit(f"unknown --arch {arch!r}; use one of {', '.join(ARCHES)}")
    dim = LATENT_DIM if dim is None else dim
    rngs = nnx.Rngs(seed)

    if arch == "plain":
        from trm.model.plain import PlainTransformer
        return PlainTransformer(dim, rngs, **overrides)
    if arch == "refiner":
        from trm.model.refiner_lm import RefinerForTraining
        return RefinerForTraining(dim, rngs, **overrides)
    from trm.model.reasoner import UniversalReasoner
    return UniversalReasoner(dim, rngs, **overrides)


def add_arch_argument(parser):
    """The flag every smoke should take, worded the same way in each."""
    parser.add_argument(
        "--arch", default=MODEL_ARCH, choices=ARCHES,
        help="architecture to exercise; defaults to MODEL_ARCH, i.e. whatever a run "
             "launched right now would actually train")
    return parser
