"""The architectures, and the one place that maps an arch name to its network.

Which names exist is `trm/config.py`'s to say (MODEL_ARCH fails closed on a typo);
this only builds them.
"""


def build_model(arch, dim, rngs, **overrides):
    """A fresh `arch` network of width `dim`, initialized from `rngs`.

    Every arch module is imported only in its own branch, so a run of one arch
    never imports another's code, and importing `trm.model` costs nothing.
    `overrides` go to the constructor as-is; the arches do not share a signature
    (`num_layers` is the plain stack's, `encoder_layers` the refiner's), so a knob
    the arch does not take is a TypeError, not something silently dropped.
    """
    if arch == "plain":
        from trm.model.plain import PlainTransformer
        return PlainTransformer(dim, rngs, **overrides)
    if arch == "refiner":
        from trm.model.refiner_lm import RefinerForTraining
        return RefinerForTraining(dim, rngs, **overrides)
    if arch == "reasoner":
        from trm.model.reasoner import UniversalReasoner
        return UniversalReasoner(dim, rngs, **overrides)
    raise ValueError(f"unknown architecture {arch!r}")
