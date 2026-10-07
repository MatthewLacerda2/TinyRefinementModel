"""The model, and the one place every entry point builds it through."""


def build_model(config, rngs, *, dim=None, **overrides):
    """A fresh PlainTransformer of width `dim` (default: config.LATENT_DIM), shaped by
    `config` and initialized from `rngs`.

    `overrides` go to the constructor as-is (`num_layers`, `num_heads`, ...), so a
    test can build a tiny instance; a knob the model does not take is a TypeError.
    The import is deferred so importing `trm.model` costs nothing.
    """
    from trm.model.plain import PlainTransformer
    return PlainTransformer(config.LATENT_DIM if dim is None else dim, rngs, config, **overrides)
