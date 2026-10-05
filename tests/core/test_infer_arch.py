"""Inference builds the model the trainer builds, and its CLI reaches every sampling knob.

`trm/infer.py` once constructed one model class by name while the trainer built
another, so serving a checkpoint died on a structure mismatch (#185, #318). Serving
now builds through `trm.model.build_model`, the factory the trainer uses.
"""

from trm import infer


def test_infer_names_no_model_class():
    """Serving builds through trm.model.build_model, the factory the trainer uses
    (#318), so it cannot name a class of its own — a class named here is how the
    old hardcoding survived unnoticed."""
    from pathlib import Path
    source = Path(infer.__file__).read_text()
    assert "import PlainTransformer" not in source, "the model class is the factory's to import, not infer's"


def test_the_factory_imports_the_model_lazily():
    """Importing the model package must cost nothing: the model's module is imported
    inside build_model, when it is called."""
    import ast
    from pathlib import Path
    import trm.model
    tree = ast.parse(Path(trm.model.__file__).read_text())

    def runs_at_import(node):
        """Every statement import runs: module level, into try/if/with/for bodies
        and class bodies (a class body executes when the class is defined), but not
        into a function body, which only runs when called."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return
        yield node
        for child in ast.iter_child_nodes(node):
            yield from runs_at_import(child)

    top_level = [node for node in runs_at_import(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not top_level, "the model module is imported inside build_model, lazily"
    lazy = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert {node.module for node in lazy} >= {"trm.model.plain"}


# --- sampling is configurable, and the default is not greedy ------------------

def test_the_default_temperature_is_warm_enough_to_see_the_model():
    """0.5 divided the logits, doubling every gap, and with max|logit| ~21 partway
    through the base run that made sampling effectively greedy — which turns an
    undertrained LM into a repetition loop. What you read then is the decoder's
    failure mode, not the weights'."""
    assert infer.DEFAULT_TEMPERATURE == 0.7
    args = infer.build_arg_parser().parse_args([])
    assert args.temperature == 0.7


def test_every_sampling_knob_is_reachable_from_the_cli():
    args = infer.build_arg_parser().parse_args(
        ["--temperature", "0.9", "--top-k", "0", "--top-p", "1.0",
         "--max-new-tokens", "32"])
    assert (args.temperature, args.top_k, args.top_p) == (0.9, 0, 1.0)
    assert args.max_new_tokens == 32
