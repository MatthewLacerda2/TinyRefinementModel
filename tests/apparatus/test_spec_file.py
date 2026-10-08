"""Every spec is read through typed tables (#478): a wrong key fails at the boundary."""

import pathlib
import textwrap

import pytest

from instruments.spec_file import SpecFile

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPECS = sorted(ROOT.glob("experiments/*/specs/*.toml"))

MINIMAL = textwrap.dedent("""
    [experiment]
    id = "x"
    title = "t"
    hypothesis = "h"
    [criteria.c]
    rule = "beats"
    treatment = "a"
    control = "b"
    sigmas = 2.0
    [verdict]
    keep_if = ["c"]
""")


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.name)
def test_every_committed_spec_reads(path):
    assert SpecFile.load(path).experiment.id


def test_a_misspelled_key_names_its_table(tmp_path):
    spec = tmp_path / "s.toml"
    spec.write_text(MINIMAL.replace('sigmas = 2.0', 'sigmas = 2.0\nmin_detla = 0.1'))
    with pytest.raises(ValueError, match=r"\[criteria\.c\.min_detla\] Extra inputs"):
        SpecFile.load(spec)


def test_a_wrong_type_names_its_key(tmp_path):
    spec = tmp_path / "s.toml"
    spec.write_text(MINIMAL + '[execution]\ncommand = ["python"]\nseeds = ["zero"]\n')
    with pytest.raises(ValueError, match=r"\[execution\.seeds\.0\]"):
        SpecFile.load(spec)


def test_the_protocol_keeps_what_a_retrofit_recorded(tmp_path):
    spec = tmp_path / "s.toml"
    spec.write_text(MINIMAL + '[protocol]\nmetric = "acc"\nmatched = "same seed"\ndim = 512\n')
    protocol = SpecFile.load(spec).protocol
    assert protocol.metric == "acc" and protocol.model_extra == {"matched": "same seed", "dim": 512}
