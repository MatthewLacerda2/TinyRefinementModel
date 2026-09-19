"""Muon's Newton-Schulz coefficients are a knob (#375), and the default is unchanged.

Newton-Schulz acts on each singular value on its own, so the whole question fits in
scalars: run the 5 steps on x in (0, 1] and see how close to 1 each lands.
"""

import numpy as np
import pytest

from trm.train.optimizers import KELLER_NS_COEFFS, ns_coefficients


def _run(coeffs, x):
    for a, b, c in coeffs:
        x = a * x + b * x ** 3 + c * x ** 5
    return x


def test_the_default_is_the_2024_quintic_so_every_run_on_record_resolves_the_same():
    assert ns_coefficients("keller") == KELLER_NS_COEFFS == (3.4445, -4.775, 2.0315)


def test_polar_express_lands_closer_to_orthogonal_in_the_same_five_steps():
    x = np.geomspace(1e-2, 1.0, 2000)
    keller = _run([KELLER_NS_COEFFS] * 5, x)
    polar = _run(ns_coefficients("polar_express", 5), x)
    assert len(ns_coefficients("polar_express", 5)) == 5
    assert np.max(np.abs(polar - 1)) < np.max(np.abs(keller - 1))
    assert np.mean(np.abs(polar - 1)) < 0.5 * np.mean(np.abs(keller - 1))


def test_an_unknown_table_is_refused():
    with pytest.raises(ValueError, match="MUON_NS_COEFFS"):
        ns_coefficients("kellr")
