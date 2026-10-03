"""Polar Express: per-step Newton-Schulz coefficients for Muon (#375).

Muon orthogonalizes its update with 5 Newton-Schulz steps. The 2024 recipe reuses one
quintic at every step; Polar Express (Amsel, Persson, Musco & Gower, 2025) solves for
the minimax-optimal quintic at each step, so the same 5 steps land closer to
orthogonal. The coefficients are generated here by the paper's own algorithm rather
than typed in, so there is no table to mistranscribe.

Adapted from the reference implementation, github.com/NoahAmsel/PolarExpress
(polar_express.py at 71cc379), MIT licence, Copyright (c) the Polar Express authors.
Only the degree-5 path, with the reference's own settings.
"""

from functools import lru_cache
from math import inf, sqrt

import numpy as np
from numpy.polynomial import Polynomial

# The reference's settings: the smallest singular value it plans for, the per-step
# safety factor against rounding past 1, the cushion that keeps early steps sane, and
# the length of the schedule it builds. A run of fewer steps takes the first ones, as
# the reference's own PolarExpress() does, so step 5 of 5 still carries the safety
# factor; more than 10 repeats the last.
LOWER_BOUND, SAFETY_EPS, CUSHION, SCHEDULE_STEPS = 1e-3, 1e-2, 0.02, 10


def optimal_quintic(lo, hi):
    """The odd quintic a*x + b*x^3 + c*x^5 closest to 1 on [lo, hi] in the max norm,
    by the simplified Remez exchange."""
    assert 0 <= lo <= hi
    if 1 - 5e-6 <= lo / hi:
        # The equioscillating polynomial is numerically this once lo ~ hi.
        return (15 / 8) / hi, (-10 / 8) / (hi ** 3), (3 / 8) / (hi ** 5)
    q, r = (3 * lo + hi) / 4, (lo + 3 * hi) / 4
    E, old_E = inf, None
    while not old_E or abs(old_E - E) > 1e-15:
        old_E = E
        lhs = np.array([[lo, lo ** 3, lo ** 5, 1], [q, q ** 3, q ** 5, -1],
                        [r, r ** 3, r ** 5, 1], [hi, hi ** 3, hi ** 5, -1]])
        a, b, c, E = np.linalg.solve(lhs, np.ones(4))
        q, r = np.sqrt((-3 * b + np.array([-1, 1]) * sqrt(9 * b ** 2 - 20 * a * c)) / (10 * c))
    return float(a), float(b), float(c)


def polar_express_coeffs(steps):
    """`steps` (a, b, c) tuples, one per Newton-Schulz step, in the form optax's
    `scale_by_muon(ns_coeffs=...)` takes."""
    schedule = _schedule()
    return schedule[:steps] + (schedule[-1],) * max(steps - len(schedule), 0)


@lru_cache
def _schedule():
    lo, hi = LOWER_BOUND, 1.0
    safety = 1 + SAFETY_EPS
    coefficients = []
    for i in range(SCHEDULE_STEPS):
        p = Polynomial.identity() * Polynomial(optimal_quintic(max(lo, CUSHION * hi), hi))(
            Polynomial.identity() ** 2)
        if CUSHION * hi > lo:
            # Re-centre around 1 on [lo, hi]: 1 - c*p(lo) = c*p(hi) - 1.
            p *= 2 / (p(lo) + p(hi))
        if i < SCHEDULE_STEPS - 1:  # never on the last polynomial
            p = p(Polynomial.identity() / safety)
        coefficients.append(tuple(float(x) for x in p.coef[1::2]))
        lo = p(lo)
        hi = 2 - lo
    return tuple(coefficients)
