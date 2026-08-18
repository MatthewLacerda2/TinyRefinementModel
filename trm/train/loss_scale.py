"""Keep f16 gradients above the floor they would otherwise fall through (#199).

Parameters are f32 but the matmul-heavy backward runs in f16 (`config.COMPUTE_DTYPE`,
a permanent policy on a card with no bf16). f16's smallest subnormal is ~6e-8, so a
gradient below that does not become small — it becomes exactly zero. Silently: no
NaN, no exception, no plateau the loss monitor can see, just a parameter group that
has stopped learning.

That is not a hypothesis. On 2026-08-18 the #157 base run died of it at opt step
11,140. The `gate` — the retention/refine gate that IS the depth recurrence — went
from a 0.002 zero-gradient fraction to **1.000** at step 11,075, `refine_block`
starved alongside it at 26-71%, and 65 steps later the model's output distribution
was uniform (val CE 10.82 against ln(50304) = 10.83). Nothing in the run was ever
non-finite. #82 built the instrument that recorded all of this and nothing was wired
to act on it.

The remedy is the standard one and predates us by years: multiply the loss by S
before the backward pass, divide the gradients by S afterwards. Every intermediate
gradient is S times larger while it is being represented in f16, so the small ones
survive; the update the optimizer finally sees is unchanged. In exact arithmetic
this is a no-op — it changes only what f16 can hold, never what is being optimized,
which is why it is safe to introduce partway through a run (Micikevicius et al.,
"Mixed Precision Training", 2018).

S is found rather than chosen. Too small and the underflow returns; too large and
the backward overflows to inf. Since the right value drifts as the model converges,
we push S upward until it overflows, back off, and repeat — the optimizer's own
behaviour tells us where the ceiling is, so nobody has to pick a constant that is
correct for step 11,000 and wrong for step 25,000.
"""

# Powers of two throughout: scaling by a power of two only shifts the exponent, so
# no mantissa bit is lost on the way in or out. The initial value is the field's
# usual starting point; being wrong here is cheap, since an overflow costs one
# skipped micro-step out of 128 in an accumulation window and immediately corrects.
INITIAL_LOSS_SCALE = 2.0 ** 15
# How many consecutive good micro-steps before trying a larger S. ~2 accumulation
# windows: often enough to track a converging model, rare enough that the periodic
# probe-and-overflow costs a negligible fraction of steps.
LOSS_SCALE_GROWTH_INTERVAL = 256
LOSS_SCALE_FACTOR = 2.0
# Below 1.0 there is nothing left to protect and the real problem is elsewhere;
# above 2**24 an f16 forward would be saturating anyway.
MIN_LOSS_SCALE = 1.0
MAX_LOSS_SCALE = 2.0 ** 24


class DynamicLossScale:
    """The scale S, and the two events that move it.

    Deliberately free of jax: it is a policy, and the trainer's existing
    non-finite branch is the only thing that drives it. Keeping it plain Python
    means the rule can be tested without a GPU, a model, or a jit trace.
    """

    def __init__(self, initial=INITIAL_LOSS_SCALE, growth_interval=LOSS_SCALE_GROWTH_INTERVAL,
                 factor=LOSS_SCALE_FACTOR, min_scale=MIN_LOSS_SCALE, max_scale=MAX_LOSS_SCALE):
        self.value = float(initial)
        self.growth_interval = growth_interval
        self.factor = float(factor)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.good_steps = 0
        self.overflows = 0

    def record_good_step(self):
        """A micro-step whose gradients were finite. Returns True if S grew."""
        self.good_steps += 1
        if self.good_steps < self.growth_interval or self.value >= self.max_scale:
            return False
        self.good_steps = 0
        self.value = min(self.value * self.factor, self.max_scale)
        return True

    def record_overflow(self):
        """A micro-step whose gradients were non-finite: S was too high.

        The caller skips the update, which it already did for non-finite steps
        before this existed — an overflowing micro-step is discarded, not applied
        at a smaller scale, because the gradients themselves are already ruined.
        """
        self.overflows += 1
        self.good_steps = 0
        self.value = max(self.value / self.factor, self.min_scale)
        return self.value

    def __repr__(self):
        return (f"DynamicLossScale(value={self.value:g}, good_steps={self.good_steps}, "
                f"overflows={self.overflows})")
