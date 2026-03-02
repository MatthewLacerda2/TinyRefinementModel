import jax
import jax.numpy as jnp
from flax import nnx

from train_local import MAX_STEPS_LIMIT, UniversalReasoner


def run_model_inference(
    model: UniversalReasoner,
    tokens: jnp.ndarray,
    max_steps: int = MAX_STEPS_LIMIT,
    threshold: float = 0.5,
) -> jnp.ndarray:
    """Run the UniversalReasoner in inference mode, reusing its internal logic."""
    z_seq, z_shared, z_output, all_time_embeds, ctx = model._prepare_reasoning_context(
        tokens, max_steps
    )

    def scan_step(state, t_signal):
        step, curr_seq, curr_shared, curr_output, halt_prob = state
        has_halted = halt_prob >= threshold

        def full_compute(_):
            new_seq, new_shared, new_output, _, _, new_halt_prob, _ = model._core_reasoning_step(
                curr_seq, curr_shared, curr_output, t_signal, ctx
            )
            return new_seq, new_shared, new_output, new_halt_prob

        def skip_compute(_):
            return curr_seq, curr_shared, curr_output, halt_prob

        # Skip heavy compute once everything in the batch has halted.
        should_run = jnp.any(~has_halted)
        new_seq, new_shared, new_output, new_halt_prob = jax.lax.cond(
            should_run, full_compute, skip_compute, operand=None
        )

        final_seq = jnp.where(has_halted[:, None, None], curr_seq, new_seq)
        final_shared = jnp.where(has_halted[:, None, None], curr_shared, new_shared)
        final_output = jnp.where(has_halted[:, None, None], curr_output, new_output)
        final_halt_prob = jnp.where(has_halted, halt_prob, new_halt_prob)

        final_seq = model.seq_norm(final_seq)
        final_shared = model.shared_norm(final_shared)
        final_output = model.output_norm(final_output)

        return (step + 1, final_seq, final_shared, final_output, final_halt_prob), None

    init_state = (0, z_seq, z_shared, z_output, jnp.zeros((ctx["batch_size"],)))
    scan_fn = nnx.scan(nnx.remat(scan_step), in_axes=(nnx.Carry, 0))
    (_, final_seq, _, _, _), _ = scan_fn(init_state, all_time_embeds)

    logits = final_seq @ model.embed.embedding.value.T
    return logits
