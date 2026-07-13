import jax
import jax.numpy as jnp
from flax import nnx
from config import (
    NUM_BLOCKS,
    SHARED_SLOTS,
    MAX_SEQ_LEN,
    VOCAB_SIZE,
    MAX_STEPS_LIMIT,
    BATCH_SIZE,
    PAD_TOKEN_ID,
    NUM_HEADS,
    COMPUTE_DTYPE,
)
from layers import (
    BlockStack,
    ScanStepOutput,
    ReasonerOutput,
    calculate_slot_stability_loss,
)

class UniversalReasoner(nnx.Module):
    def __init__(self, latent_dim, rngs, num_blocks=NUM_BLOCKS, dtype=jnp.float32, use_forget=True, batch_size=BATCH_SIZE):
        self.latent_dim = latent_dim
        self.embed = nnx.Embed(VOCAB_SIZE, latent_dim, dtype=dtype, rngs=rngs)
        self.time_embed = nnx.Embed(MAX_STEPS_LIMIT + 1, latent_dim, dtype=dtype, rngs=rngs)

        self.shared_token = nnx.Param(
            jax.nn.initializers.orthogonal()(rngs(), (1, SHARED_SLOTS, latent_dim), jnp.float32)
        )

        self.seq_norm = nnx.RMSNorm(latent_dim, rngs=rngs, dtype=dtype)
        
        # use_remat=False everywhere: benchmarked 2026-06-10 (tools/bench_train_step.py,
        # depth 8) — per-block remat cost 18% step time and saved no measurable VRAM.
        # The reasoning stack's memory is already bounded by the scan-level
        # jax.checkpoint in _reasoning_loop; per-block remat inside it was double
        # checkpointing (322ms -> 271ms/micro-step from dropping it alone).
        self.encoder_stack = BlockStack(num_blocks // 2, latent_dim, num_heads=NUM_HEADS, rngs=rngs, dtype=dtype, share_weights=False, use_remat=False)
        self.decoder_stack = BlockStack(num_blocks // 2, latent_dim, num_heads=NUM_HEADS, rngs=rngs, dtype=dtype, share_weights=False, use_remat=False)
        self.reasoning_stack = BlockStack(num_blocks, latent_dim, num_heads=NUM_HEADS, rngs=rngs, dtype=dtype, share_weights=True, use_remat=False)

        self.meta_proj = nnx.Linear(2, latent_dim, rngs=rngs, dtype=dtype)
        
        self.time_norm = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        self.forget_norm = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        self.time_signal_norm = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)

        self.hunch_norm = nnx.RMSNorm(latent_dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        self.hunch_gate = nnx.Linear(
            2 * latent_dim, latent_dim,
            bias_init=jax.nn.initializers.constant(-2.0),
            rngs=rngs, dtype=dtype,
        )

        self.raw_tau = nnx.Param(jnp.array(-2.3))

        self.use_forget = use_forget
        if self.use_forget:
            self.forget_head = nnx.Linear(
                2 * latent_dim, latent_dim,
                # Initialized to zero → sigmoid(0) = 0.5, balanced retention/overwrite.
                # Was +2.0 (sigmoid ≈ 0.88), which caused a cold-start trap where the
                # retention branch never received meaningful gradient early in training.
                bias_init=jax.nn.initializers.zeros,
                rngs=rngs, dtype=dtype
            )

        # Stateful cross-segment memory: sized at construction, so the model only
        # accepts batches of exactly this size (asserted in __call__).
        self.hunch_cache = nnx.Variable(jnp.zeros((batch_size, SHARED_SLOTS, latent_dim)))


    def _encode_sequence(self, tokens, training=False):
        pad_mask = tokens != PAD_TOKEN_ID
        pad_bias = (pad_mask.astype(jnp.float32) - 1.0) * 1e9
        pad_bias = pad_bias[:, None, None, :]
        
        seq_len = tokens.shape[1]
        seq_pos = jnp.arange(seq_len)
        
        z_seq_base = self.embed(tokens)
        z_seq = self.encoder_stack(z_seq_base, mask=pad_bias, q_pos=seq_pos, kv_pos=seq_pos, is_causal=True, training=training)
        return z_seq, pad_mask, seq_pos

    def _reasoning_loop(self, z_seq, pad_mask, seq_pos, max_steps, batch_size, z_shared, training=False):
        # Fixed base positions for slot KV keys in the cross-attention context.
        # These stay constant across iterations — slots always appear at the same
        # position as memory keys. Only query positions advance with the iteration.
        base_shared_pos = jnp.arange(MAX_SEQ_LEN, MAX_SEQ_LEN + SHARED_SLOTS)
        shared_kv_pos = jnp.concatenate([seq_pos, base_shared_pos])

        # Precompute per-step slot query positions at trace time (Python loop).
        # max_steps is a static_argname so this evaluates to a concrete constant array.
        # XLA sees static integer constants rather than a dynamic gather/arithmetic,
        # which significantly reduces compilation time.
        all_step_shared_pos = jnp.array([
            list(range(MAX_SEQ_LEN + i * SHARED_SLOTS, MAX_SEQ_LEN + (i + 1) * SHARED_SLOTS))
            for i in range(max_steps)
        ])  # [max_steps, SHARED_SLOTS]

        all_time_embeds = self.time_embed(jnp.arange(max_steps))

        pad_part = (pad_mask.astype(jnp.float32) - 1.0) * 1e9
        slot_part = jnp.zeros((batch_size, SHARED_SLOTS), dtype=jnp.float32)
        extended_ctx_bias = jnp.concatenate([pad_part, slot_part], axis=-1)[:, None, None, :]

        modules = (
            self.meta_proj, self.time_norm, self.time_signal_norm,
            self.reasoning_stack,
            self.forget_norm if self.use_forget else None,
            self.forget_head if self.use_forget else None,
            self.raw_tau,
        )
        model_graph, model_state = nnx.split(modules)

        def scan_step(carry, inputs):
            curr_shared, prev_forget, prev_div, current_state = carry
            t_signal, step_shared_pos = inputs

            (
                m_proj, t_norm, ts_norm,
                r_stack,
                f_norm, f_head, raw_tau_param,
            ) = nnx.merge(model_graph, current_state)
            
            meta_input = jnp.stack([prev_forget, prev_div], axis=-1)
            meta_signal = m_proj(meta_input)[:, None, :]
            
            shared_ctx = jnp.concatenate([z_seq, curr_shared], axis=1)

            stack_input = t_norm(curr_shared) + ts_norm(t_signal[None, None, :]) + meta_signal
            new_shared = r_stack(stack_input, context=shared_ctx, mask=extended_ctx_bias, q_pos=step_shared_pos, kv_pos=shared_kv_pos, is_causal=False, training=training)

            if self.use_forget:
                gate_in = jnp.concatenate([f_norm(new_shared), f_norm(curr_shared)], axis=-1)
                forget = jax.nn.sigmoid(f_head(gate_in))
                new_shared = forget * new_shared + (1.0 - forget) * curr_shared
                forget_val = jnp.mean(jnp.abs(forget), axis=(1, 2))
            else:
                forget_val = jnp.zeros((batch_size,))

            tau = jax.nn.softplus(raw_tau_param[...]) + 1e-4

            step_div = calculate_slot_stability_loss(new_shared, curr_shared, tau)

            _, next_state = nnx.split((m_proj, t_norm, ts_norm, r_stack, f_norm, f_head, raw_tau_param))

            return (
                new_shared, forget_val, step_div, next_state
            ), ScanStepOutput(
                shared_state=new_shared,
                forget_val=forget_val,
                step_div=step_div,
            )

        init_carry = (
            z_shared,
            jnp.zeros((batch_size,)),           # prev_forget [B]
            jnp.zeros((batch_size,)),           # prev_div [B]
            model_state
        )

        final_carry, all_outputs = jax.lax.scan(
            jax.checkpoint(scan_step), init_carry,
            (all_time_embeds, all_step_shared_pos)
        )

        final_shared = final_carry[0]
        return final_shared, all_outputs


    def __call__(self, tokens, max_steps=MAX_STEPS_LIMIT, training=False, should_refresh=True):
        batch_size = tokens.shape[0]
        assert batch_size == self.hunch_cache[...].shape[0], (
            f"Batch size {batch_size} does not match the hunch cache built for batch "
            f"size {self.hunch_cache[...].shape[0]}; construct UniversalReasoner with "
            f"batch_size={batch_size}."
        )

        z_seq, pad_mask, seq_pos = self._encode_sequence(tokens, training=training)

        z_shared_base = jnp.tile(self.shared_token[...], (batch_size, 1, 1))
        
        def get_fresh():
            return z_shared_base
            
        def get_carried():
            # The gate may only look at the hunch and the fresh prior. The current
            # window's content must not enter here: these slots feed the decoder at
            # every position, so conditioning on the window mean (as this once did)
            # leaks future tokens into past predictions.
            current_hunch = self.hunch_cache[...]
            gate_in = jnp.concatenate([self.hunch_norm(current_hunch), self.hunch_norm(z_shared_base)], axis=-1)
            gate = jax.nn.sigmoid(self.hunch_gate(gate_in))
            return gate * current_hunch + (1.0 - gate) * z_shared_base
        
        z_shared = jax.lax.cond(should_refresh, get_fresh, get_carried)

        # Decoder slots use negative positions, which index into the tail of the
        # extended RoPE cache — distinct from all query/key positions in the
        # reasoning loop, and semantically right: memory carried from previous
        # windows sits "before" the current sequence.
        past_shared_pos = jnp.arange(-SHARED_SLOTS, 0)
        decoder_kv_pos = jnp.concatenate([seq_pos, past_shared_pos], axis=0)
        
        decoder_pad_mask = jnp.concatenate([pad_mask, jnp.ones((batch_size, SHARED_SLOTS), dtype=jnp.bool_)], axis=1)
        decoder_bias = (decoder_pad_mask.astype(jnp.float32) - 1.0) * 1e9
        decoder_bias = decoder_bias[:, None, None, :]

        final_shared, all_outputs = self._reasoning_loop(
            z_seq, pad_mask, seq_pos, max_steps, batch_size, z_shared, training=training
        )

        # Causality: the decoder reads the slots this window STARTED with (fresh,
        # or the hunch carried from previous windows) — never this window's loop
        # output. The loop reads the whole window bidirectionally, so exposing its
        # output to the decoder leaks future tokens into past predictions (the
        # slot-future-leak post-mortem, ROADMAP graveyard; fixed f24f238). This
        # window's reasoning benefits the NEXT window, via the hunch cache.
        decoder_ctx = jnp.concatenate([z_seq, z_shared], axis=1)

        z_seq_out = self.decoder_stack(
            z_seq,
            context=decoder_ctx,
            mask=decoder_bias, 
            q_pos=seq_pos, 
            kv_pos=decoder_kv_pos,
            is_causal=True,
            training=training
        )
        # LM head in COMPUTE_DTYPE with f32 accumulation: the f32 x f32 matmul ran
        # without tensor cores and was ~half the model's FLOPs (benchmarked
        # 2026-06-10, see docs/PERFORMANCE_PLAN.md P3). Inputs are rounded to f16
        # but products accumulate in f32, so logits stay f32 for the softmax.
        normed = self.seq_norm(z_seq_out).astype(COMPUTE_DTYPE)
        if training:
            # Hand the loss the pre-head states; it projects the LM head per-chunk
            # (chunked CE, losses.py / #19) so the full [b, s, vocab] f32 logits —
            # the activation that OOM'd dim960 (#16) — are never materialized.
            logits, hidden = None, normed
        else:
            embed_t = self.embed.embedding[...].astype(COMPUTE_DTYPE).T
            logits = jnp.matmul(normed, embed_t, preferred_element_type=jnp.float32)
            hidden = None

        total_f_cost = jnp.mean(jnp.sum(all_outputs.forget_val, axis=0))
        total_div_cost = jnp.mean(jnp.sum(all_outputs.step_div, axis=0))
        
        # Average distance the slot state moves between consecutive reasoning steps.
        # Undefined for a 1-step trajectory (no transitions): report 0, not the NaN
        # that jnp.mean over an empty diff produces.
        states = all_outputs.shared_state
        if max_steps > 1:
            diffs = states[1:] - states[:-1]
            temporal_drift = jnp.mean(jnp.sqrt(jnp.sum(jnp.square(diffs), axis=-1) + 1e-8))
        else:
            temporal_drift = jnp.array(0.0)
        
        diag = {
            'temporal_drift': temporal_drift,
            'forget_density': jnp.mean(all_outputs.forget_val),
            'tau': jax.nn.softplus(self.raw_tau[...]) + 1e-4,
        }

        # Carry the final slot state forward as the next segment's hunch.
        self.hunch_cache[...] = final_shared

        return ReasonerOutput(
            logits=logits,
            hidden=hidden,
            forget_cost=total_f_cost,
            diversity_loss=total_div_cost,
            diag=diag,
            final_shared=final_shared,
        )
