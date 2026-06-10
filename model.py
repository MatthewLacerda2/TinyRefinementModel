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
)
from layers import (
    BlockStack,
    ScanStepOutput,
    ReasonerOutput,
    calculate_slot_stability_loss,
)

class UniversalReasoner(nnx.Module):
    def __init__(self, latent_dim, rngs, num_blocks=NUM_BLOCKS, dtype=jnp.float32, use_forget=True):
        self.latent_dim = latent_dim
        self.embed = nnx.Embed(VOCAB_SIZE, latent_dim, dtype=dtype, rngs=rngs)
        self.time_embed = nnx.Embed(MAX_STEPS_LIMIT + 1, latent_dim, dtype=dtype, rngs=rngs)

        self.shared_token = nnx.Param(
            jax.nn.initializers.orthogonal()(rngs(), (1, SHARED_SLOTS, latent_dim), jnp.float32)
        )

        self.seq_norm = nnx.RMSNorm(latent_dim, rngs=rngs, dtype=dtype)
        
        self.encoder_stack = BlockStack(num_blocks // 2, latent_dim, num_heads=NUM_HEADS, rngs=rngs, dtype=dtype, share_weights=False, use_remat=True)
        self.decoder_stack = BlockStack(num_blocks // 2, latent_dim, num_heads=NUM_HEADS, rngs=rngs, dtype=dtype, share_weights=False, use_remat=True)
        # use_remat=True: per-block checkpointing within the scan recomputation
        # prevents storing all 8 shared-block intermediates simultaneously.
        # The extra recomputation FLOPs are the correct tradeoff for limited GPU memory.
        self.reasoning_stack = BlockStack(num_blocks, latent_dim, num_heads=NUM_HEADS, rngs=rngs, dtype=dtype, share_weights=True, use_remat=True)

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

        # Lightweight probe: reads mean-pooled slot state, outputs a per-batch halting
        # probability. Trained implicitly by the ponder cost + slot stability losses.
        self.halt_probe = nnx.Linear(latent_dim, 1, rngs=rngs, dtype=dtype)

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

        self.hunch_cache = nnx.Variable(jnp.zeros((BATCH_SIZE, SHARED_SLOTS, latent_dim)))


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
        step_nums = jnp.arange(1, max_steps + 1, dtype=jnp.float32)  # [max_steps]
        
        pad_part = (pad_mask.astype(jnp.float32) - 1.0) * 1e9
        slot_part = jnp.zeros((batch_size, SHARED_SLOTS), dtype=jnp.float32)
        extended_ctx_bias = jnp.concatenate([pad_part, slot_part], axis=-1)[:, None, None, :]

        modules = (
            self.meta_proj, self.time_norm, self.time_signal_norm, 
            self.reasoning_stack,
            self.forget_norm if self.use_forget else None,
            self.forget_head if self.use_forget else None,
            self.raw_tau,
            self.halt_probe,
        )
        model_graph, model_state = nnx.split(modules)

        def scan_step(carry, inputs):
            curr_shared, cumul_expected, remaining_survival, cumul_ponder, prev_forget, prev_div, current_state = carry
            t_signal, step_shared_pos, step_num = inputs
            
            (
                m_proj, t_norm, ts_norm, 
                r_stack,
                f_norm, f_head, raw_tau_param,
                h_probe,
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

            tau = jax.nn.softplus(raw_tau_param.value) + 1e-4

            step_div = calculate_slot_stability_loss(new_shared, curr_shared, tau)

            # Halting probe: reads the mean slot state, outputs a per-batch
            # probability of stopping at this step.
            slot_mean = jnp.mean(new_shared, axis=1)  # [batch, dim]
            halt_prob = jax.nn.sigmoid(h_probe(slot_mean)).squeeze(-1)  # [batch]

            # ── In-carry ACT accumulation ─────────────────────────────────────
            # p_t = h_t * survival_t  (survival = prob of not having halted yet)
            # Instead of materializing the full trajectory for the backward,
            # we accumulate expected_shared and ponder_cost in the carry.
            # This keeps the gradient flowing only through the carry path, which
            # jax.lax.scan's reverse-mode is optimized to handle efficiently.
            step_weight = halt_prob * remaining_survival               # [B]
            new_cumul_expected = cumul_expected + step_weight[:, None, None] * new_shared
            new_remaining = remaining_survival * (1.0 - halt_prob)
            new_cumul_ponder = cumul_ponder + step_weight * step_num

            _, next_state = nnx.split((m_proj, t_norm, ts_norm, r_stack, f_norm, f_head, raw_tau_param, h_probe))

            return (
                new_shared, new_cumul_expected, new_remaining, new_cumul_ponder,
                forget_val, step_div, next_state
            ), ScanStepOutput(
                shared_state=new_shared,
                forget_val=forget_val,
                step_div=step_div,
                halt_prob=halt_prob,
            )

        init_carry = (
            z_shared,
            jnp.zeros_like(z_shared),          # cumul_expected [B, S, D]
            jnp.ones((batch_size,)),            # remaining_survival [B] starts at 1
            jnp.zeros((batch_size,)),           # cumul_ponder [B]
            jnp.zeros((batch_size,)),           # prev_forget [B]
            jnp.zeros((batch_size,)),           # prev_div [B]
            model_state
        )

        final_carry, all_outputs = jax.lax.scan(
            jax.checkpoint(scan_step), init_carry,
            (all_time_embeds, all_step_shared_pos, step_nums)
        )

        final_shared, cumul_expected, remaining, cumul_ponder = final_carry[:4]

        # The remaining survival probability mass is assigned to the last step's state.
        # This ensures the weights sum to exactly 1 without any post-scan computation.
        expected_shared = cumul_expected + remaining[:, None, None] * final_shared
        ponder_cost = jnp.mean(cumul_ponder + remaining * float(max_steps))

        return expected_shared, ponder_cost, all_outputs


    def __call__(self, tokens, max_steps=MAX_STEPS_LIMIT, training=False, should_refresh=True):
        batch_size = tokens.shape[0]
        self.encoder_stack.reset_state()
        self.reasoning_stack.reset_state()
        self.decoder_stack.reset_state()
        
        z_seq, pad_mask, seq_pos = self._encode_sequence(tokens, training=training)

        z_shared_base = jnp.tile(self.shared_token.value, (batch_size, 1, 1))
        
        def get_fresh():
            return z_shared_base
            
        def get_carried():
            current_hunch = self.hunch_cache.value
            seq_context = jnp.mean(z_seq, axis=1, keepdims=True)
            seq_context = jnp.tile(seq_context, (1, SHARED_SLOTS, 1))
            gate_in = jnp.concatenate([self.hunch_norm(current_hunch), self.hunch_norm(seq_context)], axis=-1)
            gate = jax.nn.sigmoid(self.hunch_gate(gate_in))
            return gate * current_hunch + (1.0 - gate) * z_shared_base
        
        z_shared = jax.lax.cond(should_refresh, get_fresh, get_carried)

        # Decoder slots use negative positions, which index into the tail of the
        # extended RoPE cache — distinct from all query/key positions in the reasoning loop.
        past_shared_pos = jnp.arange(-SHARED_SLOTS, 0)
        decoder_kv_pos = jnp.concatenate([seq_pos, past_shared_pos], axis=0)
        
        decoder_pad_mask = jnp.concatenate([pad_mask, jnp.ones((batch_size, SHARED_SLOTS), dtype=jnp.bool_)], axis=1)
        decoder_bias = (decoder_pad_mask.astype(jnp.float32) - 1.0) * 1e9
        decoder_bias = decoder_bias[:, None, None, :]

        expected_shared, ponder_cost, all_outputs = self._reasoning_loop(
            z_seq, pad_mask, seq_pos, max_steps, batch_size, z_shared, training=training
        )
        
        # Decoder reads from the soft-halting weighted slot state — a quality-weighted
        # summary of the full reasoning trajectory, not just the final step.
        decoder_ctx_final = jnp.concatenate([z_seq, expected_shared], axis=1)
        
        z_seq_out = self.decoder_stack(
            z_seq, 
            context=decoder_ctx_final, 
            mask=decoder_bias, 
            q_pos=seq_pos, 
            kv_pos=decoder_kv_pos,
            is_causal=True,
            training=training
        )
        logits = self.seq_norm(z_seq_out) @ self.embed.embedding.value.T
        
        total_f_cost = jnp.mean(jnp.sum(all_outputs.forget_val, axis=0))
        total_div_cost = jnp.mean(jnp.sum(all_outputs.step_div, axis=0))
        
        states = all_outputs.shared_state
        diffs = states[1:] - states[:-1]
        temporal_drift = jnp.mean(jnp.sqrt(jnp.sum(jnp.square(diffs), axis=-1) + 1e-8))
        
        halt_diag = {
            'temporal_drift': temporal_drift,
            'forget_density': jnp.mean(all_outputs.forget_val),
            'tau': jax.nn.softplus(self.raw_tau.value) + 1e-4,
            'mean_halt_step': ponder_cost,
        }
        
        # Carry the soft-weighted slot state forward — a better scratchpad summary
        # than the raw final step (which may have overwritten useful intermediate states).
        self.hunch_cache.value = expected_shared

        return ReasonerOutput(
            logits=logits,
            forget_cost=total_f_cost,
            diversity_loss=total_div_cost,
            ponder_cost=ponder_cost,
            halt_diag=halt_diag,
            expected_shared=expected_shared,
        )
