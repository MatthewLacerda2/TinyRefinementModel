import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import tiktoken
from datasets import load_dataset

LATENT_DIM = 384
BATCH_SIZE = 8
ACCUM_STEPS = 16
MAX_STEPS_LIMIT = 8
MAX_SEQ_LEN = 128
VOCAB_SIZE = 100277

SCRATCH_SLOTS = 16
NUM_BRANCHES = 4
BRANCH_NOISE = 0.02

class RotaryAttention(nnx.Module):
    def __init__(self, num_heads, in_features, rngs, dtype=jnp.float32):
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        self.scale = self.head_dim ** -0.5
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.head_dim, 2) / self.head_dim))
        t = jnp.arange(MAX_SEQ_LEN + SCRATCH_SLOTS)
        freqs = jnp.outer(t, inv_freq)
        self.sin_cached = jnp.sin(freqs)
        self.cos_cached = jnp.cos(freqs)
        self.q_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=dtype)
        self.k_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=dtype)
        self.v_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=dtype)
        self.o_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=dtype)

    def __call__(self, x, mask=None):
        b, s, d = x.shape
        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        sin = self.sin_cached[:s, :]
        cos = self.cos_cached[:s, :]
        sin_ext = jnp.stack([sin, sin], axis=-1).reshape(s, self.head_dim)[None, :, None, :]
        cos_ext = jnp.stack([cos, cos], axis=-1).reshape(s, self.head_dim)[None, :, None, :]
        def rotate(x_in):
            x_even, x_odd = x_in[..., 0::2], x_in[..., 1::2]
            x_rot = jnp.stack([-x_odd, x_even], axis=-1).reshape(x_in.shape)
            return x_in * cos_ext + x_rot * sin_ext
        q, k = rotate(q), rotate(k)
        logits = jnp.einsum('bshd,bthd->bhst', q, k) * self.scale
        if mask is not None:
            logits = jnp.where(mask, logits, -1e9)
        weights = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum('bhst,bthd->bshd', weights, v)
        return self.o_proj(out.reshape(b, s, d))

class BranchingReasoningBlock(nnx.Module):
    def __init__(self, latent_dim, num_heads, rngs, dtype=jnp.float32):
        self.latent_dim = latent_dim
        self.attn = RotaryAttention(num_heads, latent_dim, rngs, dtype)
        self.norm1 = nnx.LayerNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.ffn = nnx.Sequential(
            nnx.Linear(latent_dim, latent_dim * 2, rngs=rngs, dtype=dtype),
            nnx.gelu,
            nnx.Linear(latent_dim * 2, latent_dim, rngs=rngs, dtype=dtype)
        )
        self.norm2 = nnx.LayerNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.branch_scorer = nnx.Linear(latent_dim, 1, rngs=rngs, dtype=dtype)

    def __call__(self, x_combined, mask, num_branches=NUM_BRANCHES, key=None):
        res = self.attn(self.norm1(x_combined), mask=mask)
        x_mid = x_combined + res
        x_out = x_mid + self.ffn(self.norm2(x_mid))
        scratch_start = x_combined.shape[1] - SCRATCH_SLOTS
        proposal_full = x_out[:, scratch_start:, :]
        proposals = jnp.tile(proposal_full[:, None, :, :], (1, num_branches, 1, 1))
        if key is not None:
            noise = jax.random.normal(key, proposals.shape) * BRANCH_NOISE
            proposals = proposals + noise
        scores = self.branch_scorer(proposals).mean(axis=(2, 3))
        branch_weights = jax.nn.softmax(scores, axis=-1)
        best_proposal = jnp.einsum('bk,bksd->bsd', branch_weights, proposals)
        return best_proposal, branch_weights

class UniversalReasoner(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim
        dtype = jnp.float32
        self.embed = nnx.Embed(VOCAB_SIZE, latent_dim, dtype=dtype, rngs=rngs)
        self.scratch_token = nnx.Param(jax.random.normal(rngs(), (1, SCRATCH_SLOTS, latent_dim)) * 0.02)
        self.reasoning_penalty_weight = nnx.Param(jnp.array(0.001))
        self.processor = BranchingReasoningBlock(latent_dim, num_heads=8, rngs=rngs, dtype=dtype)
        self.halt_head = nnx.Linear(latent_dim, 1, dtype=dtype, rngs=rngs)
        self.difficulty_estimator = nnx.Linear(latent_dim, 1, dtype=dtype, rngs=rngs)
        self.decoder = nnx.Linear(latent_dim, VOCAB_SIZE, dtype=dtype, rngs=rngs)

    def get_mask(self, seq_len):
        total_len = seq_len + SCRATCH_SLOTS
        mask = jnp.ones((total_len, total_len), dtype=bool)
        causal = jnp.tril(jnp.ones((total_len, total_len), dtype=bool))
        scratch_mask = jnp.arange(total_len) >= seq_len
        final_mask = jnp.logical_or(causal, scratch_mask[:, None])
        return final_mask[None, None, :, :]

    def __call__(self, tokens, max_steps=MAX_STEPS_LIMIT, training=False, key=None):
        batch_size, seq_len = tokens.shape
        z_seq = self.embed(tokens)
        z_scratch = jnp.tile(self.scratch_token.value, (batch_size, 1, 1))
        
        difficulty_logits = self.difficulty_estimator(z_seq).mean(axis=1)
        predicted_steps = nnx.sigmoid(difficulty_logits) * max_steps
        
        if training and key is not None:
            dropout_key, loop_key = jax.random.split(key)
            mask_drop = jax.random.bernoulli(dropout_key, p=0.9, shape=(batch_size, seq_len, 1))
            z_seq_view = z_seq * mask_drop
        else:
            loop_key = None
            z_seq_view = z_seq
            
        if training and loop_key is not None:
            step_keys = jax.random.split(loop_key, max_steps)
        else:
            step_keys = jnp.zeros((max_steps, 2), dtype=jnp.uint32)

        mask = self.get_mask(seq_len)

        def refine_step(carry, step_key):
            z_s, step_idx, cum_prob, accumulated_fused = carry
            
            combined = jnp.concatenate([z_seq_view, z_s], axis=1)
            z_s_new, _ = self.processor(
                combined, mask, key=step_key if training else None
            )
            
            halt_logit = jnp.mean(self.halt_head(z_s_new), axis=1)
            halt_prob = nnx.sigmoid(halt_logit)
            p_halt_now = halt_prob * (1.0 - cum_prob)
            
            combined_readout = jnp.concatenate([z_seq, z_s_new], axis=1)
            fused = self.processor.attn(self.processor.norm1(combined_readout), mask=mask)
            fused = fused + self.processor.ffn(self.processor.norm2(fused))
            fused_seq = fused[:, :seq_len, :]
            
            new_accumulated = accumulated_fused + fused_seq * p_halt_now[:, None, None]
            new_prob = cum_prob + p_halt_now
            
            return (z_s_new, step_idx + 1, new_prob, new_accumulated), p_halt_now

        init_accumulated = jnp.zeros((batch_size, seq_len, self.latent_dim))
        init_carry = (z_scratch, 0, jnp.zeros((batch_size, 1)), init_accumulated)
        
        (final_z_s, _, final_cum_prob, accumulated_fused), step_halts = jax.lax.scan(
            refine_step, init_carry, step_keys, length=max_steps
        )
        
        remainder = 1.0 - final_cum_prob
        combined_last = jnp.concatenate([z_seq, final_z_s], axis=1)
        fused_last = self.processor.attn(self.processor.norm1(combined_last), mask=mask)
        fused_last = fused_last + self.processor.ffn(self.processor.norm2(fused_last))
        fused_seq_last = fused_last[:, :seq_len, :]
        
        final_readout_latent = accumulated_fused + fused_seq_last * remainder[:, None, None]
        logits = self.decoder(final_readout_latent)
        
        step_halts = step_halts.at[-1].add(remainder)
        step_halts_out = jnp.transpose(step_halts, (1, 0, 2))
        
        return logits, step_halts_out, predicted_steps

class TextDataGenerator:
    def __init__(self):
        self.dataset = load_dataset("HuggingFaceTB/cosmopedia-v2", split="train", streaming=True)
        self.iterator = iter(self.dataset)
        self.enc = tiktoken.get_encoding("cl100k_base")

    def get_batch(self, batch_size):
        batch_ids = []
        while len(batch_ids) < batch_size:
            try:
                item = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataset)
                item = next(self.iterator)
            text = item['text']
            tokens = self.enc.encode(text)
            if len(tokens) < MAX_SEQ_LEN:
                tokens = tokens + [100257] * (MAX_SEQ_LEN - len(tokens))
            else:
                tokens = tokens[:MAX_SEQ_LEN]
            batch_ids.append(tokens)
        return jnp.array(batch_ids, dtype=jnp.int32)

@nnx.jit
def train_step(model, optimizer, batch_tokens, key):
    inputs = batch_tokens[:, :-1]
    targets = batch_tokens[:, 1:]
    def loss_fn(model):
        preds, step_halts, pred_steps = model(inputs, training=True, key=key)
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=preds, labels=targets)
        mask = (targets != 100257)
        token_loss = jnp.sum(ce_loss * mask) / jnp.sum(mask)
        steps_taken = jnp.sum(step_halts * jnp.arange(1, MAX_STEPS_LIMIT + 1)[None, :, None], axis=1)
        avg_steps = jnp.mean(steps_taken)
        difficulty_loss = jnp.mean((pred_steps - steps_taken) ** 2)
        w = jax.nn.softplus(model.reasoning_penalty_weight.value)
        total_loss = token_loss + (w * avg_steps) + (0.1 * difficulty_loss)
        return total_loss, (token_loss, avg_steps, w)
    (loss, (raw_ce, steps, w_val)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, raw_ce, steps, w_val

if __name__ == "__main__":
    print(f"🚀 Initializing Branching Latent Reasoner (Dim={LATENT_DIM}, Scratch={SCRATCH_SLOTS})...")
    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(42))
    optimizer = nnx.Optimizer(model, optax.adamw(3e-4), wrt=nnx.Param)
    data_gen = TextDataGenerator()
    key = jax.random.key(0)
    print("Starting training loop...")
    for step in range(1, 10000):
        key, subkey = jax.random.split(key)
        batch = data_gen.get_batch(BATCH_SIZE)
        loss, raw_ce, avg_steps, penalty_w = train_step(model, optimizer, batch, subkey)
        if step % 50 == 0:
            print(f"Step {step:04d} | Loss: {loss:.4f} | CE: {raw_ce:.4f} | Avg Think Steps: {avg_steps:.2f} | Penalty W: {penalty_w:.5f}")
            print("\n--- INFERENCE CHECK ---")
            prompt = "Once upon a time there was a specific"
            tokens = jnp.array([data_gen.enc.encode(prompt)], dtype=jnp.int32)
            logits, _, pred_depth = model(tokens, training=False)
            next_tok = jnp.argmax(logits[0, -1])
            print(f"Input: {prompt}")
            print(f"Predicted Depth: {pred_depth[0].item():.2f}")
            print(f"Next Token: {data_gen.enc.decode([next_tok.item()])}")
            print("-----------------------\n")