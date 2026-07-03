"""Toy proof harness for the supervised serial latent scratchpad (#38).

Three arms on the chained-affine-maps task (docs/design/serial-scratchpad.md):

  serial    — K ordered sub-slots, each written ONCE by a shared cross-attention
              write block reading tokens + EARLIER slots only, each graded
              against its sub-result r_k. The bet.
  parallel  — identical module and parameters, but all K slots written in one
              step from tokens only (no slot-to-slot flow, no order). The
              matched control: exactly one variable removed.
  depthonly — CausalRefiner at depth K, final-answer supervision only. The
              is-it-just-depth control.

Task: r_0 = 0; r_k = (r_{k-1} * a_k + b_k) mod m from tokens [a_1 b_1 ... a_K b_K].
Affine composition mod m is non-commutative — no order-free shortcut — and
decomposes exactly into the K sub-results the slots are graded on.

    python scratchpad_harness.py --arms serial,parallel,depthonly --seeds 0,1,2
"""

import argparse
import time

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from plan_a_model import Block, CausalRefiner


def affine_chain_task(K, m):
    """tokens [B, 2K] = a_1 b_1 ... a_K b_K (a nonzero); sub-targets r_1..r_K
    with r_k = (r_{k-1} * a_k + b_k) mod m; the final answer is r_K."""
    def fn(key, batch):
        ka, kb = jax.random.split(key)
        a = jax.random.randint(ka, (batch, K), 1, m)   # nonzero -> invertible maps
        b = jax.random.randint(kb, (batch, K), 0, m)
        tokens = jnp.stack([a, b], axis=-1).reshape(batch, 2 * K)

        def step(r, ab):                                # ab: [B, 2] = (a_k, b_k)
            r_new = (r * ab[:, 0] + ab[:, 1]) % m
            return r_new, r_new
        _, subs = jax.lax.scan(step, jnp.zeros(batch, jnp.int32),
                               jnp.stack([a.T, b.T], axis=-1).astype(jnp.int32))
        sub_targets = subs.T                            # [B, K]
        return tokens.astype(jnp.int32), sub_targets
    return fn


class CrossBlock(nnx.Module):
    """Pre-norm cross-attention + SwiGLU: queries read a separate context. No
    positional encoding on the queries — slot identity comes from the caller's
    slot-index embedding; the context carries position from the causal encoder."""

    def __init__(self, dim, num_heads, rngs, dtype=jnp.float32):
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype)
        self.k = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype)
        self.v = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype)
        self.o = nnx.Linear(dim, dim, rngs=rngs, dtype=dtype)
        self.q_norm = nnx.RMSNorm(self.head_dim, epsilon=1e-6, rngs=rngs, dtype=jnp.float32)
        self.k_norm = nnx.RMSNorm(self.head_dim, epsilon=1e-6, rngs=rngs, dtype=jnp.float32)
        self.norm_q = nnx.RMSNorm(dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        self.norm_kv = nnx.RMSNorm(dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        self.norm2 = nnx.RMSNorm(dim, epsilon=1e-6, rngs=rngs, dtype=dtype)
        hidden = ((int(8 * dim / 3) + 63) // 64) * 64
        self.gate_proj = nnx.Linear(dim, hidden, rngs=rngs, dtype=dtype)
        self.up_proj = nnx.Linear(dim, hidden, rngs=rngs, dtype=dtype)
        self.down_proj = nnx.Linear(hidden, dim, kernel_init=jax.nn.initializers.zeros, rngs=rngs, dtype=dtype)

    def __call__(self, q_in, kv_in):
        b, nq, d = q_in.shape
        s = kv_in.shape[1]
        qn, kvn = self.norm_q(q_in), self.norm_kv(kv_in)
        q = self.q_norm(self.q(qn).reshape(b, nq, self.num_heads, self.head_dim))
        k = self.k_norm(self.k(kvn).reshape(b, s, self.num_heads, self.head_dim))
        v = self.v(kvn).reshape(b, s, self.num_heads, self.head_dim)
        q, k = q.astype(q_in.dtype), k.astype(q_in.dtype)
        out = jax.nn.dot_product_attention(q, k, v)
        x = q_in + self.o(out.reshape(b, nq, d))
        h = self.norm2(x)
        return x + self.down_proj(jax.nn.silu(self.gate_proj(h)) * self.up_proj(h))


class ScratchpadNet(nnx.Module):
    """embed -> causal encoder -> K slot writes (serial or parallel) -> readout.

    serial=True:  slot k's write context is tokens + slots 1..k-1 (order by
                  construction; each slot written exactly once, then frozen).
    serial=False: all K slots written in ONE call, context is tokens only.
    Both modes share the identical parameter tree — `serial` only changes the
    data flow, which is what makes serial-vs-parallel a one-variable ablation.
    """

    def __init__(self, *, dim, vocab, num_slots, num_heads=4, num_encoder_layers=2,
                 max_seq_len=64, serial=True, rngs, dtype=jnp.float32):
        self.num_slots = num_slots
        self.serial = serial
        self.embed = nnx.Embed(vocab, dim, rngs=rngs, dtype=dtype)
        self.encoder = nnx.List([
            Block(dim, num_heads, max_seq_len, rngs, dtype) for _ in range(num_encoder_layers)
        ])
        self.write_block = CrossBlock(dim, num_heads, rngs, dtype)   # shared across k
        self.slot_index = nnx.Embed(num_slots, dim, rngs=rngs, dtype=dtype)
        self.slot_readout = nnx.Linear(dim, vocab, rngs=rngs, dtype=dtype)  # the grade head
        self.read_block = CrossBlock(dim, num_heads, rngs, dtype)
        self.answer_query = nnx.Param(
            jax.nn.initializers.normal(0.02)(rngs(), (1, 1, dim), jnp.float32))
        self.answer_head = nnx.Linear(dim, vocab, rngs=rngs, dtype=dtype)

    def __call__(self, tokens):
        h = self.embed(tokens)
        for blk in self.encoder:
            h = blk(h)
        bsz = tokens.shape[0]
        queries = self.slot_index(jnp.arange(self.num_slots))[None, :, :]   # [1, K, d]
        queries = jnp.broadcast_to(queries, (bsz, self.num_slots, queries.shape[-1]))

        if self.serial:
            slots = []
            for k in range(self.num_slots):
                ctx = jnp.concatenate([h] + slots, axis=1) if slots else h
                slots.append(self.write_block(queries[:, k:k + 1], ctx))    # written once
            slots = jnp.concatenate(slots, axis=1)                          # [B, K, d]
        else:
            slots = self.write_block(queries, h)                            # one step, tokens only

        slot_logits = self.slot_readout(slots)                              # [B, K, m]
        aq = jnp.broadcast_to(self.answer_query[...].astype(h.dtype), (bsz, 1, h.shape[-1]))
        a = self.read_block(aq, jnp.concatenate([h, slots], axis=1))
        answer_logits = self.answer_head(a)[:, 0]                           # [B, m]
        return answer_logits, slot_logits


def n_params(model):
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))


def train_one_arm(arm, *, K=4, m=7, dim=64, heads=4, enc=2, steps=2500, batch=256,
                  lr=2e-3, wd=0.01, seed=0, n_pool=32768, n_test=4096, slot_lambda=1.0):
    task = affine_chain_task(K, m)
    key = jax.random.PRNGKey(seed)
    key, dk_tr, dk_te = jax.random.split(key, 3)
    tr_tok, tr_sub = task(dk_tr, n_pool)
    te_tok, te_sub = task(dk_te, n_test)

    if arm == "depthonly":
        model = CausalRefiner(dim=dim, vocab_size=m, num_heads=heads,
                              num_encoder_layers=enc, max_depth=K,
                              max_seq_len=2 * K, rngs=nnx.Rngs(seed))
    else:
        model = ScratchpadNet(dim=dim, vocab=m, num_slots=K, num_heads=heads,
                              num_encoder_layers=enc, max_seq_len=2 * K,
                              serial=(arm == "serial"), rngs=nnx.Rngs(seed))
    opt = nnx.Optimizer(model, optax.adamw(lr, weight_decay=wd), wrt=nnx.Param)

    def losses(mdl, tok, sub):
        final = sub[:, -1]
        if arm == "depthonly":
            logits = mdl(tok, depth=K)[:, -1]                    # answer at last position
            ce_final = optax.softmax_cross_entropy_with_integer_labels(logits, final).mean()
            return ce_final, (logits, None)
        answer_logits, slot_logits = mdl(tok)
        ce_final = optax.softmax_cross_entropy_with_integer_labels(answer_logits, final).mean()
        # The grade: λ fixed at 1.0 (pre-registered — see the design doc).
        ce_slots = optax.softmax_cross_entropy_with_integer_labels(slot_logits, sub).mean()
        return ce_final + slot_lambda * ce_slots, (answer_logits, slot_logits)

    @nnx.jit
    def step(mdl, op, k):
        idx = jax.random.randint(k, (batch,), 0, tr_tok.shape[0])
        def loss_fn(mm):
            loss, _ = losses(mm, tr_tok[idx], tr_sub[idx])
            return loss
        loss, grads = nnx.value_and_grad(loss_fn)(mdl)
        op.update(mdl, grads)
        return loss

    @nnx.jit
    def eval_all(mdl):
        _, (answer_logits, slot_logits) = losses(mdl, te_tok, te_sub)
        final_acc = jnp.mean(answer_logits.argmax(-1) == te_sub[:, -1])
        slot_acc = (jnp.mean(slot_logits.argmax(-1) == te_sub, axis=0)
                    if slot_logits is not None else jnp.zeros(K))
        return final_acc, slot_acc

    for _ in range(steps):
        key, k = jax.random.split(key)
        step(model, opt, k)

    final_acc, slot_acc = eval_all(model)
    return float(final_acc), [float(x) for x in slot_acc], n_params(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="serial,parallel,depthonly")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--K", type=int, default=4, help="number of chained sub-steps = number of slots")
    ap.add_argument("--m", type=int, default=7, help="modulus (prime); vocab and chance level 1/m")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--dim", type=int, default=64)
    args = ap.parse_args()

    print(f"== serial-scratchpad proof (#38): K={args.K} m={args.m} dim={args.dim} "
          f"steps={args.steps} (chance={1/args.m:.3f}) ==")
    print(f"{'arm':>10} {'seed':>5} {'params':>9} {'final_acc':>10} {'sec':>7}  slot_accs")
    for arm in args.arms.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            t0 = time.time()
            acc, slot_acc, params = train_one_arm(arm, K=args.K, m=args.m,
                                                  dim=args.dim, steps=args.steps, seed=seed)
            slots = " ".join(f"{a:.3f}" for a in slot_acc) if any(slot_acc) else "-"
            print(f"{arm:>10} {seed:>5} {params/1e6:>8.2f}M {acc:>10.4f} {time.time()-t0:>7.1f}  [{slots}]",
                  flush=True)


if __name__ == "__main__":
    main()
