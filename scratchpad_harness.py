"""Toy proof harness for the supervised serial latent scratchpad (#38, #62, #67, #63).

Five arms on the chained-affine-maps task (docs/design/serial-scratchpad.md):

  serial    — K ordered sub-slots, each written ONCE by a shared cross-attention
              write block reading tokens + EARLIER slots only, each graded
              against its sub-result r_k. The bet.
  parallel  — identical module and parameters, but all K slots written in one
              step from tokens only (no slot-to-slot flow, no order). The
              matched control: exactly one variable removed.
  depthonly — CausalRefiner at depth K, final-answer supervision only. The
              is-it-just-depth control.
  slotsonly — serial arm, but the answer readout sees ONLY the slots (tokens
              removed from its context; identical parameters again). The #62
              compression probe: if the slots really carry the computation, a
              readout blinded to the tokens costs nothing; if accuracy
              collapses, the readout was secretly re-reading the problem.
  finalonly — serial wiring and parameters, but the slot grade is detached
              (stop-gradient probe): the model is taught by final-answer CE
              only, while the readout head still measures what each slot
              carries. The does-the-decomposition-emerge ablation (#67).

Plus four fixed-budget arms (docs/design/budget-scratchpad.md, #63) — forgetting
by capacity instead of by gate:

  overwrite — BudgetScratchpadNet, S=1 physical slot, K sequential writes, each
              overwriting the last. Same final target as serial (r_K). Phase 1:
              does forgetting-by-overwrite still carry the chain with O(1) memory?
  budget1   — BudgetScratchpadNet, S=1, on the RECALL task (final = (r_1+r_K) mod
              m, needs r_1 held past the point it would normally be overwritten).
              The no-retention-possible control for phase 2.
  budget2   — BudgetScratchpadNet, S=2, same recall task. Must learn to park r_1
              in one slot and let the other churn through r_2..r_K. The bet.
  unlimited — ScratchpadNet (serial, append-all, tokens hidden from the readout)
              on the recall task — the ceiling: nothing needs to be evicted.

Task: r_0 = 0; r_k = (r_{k-1} * a_k + b_k) mod m from tokens [a_1 b_1 ... a_K b_K].
Affine composition mod m is non-commutative — no order-free shortcut — and
decomposes exactly into the K sub-results the slots are graded on.

    python scratchpad_harness.py --arms serial,parallel,depthonly --seeds 0,1,2
    python scratchpad_harness.py --arms overwrite,serial --seeds 0,1,2 --K 4
    python scratchpad_harness.py --arms budget1,budget2,unlimited --seeds 0,1,2 --K 5
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
    read_tokens=False: the answer readout's context is the slots alone — token
                  states never reach it (#62).
    probe_only=True: the slot grade becomes a diagnostic-only linear probe —
                  the slots are stop-gradiented before the readout, so the
                  slot CE can fit the readout head but can never teach the
                  slots. Final-answer CE becomes the model's only supervision
                  (#67), with the slot-by-slot readout kept as the instrument
                  that shows whether the decomposition emerged anyway.
    All modes share the identical parameter tree — the flags only change the
    data flow, which is what makes each comparison a one-variable ablation.
    """

    def __init__(self, *, dim, vocab, num_slots, num_heads=4, num_encoder_layers=2,
                 max_seq_len=64, serial=True, read_tokens=True, probe_only=False,
                 rngs, dtype=jnp.float32):
        self.num_slots = num_slots
        self.serial = serial
        self.read_tokens = read_tokens
        self.probe_only = probe_only
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

        graded = jax.lax.stop_gradient(slots) if self.probe_only else slots
        slot_logits = self.slot_readout(graded)                             # [B, K, m]
        return self.readout(h, slots), slot_logits

    def readout(self, h, slots):
        """Answer logits from the readout context. A separate method so the #62
        wiring guard can probe it directly: with read_tokens=False the token
        states must be unable to influence the answer except through slots."""
        bsz = h.shape[0]
        ctx = jnp.concatenate([h, slots], axis=1) if self.read_tokens else slots
        aq = jnp.broadcast_to(self.answer_query[...].astype(h.dtype), (bsz, 1, h.shape[-1]))
        return self.answer_head(self.read_block(aq, ctx))[:, 0]            # [B, m]


class BudgetScratchpadNet(nnx.Module):
    """Fixed-capacity memory scratchpad (#63, docs/design/budget-scratchpad.md):
    S physical slots shared across K >= S sequential writes. Each step's write
    reads tokens + the CURRENT S-slot memory (not the full history) and produces
    a candidate value; a softmax address over the S slots (computed from the
    step's identity and the memory itself) decides how much of each slot gets
    overwritten:

        addr = softmax(addr_head(concat(query, mean(memory))))    # [B, S]
        memory[i] <- (1 - addr[:, i]) * memory[i] + addr[:, i] * candidate

    S=1 degenerates to pure overwrite: a softmax over a size-1 axis is
    identically 1, so there is no addressing policy to learn, by construction.
    S>1 must learn, from step index and memory content alone, which slot to
    spare and which to let churn -- no forget gate, no forgetting loss, no
    bonus the gradient could ignore; capacity alone makes eviction mandatory.

    read_tokens gates the FINAL answer readout only (same knob as ScratchpadNet).
    Forced False for the recall task variant: m prime makes the affine chain
    invertible, so a tokens-visible readout could recompute r_1 from r_K and
    the tokens directly, without ever consulting memory -- the #62 slots-only
    wiring closes that shortcut so a win can only mean genuine retention.
    """

    def __init__(self, *, dim, vocab, num_steps, num_slots, num_heads=4,
                 num_encoder_layers=2, max_seq_len=64, read_tokens=True,
                 rngs, dtype=jnp.float32):
        self.num_steps = num_steps
        self.num_slots = num_slots
        self.read_tokens = read_tokens
        self.embed = nnx.Embed(vocab, dim, rngs=rngs, dtype=dtype)
        self.encoder = nnx.List([
            Block(dim, num_heads, max_seq_len, rngs, dtype) for _ in range(num_encoder_layers)
        ])
        self.write_block = CrossBlock(dim, num_heads, rngs, dtype)   # shared across k
        self.step_index = nnx.Embed(num_steps, dim, rngs=rngs, dtype=dtype)
        self.slot_readout = nnx.Linear(dim, vocab, rngs=rngs, dtype=dtype)  # the grade head
        self.addr_head = nnx.Linear(2 * dim, num_slots, rngs=rngs, dtype=dtype)
        self.mem_init = nnx.Param(
            jax.nn.initializers.zeros(rngs(), (num_slots, dim), jnp.float32))
        self.read_block = CrossBlock(dim, num_heads, rngs, dtype)
        self.answer_query = nnx.Param(
            jax.nn.initializers.normal(0.02)(rngs(), (1, 1, dim), jnp.float32))
        self.answer_head = nnx.Linear(dim, vocab, rngs=rngs, dtype=dtype)

    def __call__(self, tokens):
        h = self.embed(tokens)
        for blk in self.encoder:
            h = blk(h)
        bsz = tokens.shape[0]
        memory = jnp.broadcast_to(self.mem_init[...].astype(h.dtype),
                                  (bsz, self.num_slots, h.shape[-1]))

        slot_logits = []
        for k in range(self.num_steps):
            q_k = jnp.broadcast_to(self.step_index(jnp.array([k]))[None, :, :],
                                   (bsz, 1, h.shape[-1]))
            ctx = jnp.concatenate([h, memory], axis=1)
            v_k = self.write_block(q_k, ctx)                                # [B, 1, d]
            slot_logits.append(self.slot_readout(v_k))                     # graded on THIS write

            addr_in = jnp.concatenate([q_k[:, 0], memory.mean(axis=1)], axis=-1)
            addr = jax.nn.softmax(self.addr_head(addr_in), axis=-1)        # [B, S]
            memory = (1 - addr[:, :, None]) * memory + addr[:, :, None] * v_k

        slot_logits = jnp.concatenate(slot_logits, axis=1)                 # [B, K, m]
        return self.readout(h, memory), slot_logits

    def readout(self, h, memory):
        """Same contract as ScratchpadNet.readout: read_tokens=False makes the
        final answer reachable ONLY through the memory bank (#62 wiring)."""
        bsz = h.shape[0]
        ctx = jnp.concatenate([h, memory], axis=1) if self.read_tokens else memory
        aq = jnp.broadcast_to(self.answer_query[...].astype(h.dtype), (bsz, 1, h.shape[-1]))
        return self.answer_head(self.read_block(aq, ctx))[:, 0]            # [B, m]


def n_params(model):
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))


def train_one_arm(arm, *, K=4, m=7, dim=64, heads=4, enc=2, steps=2500, batch=256,
                  lr=2e-3, wd=0.01, seed=0, n_pool=32768, n_test=4096, slot_lambda=1.0):
    task = affine_chain_task(K, m)
    key = jax.random.PRNGKey(seed)
    key, dk_tr, dk_te = jax.random.split(key, 3)
    tr_tok, tr_sub = task(dk_tr, n_pool)
    te_tok, te_sub = task(dk_te, n_test)

    # #63: budget1/budget2/unlimited run the RECALL task variant (final =
    # (r_1+r_K) mod m, slots-only readout — see docs/design/budget-scratchpad.md
    # for why tokens must be hidden here). overwrite stays on the plain chain
    # task (final = r_K), matched against serial exactly as #38 was.
    recall = arm in ("budget1", "budget2", "unlimited")

    if arm == "depthonly":
        model = CausalRefiner(dim=dim, vocab_size=m, num_heads=heads,
                              num_encoder_layers=enc, max_depth=K,
                              max_seq_len=2 * K, rngs=nnx.Rngs(seed))
    elif arm == "overwrite":
        model = BudgetScratchpadNet(dim=dim, vocab=m, num_steps=K, num_slots=1,
                                    num_heads=heads, num_encoder_layers=enc,
                                    max_seq_len=2 * K, read_tokens=True, rngs=nnx.Rngs(seed))
    elif arm in ("budget1", "budget2"):
        model = BudgetScratchpadNet(dim=dim, vocab=m, num_steps=K,
                                    num_slots=(1 if arm == "budget1" else 2),
                                    num_heads=heads, num_encoder_layers=enc,
                                    max_seq_len=2 * K, read_tokens=False, rngs=nnx.Rngs(seed))
    elif arm == "unlimited":
        model = ScratchpadNet(dim=dim, vocab=m, num_slots=K, num_heads=heads,
                              num_encoder_layers=enc, max_seq_len=2 * K,
                              serial=True, read_tokens=False, rngs=nnx.Rngs(seed))
    else:
        model = ScratchpadNet(dim=dim, vocab=m, num_slots=K, num_heads=heads,
                              num_encoder_layers=enc, max_seq_len=2 * K,
                              serial=(arm != "parallel"),
                              read_tokens=(arm != "slotsonly"),
                              probe_only=(arm == "finalonly"), rngs=nnx.Rngs(seed))
    opt = nnx.Optimizer(model, optax.adamw(lr, weight_decay=wd), wrt=nnx.Param)

    def losses(mdl, tok, sub):
        final = (sub[:, 0] + sub[:, -1]) % m if recall else sub[:, -1]
        if arm == "depthonly":
            logits = mdl(tok, depth=K)[:, -1]                    # answer at last position
            ce_final = optax.softmax_cross_entropy_with_integer_labels(logits, final).mean()
            return ce_final, (logits, None)
        answer_logits, slot_logits = mdl(tok)
        ce_final = optax.softmax_cross_entropy_with_integer_labels(answer_logits, final).mean()
        # The grade: λ fixed at 1.0 (pre-registered — see the design doc). In
        # the finalonly arm the slots are stop-gradiented inside the model, so
        # this same term only fits the diagnostic probe head — the model's
        # sole teacher there is ce_final (#67).
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
    ap.add_argument("--arms", default="serial,parallel,depthonly",
                    help="any of serial,parallel,depthonly,slotsonly (#62),finalonly (#67),"
                         "overwrite,budget1,budget2,unlimited (#63)")
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
