"""Toy proof harness for the supervised serial latent scratchpad (#38, #62, #63, #67, #73, #79, #116).

The arms on the chained-affine-maps task (docs/design/serial-scratchpad.md):

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
  annealed  — the graded serial arm, but λ_slot follows a schedule: fully on
              for the first 40% of training, linear decay to zero across
              40–60%, exactly zero for the last 40%. The scaffold-or-crutch
              ablation (#73): #67 proved the grade must be present to
              nucleate the chain — this asks whether, once the chain exists,
              it sustains itself on final-answer loss alone. Held-out
              accuracy is also measured at the grade-off step so decay across
              the grade-free stretch is visible (crutch vs frozen-but-stable).
              #95 parameterizes the schedule on the command line: the arm spec
              `annealed@0.2` starts the decay at 20% of training (window stays
              20%), and `annealed@0.4f0.1` decays to a floor of λ=0.1 and
              holds, instead of going to zero. Plain `annealed` is #73's arm.
  densedepth — CausalRefiner at depth K with the serial arm's supervision
              schedule but NO slots: pass k's state at the answer position is
              graded against sub-result r_k through a dedicated linear head —
              the same grade path shape as the serial arm's slot_readout. The
              is-it-the-grade-or-the-offload ablation (#79): if this matches
              serial, the per-step grade carried the #38 win; if serial beats
              it, external slots add performance beyond identical supervision.
  densedepth_tied — same, but the per-step grade goes through the refiner's
              tied LM head (the #75 per-pass path), which forces the working
              state itself to approximate embed(r_k) each pass. Robustness
              check on the grade-head choice, not the primary arm.

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
import re
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


def variable_chain_task(K, m):
    """Variable-length affine chain (#123): k_eff ~ U{1..K} real links, then
    recognizable pad links (a = b = 0; real links always have a >= 1). The
    sub-target is STATIONARY after the chain ends — r stays r_{k_eff} through
    the pad links — so 'done' is detectable as the state no longer changing,
    and the final answer is sub[:, -1] = r_{k_eff}. Returns (tokens, subs,
    k_eff) so halting evals can correlate the halt step with the true length."""
    def fn(key, batch):
        ka, kb, kk = jax.random.split(key, 3)
        a = jax.random.randint(ka, (batch, K), 1, m)
        b = jax.random.randint(kb, (batch, K), 0, m)
        k_eff = jax.random.randint(kk, (batch,), 1, K + 1)
        live = jnp.arange(K)[None, :] < k_eff[:, None]
        a = jnp.where(live, a, 0)
        b = jnp.where(live, b, 0)
        tokens = jnp.stack([a, b], axis=-1).reshape(batch, 2 * K)

        def step(r, ab):
            r_new = jnp.where(ab[:, 0] > 0, (r * ab[:, 0] + ab[:, 1]) % m, r)
            return r_new, r_new
        _, subs = jax.lax.scan(step, jnp.zeros(batch, jnp.int32),
                               jnp.stack([a.T, b.T], axis=-1).astype(jnp.int32))
        return tokens.astype(jnp.int32), subs.T, k_eff.astype(jnp.int32)
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


def encode_links(embed, encoder, tokens, per_link):
    """Shared encode step. per_link=False: the standard causal encode over the
    whole sequence -> [B, 2K, d]. per_link=True (#116): each (a_k, b_k) link is
    embedded and encoded as its OWN 2-token sequence -> [B, K, 2, d] — there is
    no attention path between links, so link j can reach step k != j only
    through whatever the writes chose to keep in memory. This is the leak fix:
    the #114 rerun showed a full-sequence causal encoder lets the LAST write
    compute the recall answer directly, making every slot budget porous."""
    h = embed(tokens)
    if per_link:
        bsz, seq, d = h.shape
        assert seq % 2 == 0, "per-link encoding expects (a_k, b_k) pairs"
        h = h.reshape(bsz * (seq // 2), 2, d)
    for blk in encoder:
        h = blk(h)
    if per_link:
        return h.reshape(bsz, seq // 2, 2, d)
    return h


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
    local_writer=True (#116): tokens are encoded per link (encode_links), and
                  slot k's write context is link k's two states + slots 1..k-1
                  — the ONLY cross-step channel is the slots. Serial and
                  slots-only-readout by requirement (asserted), since a
                  full-sequence view anywhere would reopen the #114 leak.
    All modes share the identical parameter tree — the flags only change the
    data flow, which is what makes each comparison a one-variable ablation.
    """

    def __init__(self, *, dim, vocab, num_slots, num_heads=4, num_encoder_layers=2,
                 max_seq_len=64, serial=True, read_tokens=True, probe_only=False,
                 local_writer=False, rngs, dtype=jnp.float32):
        assert not local_writer or (serial and not read_tokens), \
            "local_writer requires serial=True and read_tokens=False (#116)"
        self.num_slots = num_slots
        self.serial = serial
        self.read_tokens = read_tokens
        self.probe_only = probe_only
        self.local_writer = local_writer
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
        if self.local_writer:
            assert tokens.shape[1] == 2 * self.num_slots, "one (a_k, b_k) link per slot"
        h = encode_links(self.embed, self.encoder, tokens, self.local_writer)
        bsz = tokens.shape[0]
        queries = self.slot_index(jnp.arange(self.num_slots))[None, :, :]   # [1, K, d]
        queries = jnp.broadcast_to(queries, (bsz, self.num_slots, queries.shape[-1]))

        if self.serial:
            slots = []
            for k in range(self.num_slots):
                base = h[:, k] if self.local_writer else h                  # link k only, or all tokens
                ctx = jnp.concatenate([base] + slots, axis=1) if slots else base
                slots.append(self.write_block(queries[:, k:k + 1], ctx))    # written once
            slots = jnp.concatenate(slots, axis=1)                          # [B, K, d]
        else:
            slots = self.write_block(queries, h)                            # one step, tokens only

        graded = jax.lax.stop_gradient(slots) if self.probe_only else slots
        slot_logits = self.slot_readout(graded)                             # [B, K, m]
        h_read = h[:, -1] if self.local_writer else h   # content unread: local_writer forces read_tokens=False
        return self.readout(h_read, slots), slot_logits

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

    local_writer (#116) closes the OTHER shortcut the #114 rerun exposed: with
    a full-sequence encoder, step k's write sees every link, so the LAST write
    can compute the recall answer itself and carry it past any slot budget
    (budget1 hit 0.897 that way). With local_writer=True each write sees only
    its own link (encode_links) + the current memory -- information can cross
    steps ONLY by surviving in memory. Requires read_tokens=False.
    """

    def __init__(self, *, dim, vocab, num_steps, num_slots, num_heads=4,
                 num_encoder_layers=2, max_seq_len=64, read_tokens=True,
                 local_writer=False, rngs, dtype=jnp.float32):
        assert not (local_writer and read_tokens), \
            "local_writer requires read_tokens=False (#116)"
        self.num_steps = num_steps
        self.num_slots = num_slots
        self.read_tokens = read_tokens
        self.local_writer = local_writer
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
        if self.local_writer:
            assert tokens.shape[1] == 2 * self.num_steps, "one (a_k, b_k) link per step"
        h = encode_links(self.embed, self.encoder, tokens, self.local_writer)
        bsz = tokens.shape[0]
        memory = jnp.broadcast_to(self.mem_init[...].astype(h.dtype),
                                  (bsz, self.num_slots, h.shape[-1]))

        slot_logits = []
        for k in range(self.num_steps):
            q_k = jnp.broadcast_to(self.step_index(jnp.array([k]))[None, :, :],
                                   (bsz, 1, h.shape[-1]))
            tok_ctx = h[:, k] if self.local_writer else h                  # link k only, or all tokens
            ctx = jnp.concatenate([tok_ctx, memory], axis=1)
            v_k = self.write_block(q_k, ctx)                                # [B, 1, d]
            slot_logits.append(self.slot_readout(v_k))                     # graded on THIS write

            addr_in = jnp.concatenate([q_k[:, 0], memory.mean(axis=1)], axis=-1)
            addr = jax.nn.softmax(self.addr_head(addr_in), axis=-1)        # [B, S]
            memory = (1 - addr[:, :, None]) * memory + addr[:, :, None] * v_k

        slot_logits = jnp.concatenate(slot_logits, axis=1)                 # [B, K, m]
        h_read = h[:, -1] if self.local_writer else h   # content unread: local_writer forces read_tokens=False
        return self.readout(h_read, memory), slot_logits

    def readout(self, h, memory):
        """Same contract as ScratchpadNet.readout: read_tokens=False makes the
        final answer reachable ONLY through the memory bank (#62 wiring)."""
        bsz = h.shape[0]
        ctx = jnp.concatenate([h, memory], axis=1) if self.read_tokens else memory
        aq = jnp.broadcast_to(self.answer_query[...].astype(h.dtype), (bsz, 1, h.shape[-1]))
        return self.answer_head(self.read_block(aq, ctx))[:, 0]            # [B, m]


class DenseDepthNet(nnx.Module):
    """CausalRefiner + a dedicated per-step grade head (#79). Pass k's normed
    state at the answer position is graded against r_k through a separate
    Linear — matching the serial arm's slot_readout grade path — so the tied
    LM head's embedding geometry is never forced onto the working state. The
    final answer is still read exactly as depthonly reads it: the last pass's
    state through the tied head at the last position."""

    def __init__(self, *, dim, vocab, K, num_heads=4, num_encoder_layers=2, rngs):
        self.K = K
        self.refiner = CausalRefiner(dim=dim, vocab_size=vocab, num_heads=num_heads,
                                     num_encoder_layers=num_encoder_layers, max_depth=K,
                                     max_seq_len=2 * K, rngs=rngs)
        self.step_readout = nnx.Linear(dim, vocab, rngs=rngs)   # the grade head

    def __call__(self, tokens):
        states, _ = self.refiner(tokens, depth=self.K, return_all_iters=True,
                                 return_all_states=True)         # [K, B, S, dim] (normed)
        step_states = states[:, :, -1, :]                        # pass k at the answer position
        step_logits = self.step_readout(step_states)             # [K, B, m]
        embed_t = self.refiner.embed.embedding[...].astype(self.refiner.dtype).T   # tied head, as depthonly
        answer_logits = jnp.matmul(states[-1, :, -1, :], embed_t,
                                   preferred_element_type=jnp.float32)
        return answer_logits, step_logits


class HaltingScratchpadNet(nnx.Module):
    """#123: serial scratchpad + a halting head scoring 'stop after write k'.

    Halting is a READOUT choice, not a compute cut: all K writes always run
    (the per-slot grade stays on, #67), a per-step answer is read from slots
    1..k after each write, and p = softmax(halt_logits) weights those answers
    in the loss — so min-depth collapse is expressible but never
    architecturally forced. halt_context is THE one variable:

      "trajectory" — the halt query cross-attends slots 1..k: the decision can
                     reread the model's own thinking, the way thinking-token
                     models can (the issue's hypothesis).
      "current"    — the SAME head, context = slot k alone: the decision sees
                     only the latest state — the graveyard-ACT configuration.

    Identical parameter tree either way; only the context width differs.
    The answer readout is slots-only (#62 wiring), so answers can only come
    from memory."""

    def __init__(self, *, dim, vocab, num_slots, num_heads=4, num_encoder_layers=2,
                 max_seq_len=64, halt_context="trajectory", rngs, dtype=jnp.float32):
        assert halt_context in ("trajectory", "current"), f"unknown halt_context {halt_context!r}"
        self.num_slots = num_slots
        self.halt_context = halt_context
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
        self.halt_block = CrossBlock(dim, num_heads, rngs, dtype)
        self.halt_query = nnx.Param(
            jax.nn.initializers.normal(0.02)(rngs(), (1, 1, dim), jnp.float32))
        self.halt_head = nnx.Linear(dim, 1, rngs=rngs, dtype=dtype)

    def __call__(self, tokens):
        h = self.embed(tokens)
        for blk in self.encoder:
            h = blk(h)
        bsz = tokens.shape[0]
        queries = jnp.broadcast_to(self.slot_index(jnp.arange(self.num_slots))[None, :, :],
                                   (bsz, self.num_slots, h.shape[-1]))
        aq = jnp.broadcast_to(self.answer_query[...].astype(h.dtype), (bsz, 1, h.shape[-1]))
        hq = jnp.broadcast_to(self.halt_query[...].astype(h.dtype), (bsz, 1, h.shape[-1]))

        slots, answers, halts = [], [], []
        for k in range(self.num_slots):
            ctx = jnp.concatenate([h] + slots, axis=1) if slots else h
            slots.append(self.write_block(queries[:, k:k + 1], ctx))    # written once
            bank = jnp.concatenate(slots, axis=1)                       # slots 1..k
            answers.append(self.answer_head(self.read_block(aq, bank))[:, 0])
            halt_ctx = bank if self.halt_context == "trajectory" else slots[-1]
            halts.append(self.halt_head(self.halt_block(hq, halt_ctx))[:, 0, 0])

        slot_logits = self.slot_readout(jnp.concatenate(slots, axis=1))  # [B, K, m]
        return jnp.stack(answers, 1), slot_logits, jnp.stack(halts, 1)   # [B,K,m], [B,K,m], [B,K]


def n_params(model):
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))


def grade_lambda(step, total_steps, onset=0.4, decay=0.2, floor=0.0):
    """λ_slot schedule for the annealed arm: fully on until onset (fraction of
    the budget), linear decay to `floor` across the decay window, held at
    `floor` afterwards. The defaults are #73's pre-registered arm — at 2500
    steps: on for 0–1000, decay 1000–1500, zero for 1500–2500. #95 sweeps the
    onset (how early can the grade start to go?) and the floor (does a small
    residual grade buy back the stability that full removal cost?)."""
    # off_from is built by summing the two scaled terms (not (onset+decay) *
    # total_steps): for the #73 defaults this reproduces that run's boundary
    # arithmetic bitwise, so plain `annealed` still replays the recorded result.
    on_until = onset * total_steps
    off_from = on_until + decay * total_steps
    if step < on_until:
        return 1.0
    if step >= off_from:
        return floor
    return 1.0 - (1.0 - floor) * (step - on_until) / (off_from - on_until)


def parse_arm_spec(spec):
    """'annealed@0.2' → onset 0.2; 'annealed@0.4f0.1' → onset 0.4, floor 0.1;
    plain arm names (including plain 'annealed') pass through unchanged.
    A near-miss anneal spec is an error, not a fall-through: silently training
    it as some other arm would print a fully-graded run under an
    annealed-looking label."""
    m = re.fullmatch(r"annealed(?:@(0?\.\d+))?(?:f(0?\.\d+))?", spec)
    if not m:
        if spec.startswith("annealed"):
            raise ValueError(f"bad anneal spec {spec!r} — annealed[@<onset>][f<floor>] "
                             "with fractions in (0,1), e.g. annealed@0.2 or annealed@0.4f0.1")
        return spec, {}
    kw = {}
    if m.group(1):
        kw["anneal_onset"] = float(m.group(1))
    if m.group(2):
        kw["anneal_floor"] = float(m.group(2))
    return "annealed", kw


# #63: budget1/budget2/unlimited run the RECALL task variant — final =
# (r_1 + r_K) mod m with a slots-only readout (see docs/design/budget-scratchpad.md
# for why tokens must be hidden there). overwrite stays on the plain chain
# task (final = r_K), matched against serial exactly as #38 was.
# The _local variants (#116) are the same arms with per-link writer context
# (local_writer=True) — the leak-closed rerun where retention must be real.
RECALL_ARMS = ("budget1", "budget2", "unlimited",
               "budget1_local", "budget2_local", "unlimited_local")


def final_target(arm, sub, m):
    """The answer an arm is trained on — and must be SCORED on. One function
    shared by the loss and the eval so the two can never disagree: the #63
    phase-2 run scored recall arms against r_K while training them on
    (r_1+r_K) mod m, which reads as chance whether the readout failed or
    solved the task, voiding that run's table."""
    return (sub[:, 0] + sub[:, -1]) % m if arm in RECALL_ARMS else sub[:, -1]


def arm_losses(arm, mdl, tok, sub, lam, depth):
    """Loss and logits for one arm. lam is the slot-grade weight λ_slot —
    fixed at 1.0 for every arm except annealed (#73), which passes
    grade_lambda(step). In the finalonly arm the slots are stop-gradiented
    inside the model, so the grade term only fits the diagnostic probe head —
    the model's sole teacher there is the final-answer CE (#67)."""
    final = sub[:, -1]
    if arm == "depthonly":
        logits = mdl(tok, depth=depth)[:, -1]                # answer at last position
        ce_final = optax.softmax_cross_entropy_with_integer_labels(logits, final).mean()
        return ce_final, (logits, None)
    if arm in ("densedepth", "densedepth_tied"):
        # The grade without the offload (#79): every refinement pass k is
        # graded at the answer position against r_k — the serial arm's
        # per-slot supervision delivered to a slotless recurrence.
        # Intermediate results must live in the model's own hidden state;
        # only the loss schedule matches serial (λ fixed at 1.0 — these arms
        # never anneal). densedepth grades through a dedicated head
        # (DenseDepthNet); the tied variant reuses the #75 all-iters path,
        # forcing the state itself toward embed(r_k).
        if arm == "densedepth":
            answer_logits, step_logits = mdl(tok)                  # [B, m], [K, B, m]
        else:
            logits_all, _ = mdl(tok, depth=depth, return_all_iters=True)   # [K, B, S, m]
            step_logits = logits_all[:, :, -1, :]                          # pass k at answer position
            answer_logits = step_logits[-1]
        ce_final = optax.softmax_cross_entropy_with_integer_labels(answer_logits, final).mean()
        ce_steps = optax.softmax_cross_entropy_with_integer_labels(step_logits, sub.T).mean()
        return ce_final + lam * ce_steps, (answer_logits, step_logits.transpose(1, 0, 2))
    answer_logits, slot_logits = mdl(tok)
    final = final_target(arm, sub, answer_logits.shape[-1])
    ce_final = optax.softmax_cross_entropy_with_integer_labels(answer_logits, final).mean()
    ce_slots = optax.softmax_cross_entropy_with_integer_labels(slot_logits, sub).mean()
    return ce_final + lam * ce_slots, (answer_logits, slot_logits)


def train_one_arm(arm, *, K=4, m=7, dim=64, heads=4, enc=2, steps=2500, batch=256,
                  lr=2e-3, wd=0.01, seed=0, n_pool=32768, n_test=4096,
                  anneal_onset=0.4, anneal_decay=0.2, anneal_floor=0.0):
    assert arm in ("serial", "parallel", "depthonly", "slotsonly", "finalonly",
                   "annealed", "densedepth", "densedepth_tied",
                   "overwrite", "budget1", "budget2", "unlimited",
                   "budget1_local", "budget2_local", "unlimited_local"), f"unknown arm {arm!r}"
    task = affine_chain_task(K, m)
    key = jax.random.PRNGKey(seed)
    key, dk_tr, dk_te = jax.random.split(key, 3)
    tr_tok, tr_sub = task(dk_tr, n_pool)
    te_tok, te_sub = task(dk_te, n_test)

    if arm in ("depthonly", "densedepth_tied"):
        model = CausalRefiner(dim=dim, vocab_size=m, num_heads=heads,
                              num_encoder_layers=enc, max_depth=K,
                              max_seq_len=2 * K, rngs=nnx.Rngs(seed))
    elif arm == "densedepth":
        model = DenseDepthNet(dim=dim, vocab=m, K=K, num_heads=heads,
                              num_encoder_layers=enc, rngs=nnx.Rngs(seed))
    elif arm == "overwrite":
        model = BudgetScratchpadNet(dim=dim, vocab=m, num_steps=K, num_slots=1,
                                    num_heads=heads, num_encoder_layers=enc,
                                    max_seq_len=2 * K, read_tokens=True, rngs=nnx.Rngs(seed))
    elif arm in ("budget1", "budget2", "budget1_local", "budget2_local"):
        model = BudgetScratchpadNet(dim=dim, vocab=m, num_steps=K,
                                    num_slots=(1 if arm.startswith("budget1") else 2),
                                    num_heads=heads, num_encoder_layers=enc,
                                    max_seq_len=2 * K, read_tokens=False,
                                    local_writer=arm.endswith("_local"), rngs=nnx.Rngs(seed))
    elif arm in ("unlimited", "unlimited_local"):
        model = ScratchpadNet(dim=dim, vocab=m, num_slots=K, num_heads=heads,
                              num_encoder_layers=enc, max_seq_len=2 * K,
                              serial=True, read_tokens=False,
                              local_writer=(arm == "unlimited_local"), rngs=nnx.Rngs(seed))
    else:
        model = ScratchpadNet(dim=dim, vocab=m, num_slots=K, num_heads=heads,
                              num_encoder_layers=enc, max_seq_len=2 * K,
                              serial=(arm != "parallel"),
                              read_tokens=(arm != "slotsonly"),
                              probe_only=(arm == "finalonly"), rngs=nnx.Rngs(seed))
    opt = nnx.Optimizer(model, optax.adamw(lr, weight_decay=wd), wrt=nnx.Param)

    @nnx.jit
    def step(mdl, op, k, lam):
        idx = jax.random.randint(k, (batch,), 0, tr_tok.shape[0])
        def loss_fn(mm):
            loss, _ = arm_losses(arm, mm, tr_tok[idx], tr_sub[idx], lam, K)
            return loss
        loss, grads = nnx.value_and_grad(loss_fn)(mdl)
        op.update(mdl, grads)
        return loss

    @nnx.jit
    def eval_all(mdl):
        _, (answer_logits, slot_logits) = arm_losses(arm, mdl, te_tok, te_sub, 1.0, K)
        final_acc = jnp.mean(answer_logits.argmax(-1) == final_target(arm, te_sub, m))
        slot_acc = (jnp.mean(slot_logits.argmax(-1) == te_sub, axis=0)
                    if slot_logits is not None else jnp.zeros(K))
        return final_acc, slot_acc

    # The grade-off step: where the annealed λ_slot reaches its floor. Every
    # arm is evaluated here too, so annealed-vs-control is a matched comparison
    # at both checkpoints and decay across the grade-free stretch is measurable.
    # Non-annealed arms keep #73's 0.6 checkpoint as a plain mid-training probe.
    cut_frac_steps = (anneal_onset * steps + anneal_decay * steps
                      if arm == "annealed" else 0.6 * steps)
    cut = min(int(round(cut_frac_steps)), steps - 1)
    cut_final = cut_slots = None
    for i in range(steps):
        if i == cut:
            cut_final, cut_slots = eval_all(model)
        key, k = jax.random.split(key)
        step(model, opt, k,
             grade_lambda(i, steps, anneal_onset, anneal_decay, anneal_floor)
             if arm == "annealed" else 1.0)

    final_acc, slot_acc = eval_all(model)
    return {
        "final_acc": float(final_acc),
        "slot_acc": [float(x) for x in slot_acc],
        "cut_step": cut,
        "cut_final_acc": float(cut_final),
        "cut_slot_acc": [float(x) for x in cut_slots],
        "params": n_params(model),
    }


# #123: the halting arms run the variable-length chain. halt_traj vs
# halt_state is the observability comparison; halt_off is the rule-4 ceiling
# (read out at K, halting head untrained).
HALT_ARMS = ("halt_traj", "halt_state", "halt_off")


def halt_task_data(K, m, seed, n_pool=32768, n_test=4096):
    """The exact train/test draw `train_one_halt_arm` makes for `seed` —
    factored so eval-only consumers (the #39 ladder) see the same split
    without retraining. Returns (key, tr_tok, tr_sub, te_tok, te_sub, te_keff)
    with `key` already advanced past the data split, ready for training."""
    task = variable_chain_task(K, m)
    key = jax.random.PRNGKey(seed)
    key, dk_tr, dk_te = jax.random.split(key, 3)
    tr_tok, tr_sub, _ = task(dk_tr, n_pool)
    te_tok, te_sub, te_keff = task(dk_te, n_test)
    return key, tr_tok, tr_sub, te_tok, te_sub, te_keff


def train_one_halt_arm(arm, *, K=6, m=7, dim=64, heads=4, enc=2, steps=2500,
                       batch=256, lr=2e-3, wd=0.01, seed=0, lam_slot=1.0,
                       lam_ponder=0.2, n_pool=32768, n_test=4096,
                       return_model=False):
    assert arm in HALT_ARMS, f"unknown halt arm {arm!r}"
    key, tr_tok, tr_sub, te_tok, te_sub, te_keff = halt_task_data(
        K, m, seed, n_pool=n_pool, n_test=n_test)

    model = HaltingScratchpadNet(
        dim=dim, vocab=m, num_slots=K, num_heads=heads, num_encoder_layers=enc,
        max_seq_len=2 * K,
        halt_context=("current" if arm == "halt_state" else "trajectory"),
        rngs=nnx.Rngs(seed))
    opt = nnx.Optimizer(model, optax.adamw(lr, weight_decay=wd), wrt=nnx.Param)
    step_frac = (jnp.arange(K, dtype=jnp.float32) + 1.0) / K   # ponder cost per halt step

    def halt_losses(mdl, tok, sub):
        answers, slot_logits, halt_logits = mdl(tok)
        target = sub[:, -1]                                    # r_{k_eff} (stationary tail)
        ce_k = optax.softmax_cross_entropy_with_integer_labels(
            answers, jnp.broadcast_to(target[:, None], target.shape + (K,)))   # [B, K]
        ce_slots = optax.softmax_cross_entropy_with_integer_labels(slot_logits, sub).mean()
        if arm == "halt_off":
            # Ceiling: full-depth readout, no halting pressure. The halting
            # head's outputs are unused, so its params receive no gradient.
            return ce_k[:, -1].mean() + lam_slot * ce_slots, (answers, halt_logits)
        p = jax.nn.softmax(halt_logits, axis=-1)               # [B, K]
        ce_halted = (p * ce_k).sum(-1).mean()
        ponder = (p * step_frac[None, :]).sum(-1).mean()
        return ce_halted + lam_slot * ce_slots + lam_ponder * ponder, (answers, halt_logits)

    @nnx.jit
    def step(mdl, op, k):
        idx = jax.random.randint(k, (batch,), 0, tr_tok.shape[0])
        def loss_fn(mm):
            loss, _ = halt_losses(mm, tr_tok[idx], tr_sub[idx])
            return loss
        loss, grads = nnx.value_and_grad(loss_fn)(mdl)
        op.update(mdl, grads)
        return loss

    @nnx.jit
    def eval_all(mdl):
        answers, slot_logits, halt_logits = mdl(te_tok)
        target = te_sub[:, -1]
        halt_step = halt_logits.argmax(-1)                     # [B], 0-indexed
        halted_pred = jnp.take_along_axis(
            answers.argmax(-1), halt_step[:, None], axis=1)[:, 0]
        full_acc = jnp.mean(answers[:, -1].argmax(-1) == target)
        halted_acc = jnp.mean(halted_pred == target)
        # Pearson correlation between the chosen halt step and the true length.
        hs = halt_step.astype(jnp.float32) + 1.0
        ke = te_keff.astype(jnp.float32)
        hs_c, ke_c = hs - hs.mean(), ke - ke.mean()
        corr = (hs_c * ke_c).mean() / jnp.sqrt((hs_c**2).mean() * (ke_c**2).mean() + 1e-12)
        p = jax.nn.softmax(halt_logits, axis=-1)
        return full_acc, halted_acc, corr, p[:, 0].mean(), hs.mean()

    for i in range(steps):
        key, k = jax.random.split(key)
        step(model, opt, k)

    full_acc, halted_acc, corr, p1, mean_halt = eval_all(model)
    results = {
        "full_acc": float(full_acc), "halted_acc": float(halted_acc),
        "corr": float(corr), "p1_mass": float(p1), "mean_halt": float(mean_halt),
        "params": n_params(model),
    }
    return (results, model) if return_model else results


# ---------------------------------------------------------------------------
# #39: deterministic convergence halting in grade-logit space. The raw-latent
# version of this rule is dead (PR #96, tombstoned); what survived is #96's
# diagnostic — consecutive-slot *grade logits* separate converged from
# computing steps where the latents do not. Everything here is eval-only on a
# trained halt_off model: no halting pressure, no learned halting (#123 killed
# that separately — the failure there is the incentive; a detector has none).

GRADE_TAUS = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def grade_cosines(slot_logits):
    """Cosine between consecutive slots' grade logits, f32. [B, K, m] ->
    [B, K-1]; column j compares slot j+1 against slot j (0-indexed), i.e. the
    transition into slot k = j+2 in 1-indexed terms."""
    x = slot_logits.astype(jnp.float32)
    a, b = x[:, 1:], x[:, :-1]
    return (a * b).sum(-1) / (
        jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1) + 1e-12)


def grade_halt_steps(slot_logits, tau):
    """First-crossing rule (#39): halt at the first slot k >= 2 (1-indexed)
    whose grade-logit cosine against slot k-1 exceeds tau; run all K slots if
    none does. Returns the 0-indexed halt step [B] — writes spent = step + 1
    (the detector reads slot k's grade, so slot k is already written)."""
    cos = grade_cosines(slot_logits)                # [B, K-1]
    crossed = cos > tau
    first = jnp.argmax(crossed, axis=-1)            # first True column, 0 if none
    K = slot_logits.shape[1]
    return jnp.where(crossed.any(-1), first + 1, K - 1)


def converged_transition_labels(k_eff, K):
    """Ground-truth label per cosine column: transition j is *converged* iff
    the state was already final before it, i.e. 1-indexed k = j+2 > k_eff.
    Exact by construction of variable_chain_task — never inferred from
    repeated residues (#96 mislabelled 11.4% of steps that way)."""
    j = jnp.arange(K - 1)
    return j[None, :] >= (k_eff[:, None] - 1)       # [B, K-1] bool


def _pearson(x, y):
    xc, yc = x - x.mean(), y - y.mean()
    return (xc * yc).mean() / jnp.sqrt((xc**2).mean() * (yc**2).mean() + 1e-12)


def grade_gate_stats(slot_logits, k_eff):
    """The pre-registered gate (#39): converged-vs-computing grade-logit
    cosine on one seed. If the two means sit within 1 pooled sigma, the signal
    is dead and the ladder must not run."""
    cos = grade_cosines(slot_logits)
    conv = converged_transition_labels(k_eff, slot_logits.shape[1])
    def masked(mask):
        w = mask.astype(jnp.float32)
        mean = (cos * w).sum() / w.sum()
        var = (((cos - mean) ** 2) * w).sum() / w.sum()
        return float(mean), float(jnp.sqrt(var))
    cm, cs = masked(conv)
    pm, ps = masked(~conv)
    pooled = ((cs**2 + ps**2) / 2) ** 0.5
    return {"converged_mean": cm, "converged_std": cs,
            "computing_mean": pm, "computing_std": ps,
            "pooled_std": pooled, "separated": abs(cm - pm) > pooled}


def grade_halting_ladder(model, te_tok, te_sub, te_keff, taus=GRADE_TAUS):
    """Eval-only tau ladder on a trained model: one forward gives every stop
    point (answers[:, k] is the readout from slots 1..k) and the detector
    signal. Per tau: halted accuracy, mean writes, corr(halt step, k_eff)."""
    answers, slot_logits, _ = model(te_tok)
    target = te_sub[:, -1]
    pred_k = answers.argmax(-1)                     # [B, K]
    rows = []
    for tau in taus:
        hs = grade_halt_steps(slot_logits, tau)     # [B], 0-indexed
        halted_pred = jnp.take_along_axis(pred_k, hs[:, None], axis=1)[:, 0]
        writes = hs.astype(jnp.float32) + 1.0
        rows.append({
            "tau": tau,
            "halted_acc": float(jnp.mean(halted_pred == target)),
            "mean_writes": float(writes.mean()),
            "corr": float(_pearson(writes, te_keff.astype(jnp.float32))),
        })
    return rows


def run_halting_protocol(*, K, m, dim, steps, seeds, taus=GRADE_TAUS):
    """The #39 protocol, verdict computed mechanically against the
    pre-registered bars: train halt_off per seed, gate on the first seed
    (stop if the signal doesn't separate), then the tau ladder, judged on
    3-seed means at one global tau. KEEP needs all of
    (a) halted acc within 2 sigma_pooled of the same seeds' full-depth acc,
    (b) mean writes <= 3.5 (oracle 3.25 at K=4), (c) corr(halt, k_eff) >= 0.8."""
    print(f"== #39 grade-logit halting: K={K} m={m} dim={dim} steps={steps} "
          f"seeds={seeds} ==", flush=True)
    runs = []
    for i, seed in enumerate(seeds):
        t0 = time.time()
        r, model = train_one_halt_arm("halt_off", K=K, m=m, dim=dim,
                                      steps=steps, seed=seed, return_model=True)
        _, _, _, te_tok, te_sub, te_keff = halt_task_data(K, m, seed)
        print(f"halt_off seed={seed} full_acc={r['full_acc']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if i == 0:
            _, slot_logits, _ = model(te_tok)
            g = grade_gate_stats(slot_logits, te_keff)
            print(f"gate: converged {g['converged_mean']:+.3f}±{g['converged_std']:.3f} "
                  f"vs computing {g['computing_mean']:+.3f}±{g['computing_std']:.3f} "
                  f"(pooled σ {g['pooled_std']:.3f}) -> "
                  f"{'separated' if g['separated'] else 'OVERLAP — STOP'}", flush=True)
            if not g["separated"]:
                print("verdict: KILL at the gate — grade-logit cosine does not "
                      "separate converged from computing on this model; the "
                      "pre-registered protocol says do not run the sweep.", flush=True)
                return {"gate": g, "verdict": "kill-at-gate"}
        runs.append((r, grade_halting_ladder(model, te_tok, te_sub, te_keff, taus)))

    import statistics
    mu = statistics.mean

    def sd(v):
        return statistics.stdev(v) if len(v) > 1 else 0.0

    full = [r["full_acc"] for r, _ in runs]
    full_mu, full_sd = mu(full), sd(full)
    print(f"\nfull-depth ceiling: {full_mu:.4f} ± {full_sd:.4f}  "
          f"(oracle writes {sum(min(k + 2, K) for k in range(K)) / K:.2f}, fixed {K})")
    print(f"{'tau':>5} {'halted_acc':>16} {'writes':>13} {'corr':>13}  verdict")
    verdict, keep_tau = "kill", None
    for ti, tau in enumerate(taus):
        acc = [lad[ti]["halted_acc"] for _, lad in runs]
        wr = [lad[ti]["mean_writes"] for _, lad in runs]
        co = [lad[ti]["corr"] for _, lad in runs]
        pooled = ((sd(acc)**2 + full_sd**2) / 2) ** 0.5
        a = abs(mu(acc) - full_mu) <= 2 * pooled
        b = mu(wr) <= 3.5
        c = mu(co) >= 0.8
        marks = f"a={'✓' if a else '✗'} b={'✓' if b else '✗'} c={'✓' if c else '✗'}"
        if a and b and c and keep_tau is None:
            verdict, keep_tau = "keep", tau
        print(f"{tau:>5.2f} {mu(acc):>7.4f}±{sd(acc):.4f} {mu(wr):>7.3f}±{sd(wr):.3f} "
              f"{mu(co):>+7.3f}±{sd(co):.3f}  {marks}", flush=True)
    print(f"\nverdict: {verdict.upper()}"
          + (f" at tau={keep_tau}" if keep_tau is not None else "")
          + "  (bars: (a) within 2σ_pooled of full-depth, (b) writes ≤ 3.5, "
            "(c) corr ≥ 0.8 — all three at one global tau)", flush=True)
    return {"verdict": verdict, "keep_tau": keep_tau, "runs": runs,
            "full_mu": full_mu, "full_sd": full_sd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="serial,parallel,depthonly",
                    help="any of serial,parallel,depthonly,slotsonly (#62),"
                         "finalonly (#67),annealed (#73),densedepth,"
                         "densedepth_tied (#79),overwrite,budget1,budget2,"
                         "unlimited (#63),budget1_local,budget2_local,"
                         "unlimited_local (#116),halt_traj,halt_state,"
                         "halt_off (#123); annealed takes an"
                         " optional onset and floor (#95), e.g. annealed@0.2"
                         " or annealed@0.4f0.1")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--K", type=int, default=4, help="number of chained sub-steps = number of slots")
    ap.add_argument("--m", type=int, default=7, help="modulus (prime); vocab and chance level 1/m")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--halting", action="store_true",
                    help="run the #39 grade-logit convergence-halting protocol "
                         "(trains halt_off per seed, gates on seed one, then the "
                         "pre-registered tau ladder) instead of --arms")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    if args.halting:
        run_halting_protocol(K=args.K, m=args.m, dim=args.dim,
                             steps=args.steps, seeds=seeds)
        return

    print(f"== serial-scratchpad proof (#38): K={args.K} m={args.m} dim={args.dim} "
          f"steps={args.steps} (chance={1/args.m:.3f}) ==")
    print(f"{'arm':>16} {'seed':>5} {'params':>9} {'cut':>5} {'acc@cut':>8} {'final_acc':>10}"
          f" {'sec':>7}  slot_accs cut[...] end[...]")
    def fmt(accs):
        return " ".join(f"{a:.3f}" for a in accs) if any(accs) else "-"

    for spec in args.arms.split(","):
        arm, anneal_kw = parse_arm_spec(spec)
        for seed in seeds:
            t0 = time.time()
            if arm in HALT_ARMS:
                r = train_one_halt_arm(arm, K=args.K, m=args.m,
                                       dim=args.dim, steps=args.steps, seed=seed)
                print(f"{spec:>16} {seed:>5} {r['params']/1e6:>8.2f}M "
                      f"full={r['full_acc']:.4f} halted={r['halted_acc']:.4f} "
                      f"corr={r['corr']:+.3f} p1={r['p1_mass']:.3f} "
                      f"mean_halt={r['mean_halt']:.2f} {time.time()-t0:>7.1f}s", flush=True)
                continue
            r = train_one_arm(arm, K=args.K, m=args.m,
                              dim=args.dim, steps=args.steps, seed=seed, **anneal_kw)
            print(f"{spec:>16} {seed:>5} {r['params']/1e6:>8.2f}M {r['cut_step']:>5} "
                  f"{r['cut_final_acc']:>8.4f} {r['final_acc']:>10.4f} {time.time()-t0:>7.1f}  "
                  f"cut[{fmt(r['cut_slot_acc'])}] end[{fmt(r['slot_acc'])}]",
                  flush=True)


if __name__ == "__main__":
    main()
