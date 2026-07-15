"""Toy proof harness for the supervised serial latent scratchpad (#38, #62, #67, #73, #79).

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

Task: r_0 = 0; r_k = (r_{k-1} * a_k + b_k) mod m from tokens [a_1 b_1 ... a_K b_K].
Affine composition mod m is non-commutative — no order-free shortcut — and
decomposes exactly into the K sub-results the slots are graded on.

`--halting` (#39) skips the arms table and instead trains the serial arm, then
runs the deterministic cosine-halting ladder (docs/design/cosine-halting.md):
stop writing slots when cosine(s_k, s_{k-1}) exceeds tau, swept over thresholds
on the uniform and identity-tail eval splits.

    python scratchpad_harness.py --arms serial,parallel,depthonly --seeds 0,1,2
    python scratchpad_harness.py --halting --seeds 0,1,2
"""

import argparse
import re
import time

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from plan_a_model import Block, CausalRefiner


def _chain(a, b, m):
    """tokens [B, 2K] and sub-targets [B, K] for r_k = (r_{k-1} * a_k + b_k) mod m."""
    tokens = jnp.stack([a, b], axis=-1).reshape(a.shape[0], 2 * a.shape[1])

    def step(r, ab):                                    # ab: [B, 2] = (a_k, b_k)
        r_new = (r * ab[:, 0] + ab[:, 1]) % m
        return r_new, r_new
    _, subs = jax.lax.scan(step, jnp.zeros(a.shape[0], jnp.int32),
                           jnp.stack([a.T, b.T], axis=-1).astype(jnp.int32))
    return tokens.astype(jnp.int32), subs.T


def affine_chain_task(K, m):
    """tokens [B, 2K] = a_1 b_1 ... a_K b_K (a nonzero); sub-targets r_1..r_K
    with r_k = (r_{k-1} * a_k + b_k) mod m; the final answer is r_K."""
    def fn(key, batch):
        ka, kb = jax.random.split(key)
        a = jax.random.randint(ka, (batch, K), 1, m)   # nonzero -> invertible maps
        b = jax.random.randint(kb, (batch, K), 0, m)
        return _chain(a, b, m)
    return fn


def affine_chain_varlen_task(K, m):
    """Identity-tail variant (#39): effective length L ~ U{1..K}; steps after L
    are the identity map (a=1, b=0), so r_k = r_L for all k > L. Both values
    are legal tokens of the base task — only the sampling differs — which gives
    a convergence halter its payoff case without leaving the training alphabet."""
    def fn(key, batch):
        ka, kb, kl = jax.random.split(key, 3)
        a = jax.random.randint(ka, (batch, K), 1, m)
        b = jax.random.randint(kb, (batch, K), 0, m)
        live = jnp.arange(K)[None, :] < jax.random.randint(kl, (batch,), 1, K + 1)[:, None]
        return _chain(jnp.where(live, a, 1), jnp.where(live, b, 0), m)
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

    def encode(self, tokens):
        h = self.embed(tokens)
        for blk in self.encoder:
            h = blk(h)
        return h

    def _queries(self, bsz):
        q = self.slot_index(jnp.arange(self.num_slots))[None, :, :]         # [1, K, d]
        return jnp.broadcast_to(q, (bsz, self.num_slots, q.shape[-1]))

    def write_slots(self, h, upto=None):
        """Serial write loop: slots 1..upto, each written once, slot k reading
        tokens + slots < k. Because writes are causal and frozen, the first n
        slots of a full run are bit-identical to a run stopped after n writes —
        the property cosine halting (#39) stands on, guarded in tests."""
        queries = self._queries(h.shape[0])
        n = self.num_slots if upto is None else upto
        slots = []
        for k in range(n):
            ctx = jnp.concatenate([h] + slots, axis=1) if slots else h
            slots.append(self.write_block(queries[:, k:k + 1], ctx))        # written once
        return jnp.concatenate(slots, axis=1)                               # [B, n, d]

    def __call__(self, tokens):
        h = self.encode(tokens)
        if self.serial:
            slots = self.write_slots(h)
        else:
            slots = self.write_block(self._queries(tokens.shape[0]), h)     # one step, tokens only

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
    ce_final = optax.softmax_cross_entropy_with_integer_labels(answer_logits, final).mean()
    ce_slots = optax.softmax_cross_entropy_with_integer_labels(slot_logits, sub).mean()
    return ce_final + lam * ce_slots, (answer_logits, slot_logits)


def train_one_arm(arm, *, K=4, m=7, dim=64, heads=4, enc=2, steps=2500, batch=256,
                  lr=2e-3, wd=0.01, seed=0, n_pool=32768, n_test=4096,
                  anneal_onset=0.4, anneal_decay=0.2, anneal_floor=0.0):
    assert arm in ("serial", "parallel", "depthonly", "slotsonly", "finalonly",
                   "annealed", "densedepth", "densedepth_tied"), f"unknown arm {arm!r}"
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
        final_acc = jnp.mean(answer_logits.argmax(-1) == te_sub[:, -1])
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
        "model": model,      # the halting ladder (#39) evaluates the trained net
    }


# --- convergence (cosine) halting, #39 -------------------------------------
# Deterministic, inference-only: after writing slot k (k >= 2), halt when
# cosine(s_k, s_{k-1}) on the raw slot latents exceeds tau. Design + the
# pre-registered criteria: docs/design/cosine-halting.md.

HALT_THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)


def slot_cosines(slots):
    """cos(s_{k+1}, s_k) in f32 -> [B, K-1]; column j is the signal available
    right after writing slot j+2 (1-indexed)."""
    s = slots.astype(jnp.float32)
    s = s / (jnp.linalg.norm(s, axis=-1, keepdims=True) + 1e-8)
    return jnp.sum(s[:, 1:] * s[:, :-1], axis=-1)


def halt_indices(cos, tau, K):
    """Writes used per example: first k >= 2 whose signal cos[:, k-2] > tau,
    else all K."""
    fired = cos > tau
    first = jnp.argmax(fired, axis=1)
    return jnp.where(fired.any(axis=1), first + 2, K)


def eval_halting(model, tokens, subs, thresholds=HALT_THRESHOLDS):
    """One forward serves every threshold: slots are write-once and causal, so
    the readout on the first n slots IS the halted computation (guarded by the
    truncation-equivalence test). Returns fixed-K accuracy, per-tau
    (accuracy, mean writes), and the mechanism readout — cosine at converged
    steps (r_k = r_{k-1}) vs computing steps."""
    K = model.num_slots
    h = model.encode(tokens)
    slots = model.write_slots(h)
    cos = slot_cosines(slots)
    # answer prediction for every stop point n in {2..K}, all examples at once
    preds = jnp.stack([model.readout(h, slots[:, :n]).argmax(-1)
                       for n in range(2, K + 1)], axis=1)          # [B, K-1]
    final = subs[:, -1]
    fixed_acc = float(jnp.mean(preds[:, -1] == final))

    per_tau = []
    for tau in thresholds:
        n = halt_indices(cos, tau, K)
        pred = jnp.take_along_axis(preds, (n - 2)[:, None], axis=1)[:, 0]
        per_tau.append((tau, float(jnp.mean(pred == final)), float(jnp.mean(n))))

    converged = subs[:, 1:] == subs[:, :-1]                        # aligns with cos columns
    def _stats(vals, mask):
        vals, msk = vals.ravel(), mask.ravel()
        cnt = jnp.sum(msk)
        mean = jnp.sum(jnp.where(msk, vals, 0)) / jnp.maximum(cnt, 1)
        var = jnp.sum(jnp.where(msk, (vals - mean) ** 2, 0)) / jnp.maximum(cnt, 1)
        return float(mean), float(jnp.sqrt(var)), int(cnt)
    # diagnostic-only secondary (pre-registered, no claim): the same signal on
    # the slot grade logits — semantic commitment, immune to whatever the
    # slot-index embedding contributes to the raw latents
    lcos = slot_cosines(model.slot_readout(slots))
    mechanism = {"converged": _stats(cos, converged), "computing": _stats(cos, ~converged),
                 "logit_converged": _stats(lcos, converged),
                 "logit_computing": _stats(lcos, ~converged)}
    return fixed_acc, per_tau, mechanism


def run_halting(seeds, *, K=4, m=7, dim=64, steps=2500, n_test=4096):
    """Train the #38 serial arm per seed (unchanged), then run the halting
    ladder on the uniform and identity-tail splits."""
    results = {"uniform": [], "varlen": []}
    for seed in seeds:
        t0 = time.time()
        r = train_one_arm("serial", K=K, m=m, dim=dim, steps=steps, seed=seed)
        model = r["model"]
        print(f"-- seed {seed}: trained serial arm ({r['params']/1e6:.2f}M params, "
              f"{time.time()-t0:.0f}s), uniform fixed-K acc {r['final_acc']:.4f} "
              f"(#38 anchor), slots [{' '.join(f'{a:.3f}' for a in r['slot_acc'])}]", flush=True)
        # same held-out derivation as train_one_arm (dk_te is the third split)
        key = jax.random.PRNGKey(seed)
        _, _, dk_te = jax.random.split(key, 3)
        for split, task in (("uniform", affine_chain_task(K, m)),
                            ("varlen", affine_chain_varlen_task(K, m))):
            te_tok, te_sub = task(dk_te, n_test)
            fixed_acc, per_tau, mech = eval_halting(model, te_tok, te_sub)
            results[split].append((seed, fixed_acc, per_tau, mech))
            c, s = mech["converged"], mech["computing"]
            lc, ls = mech["logit_converged"], mech["logit_computing"]
            print(f"   {split:>8}: fixed-K acc {fixed_acc:.4f} | cosine at "
                  f"converged steps {c[0]:.4f}±{c[1]:.4f} (n={c[2]}) vs "
                  f"computing {s[0]:.4f}±{s[1]:.4f} (n={s[2]}) | grade-logit "
                  f"cosine {lc[0]:.4f}±{lc[1]:.4f} vs {ls[0]:.4f}±{ls[1]:.4f}", flush=True)
            for tau, acc, writes in per_tau:
                print(f"     tau={tau:<6} acc {acc:.4f}  mean writes {writes:.3f}", flush=True)

    print("\n== 3-seed means (verdict table; criteria in docs/design/cosine-halting.md) ==")
    for split in ("uniform", "varlen"):
        rows = results[split]
        fixed = [r[1] for r in rows]
        print(f"[{split}] fixed-K acc mean {sum(fixed)/len(fixed):.4f} "
              f"(seeds: {' '.join(f'{a:.4f}' for a in fixed)})")
        for i, tau in enumerate(HALT_THRESHOLDS):
            accs = [r[2][i][1] for r in rows]
            wrts = [r[2][i][2] for r in rows]
            print(f"  tau={tau:<6} acc mean {sum(accs)/len(accs):.4f} "
                  f"(seeds: {' '.join(f'{a:.4f}' for a in accs)})  "
                  f"writes mean {sum(wrts)/len(wrts):.3f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="serial,parallel,depthonly",
                    help="any of serial,parallel,depthonly,slotsonly (#62),"
                         "finalonly (#67),annealed (#73),densedepth,"
                         "densedepth_tied (#79); annealed takes an"
                         " optional onset and floor (#95), e.g. annealed@0.2"
                         " or annealed@0.4f0.1")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--K", type=int, default=4, help="number of chained sub-steps = number of slots")
    ap.add_argument("--m", type=int, default=7, help="modulus (prime); vocab and chance level 1/m")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--halting", action="store_true",
                    help="#39: train the serial arm, then run the cosine-halting "
                         "threshold ladder on the uniform + identity-tail splits")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    if args.halting:
        print(f"== cosine halting (#39): K={args.K} m={args.m} dim={args.dim} "
              f"steps={args.steps} thresholds={HALT_THRESHOLDS} ==")
        run_halting(seeds, K=args.K, m=args.m, dim=args.dim, steps=args.steps)
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
            r = train_one_arm(arm, K=args.K, m=args.m,
                              dim=args.dim, steps=args.steps, seed=seed, **anneal_kw)
            print(f"{spec:>16} {seed:>5} {r['params']/1e6:>8.2f}M {r['cut_step']:>5} "
                  f"{r['cut_final_acc']:>8.4f} {r['final_acc']:>10.4f} {time.time()-t0:>7.1f}  "
                  f"cut[{fmt(r['cut_slot_acc'])}] end[{fmt(r['slot_acc'])}]",
                  flush=True)


if __name__ == "__main__":
    main()
