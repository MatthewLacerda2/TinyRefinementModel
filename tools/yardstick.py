"""The GPT-2-small yardstick, as a library (#48).

The base-model bar (CLAUDE.md) asks one question of a finished base run: does it
match GPT-2-small on a standard external metric? This module is the measuring
instrument — LAMBADA last-word prediction, scored two ways from one pass:

  - **accuracy**: greedy (argmax) prediction of every token of the final word,
    teacher-forced. All tokens must match — equivalent to greedy generation
    producing exactly the target word.
  - **perplexity**: exp(-mean over examples of the target word's total log-prob).
    One word per example, so the mean is per-word — the lm-eval-harness
    "perplexity" definition for lambada_openai.

Protocol notes, so the number is comparable to something:
  - Dataset: OpenAI's processed LAMBADA test set (lambada_test.jsonl, 5153
    examples) — the variant behind every "lambada_openai" number, sha256-pinned.
  - Split: context = text up to the last space; target = " " + last word,
    encoded separately (the lm-eval convention; BPE joins may differ from
    encoding the full text, identically for every model measured this way).
  - No stop-word filter, no candidate restriction. The GPT-2 *paper* numbers
    (45.99% acc / 35.13 ppl for the 124M model) use OpenAI's detokenizers and a
    stop-word prediction filter — NOT this protocol; don't compare against them.
    The like-for-like reference is GPT-2-small measured by this very code
    (tools/calibrate_yardstick_gpt2.py) — see GPT2_SMALL_REFERENCE below.

Everything here is model-agnostic: scoring talks to the model only through
`logits_fn(tokens) -> logits` (numpy in, numpy out, causal), so the same core
scores our checkpoints (tools/eval_yardstick.py), the calibration GPT-2, and the
tiny fakes in tests/test_yardstick.py.
"""

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass

import numpy as np

LAMBADA_URL = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"
LAMBADA_SHA256 = "4aa8d02cd17c719165fc8a7887fddd641f43fcafa4b1c806ca8abc31fabdb226"
# Cache under runs/data with the rest of the (regenerable, gitignored) data tier.
LAMBADA_CACHE = "runs/data/eval/lambada_test.jsonl"

# The like-for-like bar: GPT-2-small (124M) measured by THIS instrument
# (tools/calibrate_yardstick_gpt2.py on the full 5153-example set). External
# cross-check: lm-eval-harness reports acc 0.3256 / ppl 40.06 for gpt2 on
# lambada_openai under the same greedy, unfiltered protocol.
GPT2_SMALL_REFERENCE = {
    "lambada_acc": 0.3256,
    "lambada_ppl": 40.06,
    "source": "lm-eval-harness lambada_openai, gpt2 (124M); pending in-repo calibration",
}


def fetch_lambada(path=LAMBADA_CACHE):
    """Download-and-cache the pinned LAMBADA test set; verify sha256 either way."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"⬇️  Downloading LAMBADA test set -> {path}")
        urllib.request.urlretrieve(LAMBADA_URL, path)
    verify_sha256(path, LAMBADA_SHA256)
    return path


def verify_sha256(path, expected):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"{path}: sha256 mismatch (got {actual}, expected {expected}) — "
            "delete the file and re-fetch; a silently different dataset would "
            "make the yardstick lie."
        )


def load_examples(path):
    """One text per line of the jsonl; order preserved (it's part of the recipe)."""
    with open(path) as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


def split_last_word(text):
    """Context/target split at the last space. Returns (context, ' ' + word)."""
    context, _, word = text.rstrip().rpartition(" ")
    return context, " " + word


def encode_example(enc, text, max_seq_len):
    """(context_ids, target_ids) for one example, context left-truncated to fit.

    Returns None for degenerate examples (no space to split on, or an empty
    context after encoding) — the caller counts and reports skips.
    """
    context, target = split_last_word(text)
    if not context:
        return None
    ctx_ids = enc.encode(context)
    tgt_ids = enc.encode(target)
    if not ctx_ids or not tgt_ids or len(tgt_ids) >= max_seq_len:
        return None
    # Keep the most recent context; the model predicts from at least one token.
    ctx_ids = ctx_ids[-(max_seq_len - len(tgt_ids)):]
    return ctx_ids, tgt_ids


@dataclass
class ExampleScore:
    greedy_hit: bool      # every target token was the argmax (teacher-forced)
    logprob: float        # total log-prob of the target word's tokens
    num_target_tokens: int


def _log_softmax(rows):
    """Numerically stable log-softmax over the last axis of a small f64 array."""
    rows = rows.astype(np.float64)
    shifted = rows - rows.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def score_examples(logits_fn, encoded, pad_token_id, batch_size=4,
                   buckets=(64, 128, 256, 512), progress=None):
    """Score encoded (ctx_ids, tgt_ids) pairs through a causal logits_fn.

    Examples are grouped into the smallest fitting bucket length and right-padded
    with pad_token_id — with causal attention, padding after an example cannot
    change the logits we read, so bucketing is purely a compile/memory knob.
    Returns ExampleScores in the input order.
    """
    order = sorted(range(len(encoded)), key=lambda i: sum(map(len, encoded[i])))
    scores = [None] * len(encoded)
    done = 0
    for bucket in buckets:
        members = [i for i in order if scores[i] is None
                   and sum(map(len, encoded[i])) <= bucket]
        for start in range(0, len(members), batch_size):
            batch_idx = members[start:start + batch_size]
            tokens = np.full((len(batch_idx), bucket), pad_token_id, dtype=np.int32)
            for row, i in enumerate(batch_idx):
                ctx, tgt = encoded[i]
                tokens[row, :len(ctx) + len(tgt)] = ctx + tgt
            logits = logits_fn(tokens)
            for row, i in enumerate(batch_idx):
                ctx, tgt = encoded[i]
                # logits[p] predicts tokens[p+1]: the target tokens sit at
                # positions len(ctx)..len(ctx)+len(tgt)-1, predicted one earlier.
                pred_slice = np.asarray(logits[row, len(ctx) - 1: len(ctx) + len(tgt) - 1])
                logprobs = _log_softmax(pred_slice)
                tgt_arr = np.asarray(tgt)
                scores[i] = ExampleScore(
                    greedy_hit=bool((pred_slice.argmax(axis=-1) == tgt_arr).all()),
                    logprob=float(logprobs[np.arange(len(tgt)), tgt_arr].sum()),
                    num_target_tokens=len(tgt),
                )
            done += len(batch_idx)
            if progress:
                progress(done, len(encoded))
    unscored = [i for i, s in enumerate(scores) if s is None]
    if unscored:
        raise ValueError(
            f"{len(unscored)} examples exceed the largest bucket ({buckets[-1]}) — "
            "encode_example should have truncated them; this is a bug."
        )
    return scores


def summarize(scores):
    """The two yardstick numbers plus bookkeeping for the model-card row."""
    logprobs = np.array([s.logprob for s in scores])
    return {
        "lambada_acc": float(np.mean([s.greedy_hit for s in scores])),
        # One target word per example -> per-word perplexity (lm-eval definition).
        "lambada_ppl": float(np.exp(-logprobs.mean())),
        "num_examples": len(scores),
        "mean_target_tokens": float(np.mean([s.num_target_tokens for s in scores])),
    }
