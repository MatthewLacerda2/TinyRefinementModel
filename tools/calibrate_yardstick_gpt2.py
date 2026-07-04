"""Calibrate the yardstick against the model it measures against (#48).

Runs the exact scoring core from tools/yardstick.py over the real GPT-2-small
(124M, HF `gpt2`) on the full LAMBADA test set. If the instrument is honest, it
reproduces the known lm-eval-harness reading (acc ≈ 0.3256, ppl ≈ 40.06 —
same greedy, unfiltered protocol). A mismatch means the bug is in OUR eval,
and every future "did v1 hit the bar" verdict would inherit it — this script is
the reference-numerics test for the yardstick itself.

Needs torch + transformers, which are NOT project requirements (the RTX 2060 box
never runs GPT-2). Install into a scratch venv when re-calibrating:

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install transformers
    PYTHONPATH=. python tools/calibrate_yardstick_gpt2.py [--limit 500]

Tokenizer note: our r50k_base IS the GPT-2 BPE — tiktoken and HF `gpt2` produce
identical ids, so the same encoded examples feed both models unchanged.
"""

import argparse

import numpy as np

try:
    import torch
    from transformers import GPT2LMHeadModel
except ImportError:
    raise SystemExit("calibration needs torch + transformers — see the module docstring.")

import tiktoken

from config import TOKENIZER_NAME
from tools.yardstick import (
    GPT2_SMALL_REFERENCE,
    encode_example,
    fetch_lambada,
    load_examples,
    score_examples,
    summarize,
)

GPT2_CONTEXT = 1024
GPT2_EOT = 50256  # end-of-text doubles as pad, exactly as in our own runs


def make_gpt2_logits_fn(model):
    def logits_fn(tokens):
        with torch.no_grad():
            # Right-padding needs no attention mask for what we read: attention is
            # causal, so positions before the pad cannot see it.
            out = model(torch.from_numpy(np.asarray(tokens)).long())
        return out.logits.float().numpy()
    return logits_fn


def main():
    ap = argparse.ArgumentParser(description="yardstick calibration on HF gpt2 (124M)")
    ap.add_argument("--limit", type=int, default=None, help="first N examples only (smoke)")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    texts = load_examples(fetch_lambada())
    if args.limit:
        texts = texts[:args.limit]
    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    encoded = [pair for text in texts if (pair := encode_example(enc, text, GPT2_CONTEXT))]

    print(f"📏 Calibrating on {len(encoded)} LAMBADA examples (batch {args.batch})…")
    scores = score_examples(
        make_gpt2_logits_fn(model), encoded, GPT2_EOT, batch_size=args.batch,
        buckets=(64, 128, 256, 512, GPT2_CONTEXT),
        progress=lambda done, total: print(f"  … {done}/{total}", flush=True) if done % 512 < args.batch else None,
    )
    result = summarize(scores)

    ref_acc, ref_ppl = GPT2_SMALL_REFERENCE["lambada_acc"], GPT2_SMALL_REFERENCE["lambada_ppl"]
    print()
    print(f"{'metric':<22} {'this instrument':>16} {'reference':>10} {'delta':>8}")
    print(f"{'LAMBADA acc':<22} {result['lambada_acc']:>16.4f} {ref_acc:>10.4f} {result['lambada_acc'] - ref_acc:>+8.4f}")
    print(f"{'LAMBADA ppl':<22} {result['lambada_ppl']:>16.2f} {ref_ppl:>10.2f} {result['lambada_ppl'] - ref_ppl:>+8.2f}")
    print(f"(examples: {result['num_examples']}, mean target tokens: {result['mean_target_tokens']:.2f})")
    if args.limit:
        print(f"⚠️ --limit {args.limit}: subsampled — calibration verdicts need the full set.")


if __name__ == "__main__":
    main()
