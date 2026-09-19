"""Score any Hugging Face causal LM on the yardstick, with its own tokenizer (#387).

    python -m instruments.yardstick.score_hf --model runs/external/SmolLM2-135M [--limit N]

The same LAMBADA protocol and scoring core as everything else here
(instruments/yardstick/yardstick.py). The one difference from `calibrate_gpt2` is
the tokenizer: GPT-2 shares our r50k_base, a modern model does not, so each example
is split into context and " " + last word as text first and each half is encoded
with the model's own tokenizer. Both numbers stay comparable across tokenizers:
accuracy asks whether the greedy tokens spell exactly the word, and perplexity is
per WORD (one target word per example), so it is the probability of the same string
whatever it is cut into. Per-token CE would not be, and is not reported.

Needs torch + transformers, which are not project requirements: run it from a venv
that has them, with the repo on PYTHONPATH. yardstick.py itself imports no jax.
"""

import argparse

import numpy as np

from instruments.yardstick.yardstick import (
    encode_example,
    fetch_lambada,
    load_examples,
    score_examples,
    summarize,
)

# What each headline number is, and how it was obtained (#175): measured | sampled | estimated | cumulative.
REPORTS = {
    "LAMBADA acc, ppl (any HF model)": ("measured", "the full LAMBADA test set; with --limit it is a subsample"),
}

# Off-config defaults, on purpose (tests/apparatus/test_instrument_defaults.py).
CONFIG_DIVERGENCES = {"--batch": "examples per eval forward of an HF model, not our training micro-batch"}

# LAMBADA passages fit in 1,024 tokens under any of these tokenizers; it is also
# GPT-2's window, so every model is read over the same context budget.
CONTEXT = 1024


class HFTokenizer:
    """The `enc.encode(text)` interface encode_example expects, over an HF tokenizer:
    no BOS/EOS added, so " word" encodes exactly as it would mid-text."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="an HF model id or a local directory")
    ap.add_argument("--limit", type=int, default=None, help="first N examples only (smoke)")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # f32 on purpose: this is a reference number, not a speed benchmark.
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device).eval()
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def logits_fn(tokens):
        with torch.no_grad():
            # Right-padded and causal: no position we read can see the pad.
            out = model(torch.from_numpy(np.asarray(tokens)).long().to(device))
        return out.logits.float().cpu().numpy()

    texts = load_examples(fetch_lambada())
    if args.limit:
        texts = texts[:args.limit]
    enc = HFTokenizer(tokenizer)
    encoded = [pair for text in texts if (pair := encode_example(enc, text, CONTEXT))]
    print(f"📏 {args.model}: scoring {len(encoded)} of {len(texts)} LAMBADA examples on {device}…")
    scores = score_examples(logits_fn, encoded, pad, batch_size=args.batch,
                            buckets=(64, 128, 256, 512, CONTEXT))
    result = summarize(scores)
    print(f"LAMBADA acc {result['lambada_acc']:.4f} | per-word ppl {result['lambada_ppl']:.2f} "
          f"| examples {result['num_examples']} | mean target tokens {result['mean_target_tokens']:.2f}")
    if args.limit:
        print(f"⚠️ --limit {args.limit}: subsampled — a reference needs the full set.")


if __name__ == "__main__":
    main()
