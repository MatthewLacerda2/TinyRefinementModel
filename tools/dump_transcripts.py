"""Fixed-prompt transcripts per checkpoint — the systematized vibes eval.

Same prompts, same sampling settings, every time: quality becomes comparable
across checkpoints instead of living in scrollback. Writes a markdown file
under the run folder (runs/<run>/transcripts/step_<n>.md).

Run against the latest (or a given) checkpoint when the GPU is free:
    PYTHONPATH=. python tools/dump_transcripts.py [--max-new-tokens 128]
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import argparse
import datetime

import tiktoken
from dotenv import load_dotenv

load_dotenv()

from checkpoint_utils import discover_latest_checkpoint_run
from config import TOKENIZER_NAME
from tools.common import restore_model

PROMPTS = [
    "The water cycle begins when",
    "To compute the area of a rectangle, you",
    "def fibonacci(n):",
    "The main difference between a star and a planet is",
    "Once upon a time, in a small village,",
    "2 + 2 = 4. 3 + 5 =",
]


def main():
    parser = argparse.ArgumentParser(description="fixed-prompt transcript dump")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--prompts", type=int, default=len(PROMPTS), help="use only the first N prompts")
    args = parser.parse_args()

    run_dir = None
    if args.checkpoint_path is None:
        checkpoint_path, run_id = discover_latest_checkpoint_run()
        if checkpoint_path is not None:
            run_dir = os.path.join("runs", run_id)

    model, ckpt_step = restore_model(args.checkpoint_path)
    enc = tiktoken.get_encoding(TOKENIZER_NAME)

    from infer_local import generate_text

    out_dir = os.path.join(run_dir or ".", "transcripts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"step_{ckpt_step}.md")

    lines = [
        f"# Transcripts — checkpoint step {ckpt_step}",
        f"Generated {datetime.datetime.now().astimezone().isoformat()} | "
        f"temperature {args.temperature} | max new tokens {args.max_new_tokens}",
        "",
    ]
    for prompt in PROMPTS[:args.prompts]:
        print(f"\n👤 Prompt: {prompt}")
        tokens = generate_text(
            model, enc, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        completion = enc.decode(tokens[len(enc.encode(prompt)):])
        lines += [f"## {prompt}", "", completion.strip() or "(empty)", ""]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n✨ Saved {out_path}")


if __name__ == "__main__":
    main()
