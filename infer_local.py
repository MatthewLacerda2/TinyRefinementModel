import os
import jax
import jax.numpy as jnp
from flax import nnx
import tiktoken
import orbax.checkpoint as ocp
import time

from train_local import (
    UniversalReasoner,
    LATENT_DIM,
    MAX_SEQ_LEN,
    PAD_TOKEN_ID,
    MAX_STEPS_LIMIT,
    HUNCH_REFRESH_EVERY
)
from inference_core import run_model_inference

CHECKPOINT_DIR = os.path.abspath("orbax_checkpoints")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def analyze_first_token_contenders(logits, enc, expected_token=None, top_k_count=20):
    # Analysis of contenders for the first token
    probs = jax.nn.softmax(logits)
    top_indices = jnp.argsort(logits)[-top_k_count:][::-1]
    
    print("\n" + "─" * 30)
    print("🎯 Expected First Token Contenders:")
    found_expected = False
    for rank, idx in enumerate(top_indices):
        token_val = int(idx)
        token_str = enc.decode([token_val]).replace("\n", "\\n")
        prob = float(probs[idx])
        marker = "⭐" if expected_token and expected_token in token_str else "  "
        if marker == "⭐": found_expected = True
        print(f"{marker} {rank+1}. Rank: {token_val:6} | Prob: {prob:6.2%} | Token: [{token_str}]")
    
    if expected_token and not found_expected:
        # Find the expected token in the full vocabulary if not in top K
        # This is a bit slow as it scans the whole vocab logits
        all_ranks = jnp.argsort(logits)[::-1]
        expected_ids = enc.encode(expected_token)
        if expected_ids:
            target_id = expected_ids[0]
            # Find where this ID is in all_ranks
            actual_rank = int(jnp.where(all_ranks == target_id)[0][0]) + 1
            actual_prob = float(probs[target_id])
            actual_str = enc.decode([target_id]).replace("\n", "\\n")
            print(f"--- Expected Token Not in Top {top_k_count} ---")
            print(f"⭐ {actual_rank}. Rank: {target_id:6} | Prob: {actual_prob:6.4%} | Token: [{actual_str}]")

    print("─" * 30)
    print("🤖 Assistant: ", end="", flush=True)

def generate_text(model, enc, prompt, max_new_tokens=128, temperature=0.8, expected_token=None):
    seed = int(time.time() * 1000) % (2**31)
    rng = jax.random.PRNGKey(seed)

    tokens_list = enc.encode(prompt)

    @nnx.jit(static_argnames=['refresh'])
    def get_next_logits(m, padded_tks, valid_len, refresh):
        logits = run_model_inference(m, padded_tks, max_steps=MAX_STEPS_LIMIT, should_refresh=refresh)
        return logits[0, valid_len - 1, :]

    print("🤖 Assistant: ", end="", flush=True)

    for i in range(max_new_tokens):
        valid_len = len(tokens_list)

        if valid_len >= MAX_SEQ_LEN:
            break

        padded_array = tokens_list + [PAD_TOKEN_ID] * (MAX_SEQ_LEN - valid_len)
        input_ids = jnp.array([padded_array], dtype=jnp.int32)

        should_refresh = (i % HUNCH_REFRESH_EVERY == 0)

        logits = get_next_logits(model, input_ids, valid_len, should_refresh)

        if i == 0:
            analyze_first_token_contenders(logits, enc, expected_token)

        rng, subkey = jax.random.split(rng)

        scaled_logits = logits / temperature

        next_token = int(jax.random.categorical(subkey, scaled_logits))

        if next_token == PAD_TOKEN_ID:
            break

        tokens_list.append(next_token)
        print(enc.decode([next_token]), end="", flush=True)

    print()
    return tokens_list

def run_inference():
    print(f"🔮 Initializing TinyRefinementModel (Dim={LATENT_DIM})...")

    enc = tiktoken.get_encoding("cl100k_base")

    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(0))

    mngr = ocp.CheckpointManager(
        CHECKPOINT_DIR,
        item_names=('model', 'optimizer', 'monitor_state', 'step'),
    )

    latest_step = mngr.latest_step()
    if latest_step is None:
        print(f"❌ Error: No checkpoints found in {CHECKPOINT_DIR}")
        print("Please train the model first using start_training.py")
        return

    print(f"🔄 Loading weights from step {latest_step}...")

    restored = mngr.restore(latest_step, args=ocp.args.Composite(
        model=ocp.args.StandardRestore(nnx.state(model)),
    ))
    nnx.update(model, restored['model'])

    print("✅ Model loaded and ready!")

    print("\n" + "="*50)
    print("TinyRefinementModel CLI (Orbax-Linked)")
    print("Type your prompt and press Enter (/exit to quit)")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("👤 User: ").strip()

            if user_input.lower() == "/exit":
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            expected_input = input("❓ Expected first token (optional): ").strip()
            expected_token = expected_input if expected_input else None

            generate_text(model, enc, user_input, expected_token=expected_token)
            print("-" * 30)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    run_inference()