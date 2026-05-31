import os
import jax
import jax.numpy as jnp
from flax import nnx
import tiktoken
import orbax.checkpoint as ocp
import time
from functools import partial

from layers import (
    LATENT_DIM,
    MAX_SEQ_LEN,
    PAD_TOKEN_ID,
    MAX_STEPS_LIMIT
)
from model import UniversalReasoner

from dotenv import load_dotenv
load_dotenv()

CHECKPOINT_DIR = os.path.abspath(os.environ.get("CHECKPOINT_ROOT", "orbax_checkpoints"))
HUNCH_REFRESH_EVERY = 4

def run_model_inference(
    model: UniversalReasoner,
    tokens: jnp.ndarray,
    max_steps: int = MAX_STEPS_LIMIT,
    should_refresh: bool = True,
) -> jnp.ndarray:
    out = model(
        tokens, max_steps=max_steps, training=False, should_refresh=should_refresh
    )
    return out.logits

@partial(nnx.jit, static_argnames=['refresh'])
def get_logits_for_token(model, padded_tks, token_idx, refresh):
    all_logits = run_model_inference(model, padded_tks, max_steps=MAX_STEPS_LIMIT, should_refresh=refresh)
    return all_logits[0, token_idx, :]

def generate_text(model, enc, prompt, max_new_tokens=256, temperature=0.5):
    seed = int(time.time() * 1000) % (2**31)
    rng = jax.random.PRNGKey(seed)

    tokens_list = enc.encode(prompt)
    valid_len = len(tokens_list)

    if valid_len >= MAX_SEQ_LEN:
        tokens_list = tokens_list[:MAX_SEQ_LEN]
        valid_len = MAX_SEQ_LEN

    padded_array = tokens_list + [PAD_TOKEN_ID] * (MAX_SEQ_LEN - valid_len)
    # Initialize tensor ONCE
    input_ids = jnp.array([padded_array], dtype=jnp.int32)

    print("🤖 Assistant: ", end="", flush=True)

    for i in range(max_new_tokens):
        if valid_len >= MAX_SEQ_LEN:
            break

        should_refresh = (i % HUNCH_REFRESH_EVERY == 0)

        logits = get_logits_for_token(model, input_ids, valid_len - 1, refresh=should_refresh)

        rng, subkey = jax.random.split(rng)

        if temperature > 0.0:
            scaled_logits = logits / temperature
            next_token = int(jax.random.categorical(subkey, scaled_logits))
        else:
            next_token = int(jnp.argmax(logits))

        if next_token == PAD_TOKEN_ID:
            break

        tokens_list.append(next_token)
        print(enc.decode([next_token]), end="", flush=True)

        input_ids = input_ids.at[0, valid_len].set(next_token)
        valid_len += 1

    print()
    return tokens_list

def run_inference():
    print(f"🔮 Initializing TinyRefinementModel (Dim={LATENT_DIM})...")

    enc = tiktoken.get_encoding("cl100k_base")

    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(0))

    active_checkpoint_dir = CHECKPOINT_DIR
    if os.environ.get("CHECKPOINT_ROOT") is None:
        from checkpoint_utils import discover_latest_checkpoint_run
        discovered_path, discovered_run_id = discover_latest_checkpoint_run()
        if discovered_path is not None:
            active_checkpoint_dir = discovered_path
            print(f"🔎 Auto-discovered latest checkpointed run for inference: {discovered_run_id}")
        else:
            print("❌ Error: No available weights here.")
            print("Please train the model first using: python start_training.py")
            return

    mngr = ocp.CheckpointManager(
        active_checkpoint_dir,
        item_names=('model', 'optimizer', 'monitor_state', 'step'),
    )

    latest_step = mngr.latest_step()
    if latest_step is None:
        print(f"❌ Error: No available weights here (no checkpoints found in {active_checkpoint_dir}).")
        print("Please train the model first using: python start_training.py")
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

            generate_text(model, enc, user_input)
            print("-" * 30)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    run_inference()