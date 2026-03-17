import os
import jax
import jax.numpy as jnp
from flax import nnx
import tiktoken
import orbax.checkpoint as ocp
import time
from functools import partial

from train_local import (
    UniversalReasoner,
    LATENT_DIM,
    MAX_SEQ_LEN,
    PAD_TOKEN_ID,
    MAX_STEPS_LIMIT,
    HUNCH_REFRESH_EVERY
)
from inference_core import run_model_inference

from dotenv import load_dotenv

load_dotenv()

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_ROOT", "orbax_checkpoints")
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

@partial(nnx.jit, static_argnames=['refresh'])
def get_all_logits(model, padded_tks, refresh):
    return run_model_inference(model, padded_tks, max_steps=MAX_STEPS_LIMIT, should_refresh=refresh)

def generate_text(model, enc, prompt, max_new_tokens=128, temperature=0.8):
    seed = int(time.time() * 1000) % (2**31)
    rng = jax.random.PRNGKey(seed)

    tokens_list = enc.encode(prompt)

    print("🤖 Assistant: ", end="", flush=True)

    for i in range(max_new_tokens):
        valid_len = len(tokens_list)

        if valid_len >= MAX_SEQ_LEN:
            break

        padded_array = tokens_list + [PAD_TOKEN_ID] * (MAX_SEQ_LEN - valid_len)
        input_ids = jnp.array([padded_array], dtype=jnp.int32)

        should_refresh = (i % HUNCH_REFRESH_EVERY == 0)

        # Call JIT function with the whole sequence
        all_logits = get_all_logits(model, input_ids, refresh=should_refresh)
        
        # Slice the specific token we need OUTSIDE of JIT
        logits = all_logits[0, valid_len - 1, :]

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

            generate_text(model, enc, user_input)
            print("-" * 30)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    run_inference()