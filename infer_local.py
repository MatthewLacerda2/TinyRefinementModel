import os
import jax
import jax.numpy as jnp
from flax import nnx
import tiktoken
import time
import orbax.checkpoint as ocp

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

def generate_text(model, enc, prompt, max_new_tokens=128):
    tokens_list = enc.encode(prompt)
    
    if len(tokens_list) > MAX_SEQ_LEN - max_new_tokens:
        tokens_list = tokens_list[-(MAX_SEQ_LEN - max_new_tokens):]
    
    @nnx.jit
    def get_next_logits(m, tks, refresh):
        logits = run_model_inference(m, tks, max_steps=MAX_STEPS_LIMIT, should_refresh=refresh)
        return logits[:, -1, :]  # Return logits for the last token in sequence

    print("🤖 Assistant: ", end="", flush=True)
    
    for i in range(max_new_tokens):
        input_ids = jnp.array([tokens_list], dtype=jnp.int32)
        should_refresh = (i % HUNCH_REFRESH_EVERY == 0)
        
        logits = get_next_logits(model, input_ids, should_refresh)
        next_token = int(jnp.argmax(logits, axis=-1)[0])
        
        if next_token == PAD_TOKEN_ID:
            break
            
        tokens_list.append(next_token)
        
        print(enc.decode([next_token]), end="", flush=True)
        
        if len(tokens_list) >= MAX_SEQ_LEN:
            break
            
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