import os
import jax
import jax.numpy as jnp
from flax import nnx
import tiktoken
import pickle
import time

from train_local import (
    UniversalReasoner, 
    LATENT_DIM, 
    MAX_SEQ_LEN, 
    PAD_TOKEN_ID,
    MAX_STEPS_LIMIT
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def generate_dynamic(model, prompt_tokens, max_new_tokens, enc, max_ponder_steps=MAX_STEPS_LIMIT, threshold=0.9):
    batch_size = prompt_tokens.shape[0]
    current_tokens = prompt_tokens
    prompt_len = prompt_tokens.shape[1]
    
    print("🤖 Assistant: ", end="", flush=True)
    
    for i in range(max_new_tokens):
        # The new model handles internal pondering via its .infer() method
        # It returns logits for the entire sequence (B, S, V)
        logits = model.infer(current_tokens)
        
        # Greedily pick the next token from the last position
        next_token_logits = logits[:, -1, :]
        next_token = jnp.argmax(next_token_logits, axis=-1)[:, None]
        
        current_tokens = jnp.concatenate([current_tokens, next_token], axis=1)
        
        # Partial decoding for smooth UI
        decoded_text = enc.decode(current_tokens[0, prompt_len:].tolist())
        print(f"\r🤖 Assistant: {decoded_text}", end="", flush=True)
        
        if next_token[0, 0] == PAD_TOKEN_ID:
            break
            
    print()
    return current_tokens

def run_inference():
    print("🔮 Loading TinyRefinementModel for inference...")
    
    enc = tiktoken.get_encoding("gpt2")
    
    rngs = nnx.Rngs(42)
    model = UniversalReasoner(LATENT_DIM, rngs)
    
    checkpoint_path = "checkpoint.pkl"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint file '{checkpoint_path}' not found.")
        print("Please train the model first using train_local.py")
        return

    print(f"🔄 Loading weights from {checkpoint_path}...")
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)
    
    if "model_state" in ckpt:
        nnx.update(model, ckpt["model_state"])
    else:
        print("❌ Error: Could not find model state in checkpoint.")
        return
    print("✅ Model loaded and ready!")

    print("\n" + "="*50)
    print("Welcome to TinyRefinementModel CLI!")
    print("Type your prompt and press Enter.")
    print("Type '/exit' to quit.")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("👤 User: ").strip()
            
            if user_input.lower() == "/exit":
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue

            tokens_list = enc.encode(user_input)
            if len(tokens_list) > MAX_SEQ_LEN - 64: # Leave some space for generation
                print(f"⚠️ Warning: Prompt is long ({len(tokens_list)} tokens). Truncating...")
                tokens_list = tokens_list[-(MAX_SEQ_LEN - 64):]
            
            tokens_in = jnp.array([tokens_list], dtype=jnp.int32)
            
            gen_key = jax.random.key(int(time.time()))
            
            gen_tokens = generate_dynamic(model, tokens_in, max_new_tokens=128, enc=enc)
            
            print("\n" + "-"*30)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    run_inference()