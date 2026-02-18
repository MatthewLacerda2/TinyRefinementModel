import os
import jax
import jax.numpy as jnp
from flax import nnx
import tiktoken
import pickle
import time

# Import model architecture and constants from train_local
from train_local import (
    UniversalReasoner, 
    LATENT_DIM, 
    MAX_SEQ_LEN, 
    PAD_TOKEN_ID,
    SCRATCH_SLOTS,
    MAX_STEPS_LIMIT
)

# Prevent JAX from pre-allocating all VRAM
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def run_inference():
    print("🔮 Loading TinyRefinementModel for inference...")
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Initialize model skeleton
    rngs = nnx.Rngs(42)
    model = UniversalReasoner(LATENT_DIM, rngs)
    
    # Load checkpoint
    checkpoint_path = "checkpoint.pkl"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint file '{checkpoint_path}' not found.")
        print("Please train the model first using train_local.py")
        return

    print(f"🔄 Loading weights from {checkpoint_path}...")
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)
        state = ckpt['state']
    
    # Apply weights to model
    graphdef, _ = nnx.split(model)
    model = nnx.merge(graphdef, state)
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

            # Encode input
            tokens_list = enc.encode(user_input)
            if len(tokens_list) > MAX_SEQ_LEN - 64: # Leave some space for generation
                print(f"⚠️ Warning: Prompt is long ({len(tokens_list)} tokens). Truncating...")
                tokens_list = tokens_list[-(MAX_SEQ_LEN - 64):]
            
            tokens_in = jnp.array([tokens_list], dtype=jnp.int32)
            
            print("🤖 Assistant: ", end="", flush=True)
            
            # Generate response
            # Note: We use a different key each time for variety
            gen_key = jax.random.key(int(time.time()))
            
            # We'll generate a chunk of tokens. 
            # In a more advanced version, we could do streaming token by token,
            # but UniversalReasoner.generate is currently designed for batch generation.
            gen_tokens = model.generate(tokens_in, gen_len=128, key=gen_key)
            
            # Extract only the generated part
            new_tokens = gen_tokens[0, len(tokens_list):].tolist()
            
            # Decode and print
            # We filter out the padding tokens
            actual_tokens = [t for t in new_tokens if t != PAD_TOKEN_ID]
            response = enc.decode(actual_tokens)
            
            # Print response (handling the end sequence if model learns one, 
            # though here we just print what it gives us)
            print(response.strip())
            print("\n" + "-"*30)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    run_inference()