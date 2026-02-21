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
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
 
def generate(model, tokens, gen_len, key, temperature=0.7, top_p=0.9, top_k=50):
    batch_size, start_len = tokens.shape
    padded_tokens = jnp.pad(tokens, ((0, 0), (0, gen_len)), constant_values=PAD_TOKEN_ID)

    def cond_fn(state):
        step, current_tokens, loop_key = state
        under_limit = step < gen_len

        last_token = current_tokens[0, start_len + step - 1]
        not_finished = last_token != PAD_TOKEN_ID
        
        return jnp.logical_and(under_limit, not_finished)

    def body_fn(state):
        step, current_tokens, loop_key = state
        step_key, next_loop_key = jax.random.split(loop_key)
        logits = model.infer(current_tokens, training=False)
        
        valid_idx = start_len + step - 1
        next_logits = logits[:, valid_idx, :] / jnp.maximum(temperature, 1e-6)

        if top_k > 0:
            top_k_values, _ = jax.lax.top_k(next_logits, top_k)
            k_min = top_k_values[:, -1:]
            next_logits = jnp.where(next_logits < k_min, -1e9, next_logits)
        
        sorted_indices = jnp.argsort(next_logits, axis=-1)[:, ::-1]
        sorted_logits = jnp.take_along_axis(next_logits, sorted_indices, axis=-1)
        probs = jax.nn.softmax(sorted_logits, axis=-1)
        cum_probs = jnp.cumsum(probs, axis=-1)  #haha
        
        mask = cum_probs > top_p
        mask = jnp.concatenate([jnp.zeros((batch_size, 1), dtype=bool), mask[:, :-1]], axis=-1)
        filtered_logits = jnp.where(mask, -1e9, sorted_logits)
        
        sample_indices = jax.random.categorical(step_key, filtered_logits)
        next_token = jnp.take_along_axis(sorted_indices, sample_indices[:, None], axis=-1)
        
        new_tokens = jax.lax.dynamic_update_slice(current_tokens, next_token, (0, start_len + step))
        return (step + 1, new_tokens, next_loop_key)
    
    init_state = (0, padded_tokens, key)

    final_tokens, final_step, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return final_tokens

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
        state = ckpt['state']
    
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
            
            gen_key = jax.random.key(int(time.time()))
            
            # We'll generate a chunk of tokens. 
            # In a more advanced version, we could do streaming token by token,
            # but UniversalReasoner.generate is currently designed for batch generation.
            gen_tokens = generate(model, tokens_in, gen_len=128, key=gen_key)
            
            new_tokens = gen_tokens[0, len(tokens_list):].tolist()
            
            actual_tokens = [t for t in new_tokens if t != PAD_TOKEN_ID]
            response = enc.decode(actual_tokens)
            
            print(response.strip())
            print("\n" + "-"*30)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    run_inference()