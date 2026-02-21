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


def generate_dynamic(model, prompt_tokens, max_new_tokens, enc, max_ponder_steps=MAX_STEPS_LIMIT, threshold=0.5):
    batch_size = prompt_tokens.shape[0]
    
    current_z = model.embed(prompt_tokens)
    prompt_len = prompt_tokens.shape[1]
    
    print("🤖 Assistant: ", end="", flush=True)
    
    for step in range(max_new_tokens):
        seq_len = current_z.shape[1]
        
        pad_tokens = jnp.full((batch_size, 1), PAD_TOKEN_ID, dtype=jnp.int32)
        pad_z = model.embed(pad_tokens)
        z_seq = jnp.concatenate([current_z, pad_z], axis=1)
        new_seq_len = seq_len + 1
        
        z_scratch = jnp.tile(model.scratch_token.value, (batch_size, 1, 1))
        z_combined = jnp.concatenate([z_seq, z_scratch], axis=1)
        mask = model.get_mask(new_seq_len)

        def cond_fn(state):
            p_step, _, halt_prob = state
            return jnp.logical_and(p_step < max_ponder_steps, jnp.all(halt_prob < threshold))

        def body_fn(state):
            p_step, curr_z, _ = state
            
            t_signal = model.time_embed(p_step)[None, None, :]
            z_scratch_with_time = curr_z[:, new_seq_len:, :] + t_signal
            z_input = jnp.concatenate([curr_z[:, :new_seq_len, :], z_scratch_with_time], axis=1)
            
            new_z_raw, _ = model.processor(z_input, mask)

            curr_seq = curr_z[:, :new_seq_len, :]
            new_seq_raw = new_z_raw[:, :new_seq_len, :]

            salience = jax.nn.sigmoid(model.salience_head(curr_seq))
            new_seq = curr_seq + salience * (new_seq_raw - curr_seq)
            new_scratch = new_z_raw[:, new_seq_len:, :]
            
            new_z = jnp.concatenate([new_seq, new_scratch], axis=1)

            latent_shift = jnp.mean(jnp.abs(new_seq - curr_seq), axis=(1, 2))
            base_halt_logit = model.halt_head(new_scratch).mean(axis=(1, 2))
            halt_prob = jax.nn.sigmoid(base_halt_logit - latent_shift)
            
            return (p_step + 1, new_z, halt_prob)

        init_state = (0, z_combined, jnp.zeros((batch_size,)))
        final_step, final_z, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
        
        current_z = final_z[:, :new_seq_len, :]
        
        logits = model.decoder(current_z)
        current_tokens = jnp.argmax(logits, axis=-1)[0]
        
        generated_tokens = [t for t in current_tokens[prompt_len:].tolist() if t != PAD_TOKEN_ID]
        text_output = enc.decode(generated_tokens)
        
        print(f"\r🤖 Assistant: {text_output}", end="", flush=True)
        
        if current_tokens[-1] == PAD_TOKEN_ID:
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
    elif "state" in ckpt:
        state = ckpt["state"]
        if "model" in state:
            nnx.update(model, state["model"])
        else:
            nnx.update(model, state)
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