import jax
import jax.numpy as jnp
from flax import nnx
import optax
import tiktoken
from datasets import load_dataset

LATENT_DIM = 512
BATCH_SIZE = 8
ACCUM_STEPS = 16
MAX_STEPS_LIMIT = 8
MAX_SEQ_LEN = 128
SCRATCH_SLOTS = 64 # Half the MAX_SEQ_LEN
VOCAB_SIZE = 100277

class RotaryAttention(nnx.Module):
    def __init__(self, num_heads, in_features, rngs, dtype=jnp.float32):
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        self.scale = self.head_dim ** -0.5
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.head_dim, 2) / self.head_dim))
        t = jnp.arange(MAX_SEQ_LEN + SCRATCH_SLOTS)
        freqs = jnp.outer(t, inv_freq)
        self.sin_cached = jnp.sin(freqs)
        self.cos_cached = jnp.cos(freqs)
        self.q_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float16)
        self.k_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float16)
        self.v_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float16)
        self.o_proj = nnx.Linear(in_features, in_features, rngs=rngs, dtype=jnp.float16)

    def __call__(self, x, mask=None):
        b, s, d = x.shape
        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        sin = self.sin_cached[:s, :]
        cos = self.cos_cached[:s, :]
        sin_ext = jnp.stack([sin, sin], axis=-1).reshape(s, self.head_dim)[None, :, None, :]
        cos_ext = jnp.stack([cos, cos], axis=-1).reshape(s, self.head_dim)[None, :, None, :]
        def rotate(x_in):
            x_even, x_odd = x_in[..., 0::2], x_in[..., 1::2]
            x_rot = jnp.stack([-x_odd, x_even], axis=-1).reshape(x_in.shape)
            return x_in * cos_ext + x_rot * sin_ext
        q, k = rotate(q), rotate(k)
        logits = jnp.einsum('bshd,bthd->bhst', q, k) * self.scale
        if mask is not None:
            logits = jnp.where(mask, logits, -1e9)
        weights = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum('bhst,bthd->bshd', weights, v)
        return self.o_proj(out.reshape(b, s, d))

class StandardReasoningBlock(nnx.Module):
    def __init__(self, latent_dim, num_heads, rngs, dtype=jnp.float32):
        self.attn = RotaryAttention(num_heads, latent_dim, rngs, dtype)
        self.norm1 = nnx.LayerNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.norm2 = nnx.LayerNorm(latent_dim, rngs=rngs, dtype=dtype)
        self.mlp = nnx.Sequential(
            nnx.Linear(latent_dim, latent_dim * 4, rngs=rngs, dtype=dtype),
            nnx.gelu,
            nnx.Linear(latent_dim * 4, latent_dim, rngs=rngs, dtype=dtype)
        )

    def __call__(self, x, mask):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x

class UniversalReasoner(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim
        self.num_scratch = 64 
        
        self.embed = nnx.Embed(VOCAB_SIZE, latent_dim, dtype=jnp.float16, rngs=rngs)
        self.time_embed = nnx.Embed(MAX_STEPS_LIMIT + 1, latent_dim, dtype=jnp.float16, rngs=rngs)
        self.scratch_token = nnx.Param(jax.random.normal(rngs(), (1, self.num_scratch, latent_dim)) * 0.02)
        self.processor = StandardReasoningBlock(latent_dim, num_heads=8, rngs=rngs)
        
        self.halt_head = nnx.Linear(latent_dim, 1, dtype=jnp.float32, rngs=rngs)
        self.decoder = nnx.Linear(latent_dim, VOCAB_SIZE, dtype=jnp.float32, rngs=rngs)

    def get_mask(self, seq_len):
        total_len = seq_len + self.num_scratch
        
        causal = jnp.tril(jnp.ones((total_len, total_len), dtype=bool))
        
        mask = causal.at[seq_len:, :].set(True)
        
        return mask[None, None, :, :]

    def __call__(self, tokens, max_steps=MAX_STEPS_LIMIT, training=False, key=None):
        batch_size, seq_len = tokens.shape
        z_seq = self.embed(tokens)
        z_scratch = jnp.tile(self.scratch_token.value, (batch_size, 1, 1))
        
        z_combined = jnp.concatenate([z_seq, z_scratch], axis=1)
        mask = self.get_mask(seq_len)

        if training:
            active_steps = jax.random.randint(key, (), 1, max_steps + 1)
        else:
            active_steps = max_steps

        def loop_step(i, carry):
            curr_z, halts = carry
            
            t_signal = self.time_embed(i)[None, None, :] 
            
            # We add this signal to the CURRENT state before processing
            # This allows the dense layer to say "If step==0, do extraction. If step==8, do summary."
            z_with_time = curr_z + t_signal
            
            new_z = self.processor(z_with_time, mask)
            
            halt_logit = self.halt_head(new_z[:, seq_len:, :]).mean(axis=(1, 2))
            halt_score = nnx.sigmoid(halt_logit)
            
            return (new_z, halts.at[i].set(halt_score))

        init_halts = jnp.zeros((max_steps, batch_size))
        
        final_z, halt_scores = jax.lax.fori_loop(0, active_steps, loop_step, (z_combined, init_halts))
        
        final_seq = final_z[:, :seq_len, :]
        return self.decoder(final_seq), halt_scores

class TextDataGenerator:
    def __init__(self):
        self.dataset = load_dataset("HuggingFaceTB/cosmopedia-v2", split="train", streaming=True)
        self.iterator = iter(self.dataset)
        self.enc = tiktoken.get_encoding("cl100k_base")

    def get_batch(self, batch_size):
        batch_ids = []
        while len(batch_ids) < batch_size:
            try:
                item = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataset)
                item = next(self.iterator)
            text = item['text']
            tokens = self.enc.encode(text)
            if len(tokens) < MAX_SEQ_LEN:
                tokens = tokens + [100257] * (MAX_SEQ_LEN - len(tokens))
            else:
                tokens = tokens[:MAX_SEQ_LEN]
            batch_ids.append(tokens)
        return jnp.array(batch_ids, dtype=jnp.int32)

@nnx.jit
def train_step(model, optimizer, batch_tokens, key):
    inputs = batch_tokens[:, :-1]
    targets = batch_tokens[:, 1:]
    def loss_fn(model):
        # We pass a key so the model can pick a random depth
        preds, halt_scores = model(inputs, training=True, key=key)
        
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=preds, labels=targets)
        mask = (targets != 100257)
        token_loss = jnp.sum(ce_loss * mask) / jnp.sum(mask)
        
        # 2. Halt Loss: Teach it to say "I'm done" as steps increase
        # We want halt_score to approach 1.0 as we get deeper
        step_indices = jnp.arange(MAX_STEPS_LIMIT)[:, None] # [Steps, 1]
        target_halt = jnp.minimum(1.0, step_indices / 4.0) # Linearly go to 1.0 by step 4
        
        halt_loss = jnp.mean((halt_scores - target_halt) ** 2)
        
        total_loss = token_loss + 0.1 * halt_loss
        return total_loss, (token_loss, halt_loss)

    (loss, (raw_ce, h_loss)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, raw_ce, h_loss

if __name__ == "__main__":
    print(f"🚀 Initializing Standard Latent Reasoner (Dim={LATENT_DIM})...")
    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(42))
    optimizer = nnx.Optimizer(model, optax.adamw(3e-4), wrt=nnx.Param)
    data_gen = TextDataGenerator()
    key = jax.random.key(0)
    print("Starting training loop...")
    for step in range(1, 10000):
        key, subkey = jax.random.split(key)
        batch = data_gen.get_batch(BATCH_SIZE)
        loss, raw_ce, h_loss = train_step(model, optimizer, batch, subkey)
        if step % 50 == 0:
            print(f"Step {step:04d} | Loss: {loss:.4f} | CE: {raw_ce:.4f} | Halt Loss: {h_loss:.4f}")
            print("\n--- INFERENCE CHECK ---")
            prompt = "Once upon a time there was a specific"
            tokens = jnp.array([data_gen.enc.encode(prompt)], dtype=jnp.int32)
            logits, halt_scores = model(tokens, training=False)
            next_tok = jnp.argmax(logits[0, -1])
            print(f"Input: {prompt}")
            print(f"Halt Scores: {jnp.round(halt_scores[:, 0], 2)}")
            print(f"Next Token: {data_gen.enc.decode([next_tok.item()])}")
            print("-----------------------\n")