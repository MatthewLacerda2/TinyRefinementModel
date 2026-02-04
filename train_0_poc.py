import os
import pickle
import time
import jax
import jax.numpy as jnp
from flax import nnx
import optax

# Optimized for RTX 2060 / Spot TPU
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# --- 1. THE LEVEL 0 GENERATOR (1D Motion) ---
# We focus ONLY on this. One dimension. Pure continuous logic.
class PhysicsLevel0:
    @staticmethod
    def get_input_dim():
        return 3 # Position (x), Velocity (v), Time (t)

    @staticmethod
    def get_output_dim():
        return 1 # Final Position (x_f)

    @staticmethod
    @jax.jit
    def generate_batch(key, batch_size):
        # x_f = x_i + v * t
        k1, k2, k3 = jax.random.split(key, 3)
        
        # 1. Random continuous values
        x = jax.random.uniform(k1, (batch_size, 1), minval=-10.0, maxval=10.0)
        v = jax.random.uniform(k2, (batch_size, 1), minval=-5.0, maxval=5.0)
        t = jax.random.uniform(k3, (batch_size, 1), minval=0.5, maxval=5.0)
        
        # 2. The "Ground Truth" Physics
        target_x = x + (v * t)
        
        # 3. Pack Data
        # Input: [x, v, t]
        inputs = jnp.concatenate([x, v, t], axis=-1)
        # Target: [x_f]
        targets = target_x
        # Aux Target (For Recognition Circuit): [v, t]
        # The model must REMEMBER v and t while it thinks.
        aux_targets = jnp.concatenate([v, t], axis=-1)
        
        return inputs, targets, aux_targets

# --- 2. THE REFINER MODEL (With Recognition Circuit) ---
class RefinePhysics(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim
        
        # ENCODER: Reality (3 vars) -> Latent
        self.encoder = nnx.Linear(PhysicsLevel0.get_input_dim(), latent_dim, rngs=rngs)
        
        # DECODER: Latent -> Reality (1 var)
        self.decoder = nnx.Linear(latent_dim, PhysicsLevel0.get_output_dim(), rngs=rngs)
        
        # --- THE RECOGNITION CIRCUIT ---
        # This circuit reviews the latent thought and tries to recover the initial facts (v, t).
        # If this fails, the model has "forgotten" the physics parameters.
        self.recog_fc = nnx.Linear(latent_dim, 2, rngs=rngs) # Predicting v, t
        
        # --- THE REFINEMENT LOOP ---
        self.fc1 = nnx.Linear(latent_dim + 1, latent_dim * 2, rngs=rngs)
        self.fc2 = nnx.Linear(latent_dim * 2, latent_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(latent_dim, rngs=rngs)
        
        # Stop button
        self.halt_fc = nnx.Linear(latent_dim, 1, rngs=rngs)

    def __call__(self, raw_input, max_steps=20):
        # 1. Encode
        z = self.encoder(raw_input)
        z = nnx.gelu(z)
        
        # 2. Latent Refinement Loop
        def step_fn(carry):
            curr_z, step = carry
            
            # Inject Step Counter
            step_feat = jnp.full((curr_z.shape[0], 1), step, dtype=curr_z.dtype)
            combined = jnp.concatenate([curr_z, step_feat], axis=-1)
            
            # Thinking Step
            h = nnx.gelu(self.fc1(combined))
            delta = self.fc2(h)
            next_z = self.norm(curr_z + 0.1 * delta)
            
            # Recognition Check (Does it still know v and t?)
            recog_out = self.recog_fc(next_z)
            
            return next_z, step + 1, recog_out

        # We scan to run the loop efficiently on GPU/TPU
        # We collect 'recog_outs' to penalize forgetting LATER
        final_z, _, all_recogs = jax.lax.scan(
            lambda c, _: (step_fn(c)[0], step_fn(c)[1]), 
            (z, 0), 
            None, 
            length=max_steps
        )
        
        # 3. Decode final answer
        prediction = self.decoder(final_z)
        
        # Return answer + the recognition history (did it remember v & t?)
        # We perform one last check on the final_z
        final_recog = self.recog_fc(final_z)
        return prediction, final_z, final_recog

# --- 3. TRAINING STEP ---
@nnx.jit
def train_step(model, optimizer, subkeys, micro_batch):
    loss_scale = 1000.0
    
    def loss_fn(model):
        graphdef, state = nnx.split(model)

        def scan_body(carry, key):
            m = nnx.merge(graphdef, state)
            inputs, targets, aux_targets = PhysicsLevel0.generate_batch(key, micro_batch)
            
            preds, final_z, final_recog = m(inputs)
            
            # 1. Prediction Loss (Did it get the right answer?)
            main_loss = jnp.mean((preds - targets) ** 2)
            
            # 2. Recognition Loss (Did it remember v and t?)
            # This forces the latent space to REMAIN "physically grounded"
            recog_loss = jnp.mean((final_recog - aux_targets) ** 2)
            
            # 3. Stability (Don't explode)
            stability_loss = jnp.mean(final_z ** 2) * 1e-5
            
            total_loss = main_loss + (0.5 * recog_loss) + stability_loss
            return None, total_loss

        _, losses = jax.lax.scan(scan_body, None, subkeys)
        return jnp.mean(losses)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    grads = jax.tree.map(lambda g: g / loss_scale, grads)
    optimizer.update(model, grads)
    return loss / loss_scale

# --- 4. EXECUTION ---
latent_dim = 768
accum_steps = 4 
micro_batch = 64

model = RefinePhysics(latent_dim, nnx.Rngs(42))

# SOTA Optimizer: Muon for weights, Adam for biases
def muon_schedule(step): return optax.cosine_decay(0.01, 5000)(step)
def adam_schedule(step): return optax.cosine_decay(3e-4, 5000)(step)

def param_labels(params):
    return jax.tree_util.tree_map(lambda p: 'muon' if p.ndim == 2 else 'adam', params)

tx = optax.multi_transform(
    {'muon': optax.contrib.muon(learning_rate=muon_schedule),
     'adam': optax.adam(learning_rate=adam_schedule)},
    param_labels
)

optimizer = nnx.Optimizer(model, tx)

print("🚀 Training Level 0: 1D Continuous Motion...")
key = jax.random.key(0)

for step in range(5001):
    key, subkey = jax.random.split(key)
    subkeys = jax.random.split(subkey, accum_steps)
    
    loss = train_step(model, optimizer, subkeys, micro_batch)
    
    if step % 100 == 0:
        print(f"Step {step} | Loss: {loss:.6f}")

# TEST RUN
print("\n--- FINAL TEST ---")
test_in = jnp.array([[0.0, 2.0, 3.0]]) # Start 0, Velocity 2, Time 3 -> Should be 6.0
pred, _, recog = model(test_in)
print(f"Input: x=0, v=2, t=3")
print(f"Target: 6.0")
print(f"Model Prediction: {pred[0,0]:.4f}")
print(f"Recognition Circuit (Recovered v, t): {recog[0]}")