import os
import pickle
import json
import time
import jax
import jax.numpy as jnp
from flax import nnx
import optax

# Configuration for RTX 2060 (6GB VRAM)
# We set XLA to not preallocate everything so you don't OOM immediately.
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# --- 1. THE PHYSICS GENERATOR (Infinite Gym) ---
class PhysicsWorld:
    """
    Procedurally generates physics problems of increasing difficulty.
    Input: Initial State (Problem)
    Target: Future State (Solution)
    """
    
    @staticmethod
    def get_input_dim():
        # Maximum dimensions needed for 3-Body problem (3 bodies * 5 vars)
        return 15 

    @staticmethod
    def get_output_dim():
        # Maximum dimensions for output (3 bodies * 2 positions)
        return 6

    @staticmethod
    @jax.jit
    def generate_batch(key, batch_size, level):
        """
        Switches between physics engines based on 'level'.
        Returns: 
           raw_inputs: (B, 15) - The initial conditions
           raw_targets: (B, 6) - The ground truth future state
        """
        def level_0_linear(k):
            # Level 0: 1D Motion (x_f = x_i + v*t)
            # Simple linear extrapolation. Good for warming up the Encoder.
            k1, k2, k3 = jax.random.split(k, 3)
            x = jax.random.uniform(k1, (batch_size, 1), minval=-10, maxval=10)
            v = jax.random.uniform(k2, (batch_size, 1), minval=-5, maxval=5)
            t = jax.random.uniform(k3, (batch_size, 1), minval=1, maxval=5)
            
            target_x = x + v * t
            
            # Pack: [x, v, t, 0...] 
            inputs = jnp.concatenate([x, v, t, jnp.zeros((batch_size, 12))], axis=-1)
            # Target: [target_x, 0...]
            targets = jnp.concatenate([target_x, jnp.zeros((batch_size, 5))], axis=-1)
            return inputs, targets

        def level_1_projectile(k):
            # Level 1: 2D Ballistics (Parabolas) with gravity
            k1, k2, k3, k4, k5, k6 = jax.random.split(k, 6)
            x = jax.random.uniform(k1, (batch_size, 1), minval=-50, maxval=50)
            y = jax.random.uniform(k2, (batch_size, 1), minval=0, maxval=100)
            vx = jax.random.uniform(k3, (batch_size, 1), minval=-20, maxval=20)
            vy = jax.random.uniform(k4, (batch_size, 1), minval=-20, maxval=20)
            g = jax.random.uniform(k5, (batch_size, 1), minval=5, maxval=15)
            t = jax.random.uniform(k6, (batch_size, 1), minval=1, maxval=5)
            
            tf_x = x + vx * t
            tf_y = y + vy * t - 0.5 * g * t**2
            
            # Pack: [x, y, vx, vy, g, t, 0...]
            feats = jnp.concatenate([x, y, vx, vy, g, t], axis=-1)
            inputs = jnp.concatenate([feats, jnp.zeros((batch_size, 9))], axis=-1)
            
            # Target: [final_x, final_y, 0...]
            t_feats = jnp.concatenate([tf_x, tf_y], axis=-1)
            targets = jnp.concatenate([t_feats, jnp.zeros((batch_size, 4))], axis=-1)
            return inputs, targets

        def level_2_nbody(k):
            # Level 2: 3-Body Problem (Chaos)
            # We simulate 50 steps of gravity to find the truth.
            N = 3
            dt = 0.05
            steps = 40
            
            ks = jax.random.split(k, 4)
            pos = jax.random.uniform(ks[0], (batch_size, N, 2), minval=-2, maxval=2)
            vel = jax.random.normal(ks[1], (batch_size, N, 2)) * 0.3
            mass = jax.random.uniform(ks[2], (batch_size, N, 1), minval=0.5, maxval=2.0)
            
            # Simulation Loop (Ground Truth Generator)
            def get_acc(p, m):
                # Pairwise difference vectors
                diff = p[:, None, :, :] - p[:, :, None, :] # (B, N, N, 2)
                dist_sq = jnp.sum(diff**2, axis=-1, keepdims=True) + 1e-2
                force = (diff / jnp.sqrt(dist_sq)) * (m[:, None, :, :] / dist_sq)
                return jnp.sum(force, axis=2) # Sum forces

            def sim_step(carry, _):
                p, v = carry
                a = get_acc(p, mass)
                return (p + v*dt, v + a*dt), None

            (final_p, final_v), _ = jax.lax.scan(sim_step, (pos, vel), None, length=steps)
            
            # Pack Inputs: [p(3x2), v(3x2), m(3x1)] -> 15 vars
            inputs = jnp.concatenate([pos.reshape(batch_size, -1), vel.reshape(batch_size, -1), mass.reshape(batch_size, -1)], axis=-1)
            # Pack Targets: [final_p(3x2)] -> 6 vars
            targets = final_p.reshape(batch_size, -1)
            return inputs, targets

        # Switch logic: Level 0 -> 1 -> 2
        return jax.lax.switch(level, [level_0_linear, level_1_projectile, level_2_nbody], key)


# --- 2. THE REASONING MODEL ---
class RefineMathPhysics(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim
        
        # --- The Interface (Encoder/Decoder) ---
        # Projects Reality (15 vars) -> Latent Thought (768 vars)
        self.encoder = nnx.Linear(PhysicsWorld.get_input_dim(), latent_dim, rngs=rngs)
        
        # Projects Latent Thought -> Reality Prediction (6 vars)
        self.decoder = nnx.Linear(latent_dim, PhysicsWorld.get_output_dim(), rngs=rngs)
        
        # --- The Latent Brain (Refiner) ---
        # "Thinking" layers
        self.fc1 = nnx.Linear(latent_dim + 1, latent_dim * 2, rngs=rngs) # +1 for step counter
        self.fc2 = nnx.Linear(latent_dim * 2, latent_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(latent_dim, rngs=rngs)
        
        # Meta-Cognition: "Am I done thinking?"
        self.halt_fc = nnx.Linear(latent_dim, 1, rngs=rngs)

    def __call__(self, raw_input, max_steps=40):
        # 1. Encode: Turn the physics problem into a thought vector
        z = self.encoder(raw_input) 
        z = nnx.gelu(z) # Activation to make it non-linear
        
        # 2. Refine: The "Latent Loop"
        # We don't use a Python loop here for JAX compatibility in training, 
        # but conceptually this is the 'while' loop.
        
        def refine_step(carry):
            current_z, step = carry
            
            # Tell the model which step it is on
            step_feat = jnp.full((current_z.shape[0], 1), step, dtype=current_z.dtype)
            combined = jnp.concatenate([current_z, step_feat], axis=-1)
            
            # Resonate
            h = nnx.gelu(self.fc1(combined))
            delta = self.fc2(h)
            
            # Update thought (Residual connection)
            next_z = self.norm(current_z + 0.1 * delta)
            return next_z, step + 1

        # Run fixed steps for stability, or early exit if measuring inference
        final_z, _ = jax.lax.scan(lambda c, _: (refine_step(c)[0], None), (z, 0), None, length=max_steps)
        
        # 3. Decode: Translate the final thought back to physics numbers
        prediction = self.decoder(final_z)
        return prediction, final_z


# --- 3. TRAINING INFRASTRUCTURE ---
@nnx.jit
def train_step(model, optimizer, subkeys, micro_batch, level_int):
    loss_scale = 1000.0 # Standard scaling
    
    def loss_fn(model):
        graphdef, state = nnx.split(model)

        def scan_body(carry, key):
            m = nnx.merge(graphdef, state)
            
            # 1. Generate Data for current Level
            inputs, targets = PhysicsWorld.generate_batch(key, micro_batch, level_int)
            
            # 2. Forward Pass
            preds, final_z = m(inputs)
            
            # 3. Physics Loss (MSE)
            # We only care about the dimensions relevant to the current level
            # but since we zero-pad targets, MSE on all 6 dims works fine.
            mse = jnp.mean((preds - targets) ** 2)
            
            # 4. Latent Stability Loss (Energy Conservation Proxy)
            # Penalize exploding latent values
            stability = jnp.mean(final_z ** 2) * 1e-4
            
            loss = mse + stability
            return None, loss

        _, losses = jax.lax.scan(scan_body, None, subkeys)
        return jnp.mean(losses)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    grads = jax.tree.map(lambda g: g / loss_scale, grads)
    optimizer.update(model, grads)
    return loss / loss_scale

# --- 4. MAIN EXECUTION ---
latent_dim = 768
accum_steps = 8 # Accumulate gradients to simulate larger batch on RTX 2060
micro_batch = 64 # Fits in 6GB VRAM comfortably

# Initialize
model = RefineMathPhysics(latent_dim, nnx.Rngs(42))

# Optimization (Muon + Adam)
def muon_schedule(step):
    return optax.cosine_decay(0.01, 10000)(step)

def adam_schedule(step):
    return optax.cosine_decay(3e-4, 10000)(step)

# Simple partition: Weights (2D) -> Muon, Biases (1D) -> Adam
def param_labels(params):
    return jax.tree_util.tree_map(lambda p: 'muon' if p.ndim == 2 else 'adam', params)

tx = optax.multi_transform(
    {'muon': optax.contrib.muon(learning_rate=muon_schedule),
     'adam': optax.adam(learning_rate=adam_schedule)},
    param_labels
)

optimizer = nnx.Optimizer(model, tx)

# Loop
print("🚀 Starting Physics Refinement Training on RTX 2060...")
key = jax.random.key(0)
loss_history = []
current_level = 0 # Start at Level 0 (Linear)

for step in range(100000):
    key, subkey = jax.random.split(key)
    subkeys = jax.random.split(subkey, accum_steps)
    
    # Train
    loss = train_step(model, optimizer, subkeys, micro_batch, current_level)
    loss_val = float(loss)
    loss_history.append(loss_val)
    if len(loss_history) > 100: loss_history.pop(0)
    
    avg_loss = sum(loss_history) / len(loss_history)
    
    # --- AUTO LEVEL UP LOGIC ---
    # Thresholds: Simple linear (0.1), Projectile (0.5), N-Body (1.0)
    # If loss is super low for 100 steps, level up.
    if len(loss_history) == 100:
        if current_level == 0 and avg_loss < 0.1:
            current_level = 1
            loss_history = [] # Reset buffer
            print(f"\n🎉 LEVEL UP! Promoted to Level 1: Projectile Motion\n")
        elif current_level == 1 and avg_loss < 0.5:
            current_level = 2
            loss_history = []
            print(f"\n🎉 LEVEL UP! Promoted to Level 2: Three-Body Chaos\n")

    if step % 100 == 0:
        print(f"Step {step} | Level {current_level} | Loss: {loss_val:.4f} | Avg: {avg_loss:.4f}")
        
        # Save Checkpoint every 1k
        if step % 1000 == 0:
            with open("physics_ckpt.pkl", "wb") as f:
                pickle.dump({'state': nnx.state(model), 'level': current_level}, f)