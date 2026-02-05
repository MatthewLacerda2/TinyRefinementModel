import os
import pickle
import json
import time
import jax
import jax.numpy as jnp
from flax import nnx
import optax

# --- CONFIGURATION FOR RTX 2060 (6GB) ---
# If it crashes on startup, lower MAX_N.
# If it runs fine, try increasing MAX_N to 128 or 150.
MAX_N = 96         # Maximum particles (The World Size)
LATENT_DIM = 512   # Brain Size (Keep smaller to save VRAM for particles)
BATCH_SIZE = 64    # Micro-batch size
ACCUM_STEPS = 4    # Gradient accumulation (Total Batch = 256)

# Prevent JAX from pre-allocating 100% of memory (leaves room for OS)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# --- 1. THE UNIFIED PHYSICS ENGINE ---
class PhysicsWorld:
    @staticmethod
    def get_input_dim():
        # [x, y, vx, vy, mass, is_active] per particle
        return MAX_N * 6

    @staticmethod
    def get_output_dim():
        # Predict [x, y] for every particle
        return MAX_N * 2

    @staticmethod
    def generate_batch(key, batch_size, difficulty):
        # 0.0 - 1.0: Linear (1 particle)
        # 1.0 - 2.0: Projectile/Orbit (2-3 particles)
        # 2.0 - 10.0: Chaos/Fluids (up to MAX_N particles)
        
        # Scale active particles based on difficulty
        target_n = 1.0 + (difficulty * 3.0)
        active_count = jnp.clip(target_n, 1.0, MAX_N).astype(jnp.int32)
        
        # Difficulty Modifiers
        G = jnp.clip(difficulty, 1.0, 10.0)             # Gravity Strength
        repulsion = jnp.clip(difficulty - 2.0, 0.0, 5.0) # Fluid Pressure
        
        k1, k2, k3 = jax.random.split(key, 3)
        
        # Init all MAX_N particles
        pos = jax.random.uniform(k1, (batch_size, MAX_N, 2), minval=-5, maxval=5)
        vel = jax.random.normal(k2, (batch_size, MAX_N, 2)) * 0.5
        mass = jax.random.uniform(k3, (batch_size, MAX_N, 1), minval=0.5, maxval=2.0)
        
        # Create Mask (1 for active, 0 for ghost)
        indices = jnp.arange(MAX_N)[None, :, None]
        mask = (indices < active_count).astype(jnp.float32)
        
        # Zero out ghosts
        pos = pos * mask
        vel = vel * mask
        mass = mass * mask

        # --- PHYSICS KERNEL ---
        def get_acc(p, m, active_mask):
            # Pairwise Distances
            diff = p[:, :, None, :] - p[:, None, :, :]
            dist_sq = jnp.sum(diff**2, axis=-1, keepdims=True) + 1e-3
            dist = jnp.sqrt(dist_sq)
            
            # Gravity (Attraction)
            force_mag = (G * m[:, None, :, :] * m[:, :, None, :]) / dist_sq
            
            # Repulsion (Pressure - prevents collapse)
            repulse_force = (repulsion * 10.0) / (dist_sq * dist + 1e-3)
            
            total_f_mag = force_mag - repulse_force
            force_vec = (diff / dist) * total_f_mag
            
            # Mask interactions
            valid = active_mask[:, :, None, :] * active_mask[:, None, :, :]
            force_vec = force_vec * valid
            
            acc = jnp.sum(force_vec, axis=2) / (m + 1e-6)
            return acc * active_mask

        # Simulation Loop
        dt = 0.05
        steps = 40 

        def sim_step(carry, _):
            p, v = carry
            a = get_acc(p, mass, mask)
            v_new = v + a * dt
            p_new = p + v_new * dt
            
            # Bouncy Box Walls
            v_new = jnp.where((p_new > 10) | (p_new < -10), -v_new, v_new)
            p_new = jnp.clip(p_new, -10, 10)
            
            # Re-mask
            p_new = p_new * mask
            v_new = v_new * mask
            return (p_new, v_new), None

        (final_p, _), _ = jax.lax.scan(sim_step, (pos, vel), None, length=steps)

        mask_broadcasted = jnp.broadcast_to(mask, (batch_size, MAX_N, 1))
        
        # Flatten for Model
        inputs = jnp.concatenate([pos, vel, mass, mask_broadcasted], axis=-1).reshape(batch_size, -1)
        targets = final_p.reshape(batch_size, -1)
        
        return inputs, targets

# --- 2. THE BRAIN (PonderNet) ---
class RefineMathPhysics(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim
        self.encoder = nnx.Linear(PhysicsWorld.get_input_dim(), latent_dim, rngs=rngs)
        self.decoder = nnx.Linear(latent_dim, PhysicsWorld.get_output_dim(), rngs=rngs)
        self.recog_fc = nnx.Linear(latent_dim, PhysicsWorld.get_input_dim(), rngs=rngs)
        
        self.fc1 = nnx.Linear(latent_dim + 1, latent_dim * 2, rngs=rngs)
        self.fc2 = nnx.Linear(latent_dim * 2, latent_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(latent_dim, rngs=rngs)
        self.halt_fc = nnx.Linear(latent_dim, 1, rngs=rngs)
        self.complexity_head = nnx.Linear(latent_dim, 1, rngs=rngs)

    def __call__(self, raw_input, max_steps=40, training=False, key=None):
        z = nnx.gelu(self.encoder(raw_input))
        batch_size = z.shape[0]

        # Planner: Guess how hard this is before starting
        # We scale sigmoid output to [0, max_steps]
        predicted_steps = nnx.sigmoid(self.complexity_head(z)) * max_steps

        # Prepare keys for noise injection (one key per step)
        if training and key is not None:
            step_keys = jax.random.split(key, max_steps)
        else:
            # If not training, just pass dummy keys (zeros)
            step_keys = jnp.zeros((max_steps, 2), dtype=jnp.uint32)

        def refine_step(carry, step_key_input):
            # Unpack the "State Vector"
            # curr_z: Current thought position
            # curr_v: Current "Latent Velocity" (Momentum)
            curr_z, curr_v, step_idx, run_prob, w_out, w_z = carry
            
            # 1. Feature Engineering (Add Time)
            step_feat = jnp.full((curr_z.shape[0], 1), step_idx, dtype=curr_z.dtype)
            combined = jnp.concatenate([curr_z, step_feat], axis=-1)
            
            # 2. Calculate "Force" (The update vector)
            h = nnx.gelu(self.fc1(combined))
            force = self.fc2(h) # This is 'Acceleration'
            
            # 3. Apply Second-Order Dynamics (Momentum)
            # v_{t+1} = friction * v_t + force
            # friction=0.6 gives it "weight" but stops it from spiraling out of control
            next_v = (0.6 * curr_v) + (0.1 * force)
            
            # 4. Update Position
            next_z_raw = curr_z + next_v
            
            # Only applies if 'training' was True (handled by key splitting)
            if training:
                noise = jax.random.normal(step_key_input, next_z_raw.shape) * 0.02
                next_z_raw = next_z_raw + noise
            
            next_z = self.norm(next_z_raw)
            
            # 6. Halt Logic (PonderNet)
            halt = nnx.sigmoid(self.halt_fc(next_z))
            p = halt * (1.0 - run_prob)
            
            # 7. Accumulate Output
            new_out = w_out + (p * self.decoder(next_z))
            new_z = w_z + (p * next_z)
            
            return (next_z, next_v, step_idx + 1, run_prob + p, new_out, new_z), p

        # Initialize State
        # Velocity starts at zero
        init_v = jnp.zeros_like(z)
        
        init_carry = (
            z,                                   # curr_z
            init_v,                              # curr_v (NEW!)
            0,                                   # step_idx
            jnp.zeros((batch_size, 1)),          # run_prob
            jnp.zeros((batch_size, PhysicsWorld.get_output_dim())), # w_out
            jnp.zeros((batch_size, self.latent_dim))                # w_z
        )
        
        # Scan over step_keys to inject unique noise per step
        (final_z, _, _, final_prob, w_out, w_z), step_probs = jax.lax.scan(
            refine_step, 
            init_carry, 
            step_keys, # Iterate over these keys
            length=max_steps
        )
        
        # Handle Remainder
        rem = 1.0 - final_prob
        w_out = w_out + (rem * self.decoder(final_z))
        w_z = w_z + (rem * final_z)
        
        return w_out, w_z, self.recog_fc(w_z), step_probs, predicted_steps

# --- 3. TRAINING LOOP ---
@nnx.jit(static_argnums=(3,))
def train_step(model, optimizer, subkeys, micro_batch, difficulty):
    loss_scale = 1000.0
    
    def loss_fn(model):
        graphdef, state = nnx.split(model)
        
        def scan_body(carry, key):
            # Split key: one for Physics gen, one for Latent Noise
            phys_key, noise_key = jax.random.split(key)
            
            m = nnx.merge(graphdef, state)
            inputs, targets = PhysicsWorld.generate_batch(phys_key, micro_batch, difficulty)
            
            # Pass training=True and the noise_key
            preds, _, recognition, step_probs, pred_steps = m(inputs, training=True, key=noise_key)
            
            # Loss Components
            main_loss = jnp.mean((preds - targets) ** 2)
            recog_loss = jnp.mean((recognition - inputs) ** 2)
            
            # Calculate actual steps taken per sample
            steps_range = jnp.arange(step_probs.shape[0], dtype=jnp.float32)[:, None, None]
            actual_steps = jnp.sum(step_probs * steps_range, axis=0)
            
            # 1. Planner Loss: Asymmetric (Punish Hubris)
            # hubris = predicting easy when it's hard (underestimation)
            diff = pred_steps - actual_steps
            # If diff < 0 (underestimated), multiply penalty by 3.0
            planner_err_sq = jnp.where(diff < 0, 3.0 * (diff**2), diff**2)
            planner_loss = jnp.mean(planner_err_sq)
            
            # 2. Ponder Loss: Penalize thinking too long (efficiency)
            avg_steps = jnp.mean(actual_steps)
            
            loss = main_loss + (0.5 * recog_loss) + (planner_loss * 0.1) + (avg_steps * 0.005)
            
            metrics = {
                'main_loss': main_loss,
                'planner_err': jnp.mean(jnp.abs(diff)),
                'avg_steps': avg_steps
            }
            return None, (loss * loss_scale, metrics)

        _, (losses_with_scale, all_metrics) = jax.lax.scan(scan_body, None, subkeys)
        
        # Mean across accumulation steps
        avg_metrics = jax.tree.map(jnp.mean, all_metrics)
        return jnp.mean(losses_with_scale), avg_metrics

    loss_s, grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    grads = jax.tree.map(lambda g: g / loss_scale, grads)
    optimizer.update(model, grads)
    
    total_loss, metrics = loss_s
    return total_loss / loss_scale, metrics

# --- 4. EXECUTION ---
print(f"🚀 Initializing Infinite Physics (Max N={MAX_N})...")
model = RefineMathPhysics(LATENT_DIM, nnx.Rngs(42))
optimizer = nnx.Optimizer(model, optax.adam(3e-4), wrt=nnx.Param)

# Infinite Loop State
key = jax.random.key(0)
loss_history = []
difficulty = 0.0
target_loss = 0.05
start_time = time.time()

print("🔥 Compiling Kernels (This may take 30s)...")

for step in range(1000000):
    key, subkey = jax.random.split(key)
    subkeys = jax.random.split(subkey, ACCUM_STEPS)
    
    loss_val, step_metrics = train_step(model, optimizer, subkeys, BATCH_SIZE, difficulty)
    
    # --- SELF-AWARE AUTO-PACER ---
    loss_history.append(float(step_metrics['main_loss']))
    if len(loss_history) > 50: loss_history.pop(0)
    avg_main_loss = sum(loss_history) / len(loss_history)
    
    planner_err = float(step_metrics['planner_err'])
    avg_steps = float(step_metrics['avg_steps'])

    if len(loss_history) == 50:
        # Mastery Check: High accuracy, low steps, AND good planning
        if avg_main_loss < target_loss and planner_err < 1.0:
            # Everything is "too easy"
            difficulty += 0.003
        
        # Hubris Check: Getting it wrong, especially if it thought it was easy
        elif avg_main_loss > target_loss * 3 or planner_err > 5.0:
            # Back off - we are moving too fast
            difficulty = max(0.0, difficulty - 0.01)

    if step % 50 == 0:
        sps = 50 / (time.time() - start_time + 1e-6)
        active = min(1.0 + (difficulty * 3.0), MAX_N)
        print(f"Step {step} | Diff: {difficulty:.3f} (N~{int(active)}) | Loss: {avg_main_loss:.4f} | PlanErr: {planner_err:.2f} | {sps:.1f} steps/s")
        start_time = time.time()
        
        # Save telemetry for plotting
        telemetry = {
            'step': step,
            'difficulty': float(difficulty),
            'loss': loss_val,
            'avg_loss': float(avg_main_loss),
            'speed': float(sps)
        }
        
        history_file = "training_history.json"
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                try:
                    full_history = json.load(f)
                except:
                    full_history = []
        else:
            full_history = []
            
        full_history.append(telemetry)
        with open(history_file, "w") as f:
            json.dump(full_history, f)
            
        if step % 1000 == 0 and step > 0:
            with open("physics_ckpt.pkl", "wb") as f:
                pickle.dump({'state': nnx.state(model), 'difficulty': float(difficulty)}, f)