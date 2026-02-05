import os
import pickle
import json
import time
import jax
import jax.numpy as jnp
from flax import nnx
import optax

MAX_N = 64         # Maximum particles (The World Size)
LATENT_DIM = 512   # Brain Size (Keep smaller to save VRAM for particles)
BATCH_SIZE = 128    # Micro-batch size
ACCUM_STEPS = 2    # Gradient accumulation (Total Batch = 256)

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# --- 1. THE UNIFIED PHYSICS ENGINE ---
class PhysicsWorld:
    @staticmethod
    def get_input_dim():
        return MAX_N * 6

    @staticmethod
    def get_output_dim():
        return MAX_N * 2

    @staticmethod
    def generate_batch(key, batch_size, difficulty):
        # 0.0 - 1.0: Linear (1 particle)
        # 1.0 - 2.0: Projectile/Orbit (2-3 particles)
        # 2.0 - 10.0: Chaos/Fluids (up to MAX_N particles)
        
        # Scale active particles based on difficulty
        target_n = 1.0 + (difficulty * 3.0)
        active_count = jnp.clip(target_n, 1.0, MAX_N).astype(jnp.int32)
        
        G = jnp.clip(difficulty, 1.0, 10.0)             # Gravity Strength
        repulsion = jnp.clip(difficulty - 0.5, 0.0, 5.0) * 0.5 # Fluid Pressure
        
        k1, k2, k3 = jax.random.split(key, 3)
        
        pos = jax.random.uniform(k1, (batch_size, MAX_N, 2), minval=-5, maxval=5)
        vel = jax.random.normal(k2, (batch_size, MAX_N, 2)) * 0.5
        mass = jax.random.uniform(k3, (batch_size, MAX_N, 1), minval=0.5, maxval=2.0)
        
        indices = jnp.arange(MAX_N)[None, :, None]
        mask = (indices < active_count).astype(jnp.float32)
        
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
        dtype = jnp.bfloat16
        
        self.encoder = nnx.Linear(PhysicsWorld.get_input_dim(), latent_dim, dtype=dtype, rngs=rngs)
        self.decoder = nnx.Linear(latent_dim, PhysicsWorld.get_output_dim(), dtype=dtype, rngs=rngs)
        self.recog_fc = nnx.Linear(latent_dim, PhysicsWorld.get_input_dim(), dtype=dtype, rngs=rngs)
        
        self.update_layer = nnx.Linear(latent_dim + 1, latent_dim, dtype=dtype, rngs=rngs)
        
        self.norm = nnx.LayerNorm(latent_dim, dtype=dtype, rngs=rngs)
        self.halt_fc = nnx.Linear(latent_dim, 1, dtype=dtype, rngs=rngs)
        self.complexity_head = nnx.Linear(latent_dim, 1, dtype=dtype, rngs=rngs)

    def __call__(self, raw_input, max_steps=40, training=False, key=None):
        z = nnx.gelu(self.encoder(raw_input.astype(jnp.bfloat16)))
        
        # Calculate batch size from input (e.g., 1024)
        batch_size = z.shape[0] if z.ndim > 1 else 1
        
        predicted_steps = nnx.sigmoid(self.complexity_head(z)) * max_steps
        
        if training and key is not None:
            # This handles both shape (Batch,) [New JAX] and (Batch, 2) [Old JAX]
            is_key_batched = (key.ndim > 0) and (key.shape[0] == batch_size)
            
            if is_key_batched:
                step_keys = jax.vmap(lambda k: jax.random.split(k, max_steps))(key)
                step_keys = jnp.swapaxes(step_keys, 0, 1)
            else:
                # Case: Single key
                step_keys = jax.random.split(key, max_steps)
        else:
            # Dummy keys must match batch shape for scan
            if z.ndim > 1:
                step_keys = jnp.zeros((max_steps, batch_size, 2), dtype=jnp.uint32)
            else:
                step_keys = jnp.zeros((max_steps, 2), dtype=jnp.uint32)

        def refine_step(carry, step_key_input):
            curr_z, step_idx, run_prob, w_out, w_z = carry
            
            # 1. Feature Engineering
            step_feat = jnp.full((curr_z.shape[0], 1), step_idx, dtype=curr_z.dtype)
            combined = jnp.concatenate([curr_z, step_feat], axis=-1)
            
            # 2. Calculate Update (ResNet block)
            update = nnx.gelu(self.update_layer(combined))
            next_z_raw = curr_z + update
            
            # 3. Noise
            if training:
                noise = jax.random.normal(step_key_input, next_z_raw.shape, dtype=curr_z.dtype) * 0.02
                next_z_raw = next_z_raw + noise
            
            next_z = self.norm(next_z_raw)
            
            # 4. Halt Logic
            halt = nnx.sigmoid(self.halt_fc(next_z))
            p = halt * (1.0 - run_prob)
            
            # 5. Accumulate
            new_out = w_out + (p * self.decoder(next_z))
            new_z = w_z + (p * next_z)
            
            return (next_z, step_idx + 1, run_prob + p, new_out, new_z), p

        # Init Carry
        init_carry = (
            z,
            0,
            jnp.zeros((batch_size, 1), dtype=jnp.bfloat16),
            jnp.zeros((batch_size, PhysicsWorld.get_output_dim()), dtype=jnp.bfloat16),
            jnp.zeros((batch_size, self.latent_dim), dtype=jnp.bfloat16)
        )
        
        (final_z, _, final_prob, w_out, w_z), step_probs = jax.lax.scan(
            refine_step, init_carry, step_keys, length=max_steps
        )
        
        rem = 1.0 - final_prob
        w_out = w_out + (rem * self.decoder(final_z))
        w_z = w_z + (rem * final_z)
        
        # Cast output back to float32 for loss calculation stability
        return w_out.astype(jnp.float32), w_z.astype(jnp.float32), self.recog_fc(w_z).astype(jnp.float32), step_probs, predicted_steps

# --- 3. TRAINING LOOP ---
@nnx.jit(static_argnums=(3,))
def train_step(model, optimizer, subkeys, micro_batch, difficulty, baseline_error):
    loss_scale = 1000.0
    
    def loss_fn(model):
        graphdef, state = nnx.split(model)
        
        def scan_body(current_baseline, key):
            # Split key: one for Physics gen, one for Latent Noise
            phys_key, noise_key = jax.random.split(key)
            
            m = nnx.merge(graphdef, state)
            inputs, targets = jax.lax.stop_gradient(
                PhysicsWorld.generate_batch(phys_key, micro_batch, difficulty)
            )
            
            # --- START ADVERSARIAL ATTACK ---
            def attack_loss_fn(input_perturbation):
                corrupted_inputs = inputs + input_perturbation
                preds, _, _, _, _ = m(corrupted_inputs, training=False, key=noise_key)
                return jnp.mean((preds - targets) ** 2)

            attack_grad = jax.grad(attack_loss_fn)(jnp.zeros_like(inputs))
            epsilon = 0.05 
            adversarial_inputs = inputs + (epsilon * jnp.sign(attack_grad))
            # --- END ADVERSARIAL ATTACK ---
            
            # --- GRPO/RFT SETUP ---
            G = 8
            expanded_inputs = jnp.repeat(adversarial_inputs, G, axis=0)
            expanded_targets = jnp.repeat(targets, G, axis=0)
            batch_keys = jax.random.split(noise_key, expanded_inputs.shape[0])
            
            # Run model on super-batch
            preds, _, recognition, step_probs, pred_steps = m(expanded_inputs, 40, True, batch_keys)
            
            # Calculate Errors per clone
            sq_err = jnp.mean((preds - expanded_targets) ** 2, axis=-1)
            
            # --- ADAPTIVE REJECTION SAMPLING (RFT) ---
            # Define Success: Beat the historical baseline (with pressure to improve)
            dynamic_threshold = current_baseline * 0.95
            
            # Create Mask (1.0 if beat history, 0.0 if failed)
            winners_mask = (sq_err < dynamic_threshold).astype(jnp.float32)
            
            # Fallback Logic: If no one beats the threshold, fall back to standard MSE
            # so we don't stop learning when things get hard.
            num_winners = jnp.sum(winners_mask)
            effective_mask = jnp.where(num_winners > 0, winners_mask, jnp.ones_like(winners_mask))
            
            safe_denom = jnp.sum(effective_mask) + 1e-6
            main_loss = jnp.sum(sq_err * effective_mask) / safe_denom
            
            # Update the local baseline for the next accumulation step (EMA)
            current_batch_avg = jnp.mean(sq_err)
            next_baseline = (0.99 * current_baseline) + (0.01 * current_batch_avg)
            # --- END RFT LOGIC ---
            
            # Loss Components (using expanded batch)
            recog_loss = jnp.mean((recognition - expanded_inputs) ** 2)
            
            # Calculate actual steps taken per sample
            # step_probs shape: (Batch*G, MaxSteps, 1)
            steps_range = jnp.arange(step_probs.shape[1], dtype=jnp.float32)[None, :, None]
            actual_steps = jnp.sum(step_probs * steps_range, axis=1)
            
            # Planner Loss
            diff = pred_steps - actual_steps
            planner_err_sq = jnp.where(diff < 0, 3.0 * (diff**2), diff**2)
            planner_loss = jnp.mean(planner_err_sq)
            
            # Ponder Loss
            avg_steps = jnp.mean(actual_steps)
            
            loss = main_loss + (0.5 * recog_loss) + (planner_loss * 0.1) + (avg_steps * 0.005)

            # Calculate the loss of only the "winners" (or best case) for the telemetry
            # We reuse effective_mask to get the same logic the gradients used
            telemetry_loss = jnp.sum(sq_err * effective_mask) / (jnp.sum(effective_mask) + 1e-6)

            metrics = {
                'main_loss': jnp.mean(sq_err),      # Keep this for general debug
                'best_loss': telemetry_loss,
                'planner_err': jnp.mean(jnp.abs(diff)),
                'avg_steps': avg_steps,
                'baseline': current_baseline
            }
            return next_baseline, (loss * loss_scale, metrics)

        updated_baseline, (losses_with_scale, all_metrics) = jax.lax.scan(scan_body, baseline_error, subkeys)
        
        # Mean across accumulation steps
        avg_metrics = jax.tree.map(jnp.mean, all_metrics)
        return jnp.mean(losses_with_scale), (avg_metrics, updated_baseline)

    (loss_s, (metrics, new_baseline)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    grads = jax.tree.map(lambda g: g / loss_scale, grads)
    optimizer.update(model, grads)
    
    total_loss = loss_s
    return total_loss / loss_scale, metrics, new_baseline

# --- 4. EXECUTION ---
print(f"🚀 Initializing Infinite Physics (Max N={MAX_N})...")
model = RefineMathPhysics(LATENT_DIM, nnx.Rngs(42))
optimizer = nnx.Optimizer(model, optax.adam(3e-4), wrt=nnx.Param)

# Infinite Loop State
key = jax.random.key(0)
loss_history = []
difficulty = 0.0
start_time = time.time()

# PID State for Auto-Pacer
integral_error = 0.0
prev_error = 0.0
kp, ki, kd = 0.005, 0.0001, 0.01  # Tuning knobs

# --- ADAPTIVE RFT STATE ---
baseline_error = 0.1

# --- LOAD CHECKPOINT ---
ckpt_path = "physics_ckpt.pkl"
if os.path.exists(ckpt_path):
    print(f"📂 Loading checkpoint from {ckpt_path}...")
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
        nnx.update(model, ckpt['state'])
        # We only use the weights; reset difficulty to 0.0 for the 're-training'
        difficulty = 0.0
    print(f"✅ Loaded checkpoint weights | Resetting to Difficulty 0.000")

# --- RESET LOGGING ---
log_file = "training_log.csv"
if os.path.exists(log_file):
    os.remove(log_file) # Start fresh

print("🔥 Compiling Kernels (This may take 30s)...")

for step in range(50000):
    key, subkey = jax.random.split(key)
    subkeys = jax.random.split(subkey, ACCUM_STEPS)
    
    loss_val, step_metrics, baseline_error = train_step(model, optimizer, subkeys, BATCH_SIZE, difficulty, baseline_error)
    
    # --- AUTO-PACER (PID Controller) ---
    loss_history.append(float(step_metrics['best_loss']))
    if len(loss_history) > 50: loss_history.pop(0)
    avg_main_loss = sum(loss_history) / len(loss_history)
    
    planner_err = float(step_metrics['planner_err'])
    avg_steps = float(step_metrics['avg_steps'])

    # PID Calculation
    current_loss = avg_main_loss
    # Dynamic Target: We accept higher loss for higher difficulty
    dynamic_target = 0.05 + (0.02 * difficulty) 
    error = dynamic_target - current_loss

    P = kp * error
    I = ki * integral_error
    D = kd * (error - prev_error)
    adjustment = P + I + D

    # Update Difficulty (Smoothly)
    difficulty += adjustment
    difficulty = max(0.0, difficulty) # Clamp at 0

    # Update state
    integral_error += error
    prev_error = error

    # Reset integral if we swing too hard (Anti-windup)
    if abs(error) > 0.1: integral_error = 0.0

    if step % 50 == 0:
        sps = 50 / (time.time() - start_time + 1e-6)
        active = min(1.0 + (difficulty * 3.0), MAX_N)
        print(f"Step {step} Diff: {difficulty:.3f} (N~{int(active)}) | Loss: {avg_main_loss:.4f} | Base: {baseline_error:.4f} | PlanErr: {planner_err:.2f} | {sps:.1f} steps/s")
        start_time = time.time()
        
        # Save telemetry for plotting (Fast & Infinite)
        file_exists = os.path.isfile(log_file)
        with open(log_file, "a") as f:
            if not file_exists:
                f.write("step,difficulty,loss,avg_loss,speed,planner_err,avg_steps\n")
            f.write(f"{step},{difficulty:.4f},{float(loss_val):.4f},{avg_main_loss:.4f},{sps:.1f},{planner_err:.4f},{avg_steps:.2f}\n")
            
        if step % 1000 == 0 and step > 0:
            with open("physics_ckpt.pkl", "wb") as f:
                pickle.dump({'state': nnx.state(model), 'difficulty': float(difficulty)}, f)