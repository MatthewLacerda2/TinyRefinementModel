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
        # Added 1 extra dimension for "Context" (Is this Space or Earth?)
        # This helps the brain switch its reasoning mode.
        return (MAX_N * 6) + 1 

    @staticmethod
    def get_output_dim():
        return MAX_N * 2

    @staticmethod
    def generate_batch(key, batch_size, difficulty, steps):
        # --- 1. SETUP CONSTANTS ---
        # We mix two modes: 
        # Mode 0 = Orbital/Space (Central Gravity, No Friction)
        # Mode 1 = Terrestrial/Earth (Down Gravity, Floor, Friction)
        
        mode_key, init_key, sim_key = jax.random.split(key, 3)
        
        # 50/50 chance of Space vs Earth
        mode = jax.random.bernoulli(mode_key, p=0.5, shape=(batch_size, 1)).astype(jnp.float32)
        
        # Difficulty scales the "Chaos"
        active_count = jnp.clip(2.0 + difficulty, 2.0, MAX_N).astype(jnp.int32)
        
        # --- 2. INITIALIZATION ---
        # Position: Spread out more for space, start higher up for Earth
        pos_space = jax.random.uniform(init_key, (batch_size, MAX_N, 2), minval=-5, maxval=5)
        pos_earth = jax.random.uniform(init_key, (batch_size, MAX_N, 2), minval=-4, maxval=4)
        # Shift earth particles up so they can fall
        pos_earth = pos_earth.at[:, :, 1].add(3.0) 
        
        pos = jnp.where(mode[..., None], pos_earth, pos_space)

        # Velocity: Space has high velocity (orbits), Earth has low velocity (drops/throws)
        vel = jax.random.normal(init_key, (batch_size, MAX_N, 2))
        vel = jnp.where(mode[..., None], vel * 0.5, vel * 1.5) # Slower on Earth
        
        mass = jax.random.uniform(init_key, (batch_size, MAX_N, 1), minval=0.5, maxval=3.0)

        # Masking inactive particles
        indices = jnp.arange(MAX_N)[None, :, None]
        mask = (indices < active_count).astype(jnp.float32)
        
        pos = pos * mask
        vel = vel * mask
        mass = mass * mask

        # --- 3. PHYSICS KERNEL ---
        def get_acc(p, v, m, active_mask, current_mode):
            # A. Particle Interaction (Everything repels to simulate solidity)
            diff = p[:, :, None, :] - p[:, None, :, :]
            dist_sq = jnp.sum(diff**2, axis=-1, keepdims=True) + 1e-4
            dist = jnp.sqrt(dist_sq)
            
            # Repulsion (Hooke's/Lennard-Jones style very short range)
            # This makes things "solid" rather than ghostly
            repulse = 50.0 * jnp.exp(-dist_sq * 2.0)
            
            # Mutual Gravity (Only matters in Space mode largely, but calculated always)
            force_g = (2.0 * m[:, None, :, :] * m[:, :, None, :]) / dist_sq
            
            total_force = (diff / dist) * (force_g - repulse)
            
            # Mask self-interaction and inactive
            valid = active_mask[:, :, None, :] * active_mask[:, None, :, :]
            # Remove diagonal (self-interaction)
            eye = jnp.eye(MAX_N)[None, :, :, None]
            valid = valid * (1.0 - eye)
            
            interaction_acc = jnp.sum(total_force * valid, axis=2) / (m + 1e-6)
            
            # B. Global Gravity
            # Space: Gravity is 0 (or central). Let's say 0 for pure momentum.
            # Earth: Gravity is (0, -9.8)
            gravity_earth = jnp.array([0.0, -9.8])[None, None, :]
            gravity_space = -0.5 * p # Slight central pull to keep things in frame
            
            env_acc = jnp.where(current_mode[..., None] > 0.5, gravity_earth, gravity_space)
            
            # C. Drag / Air Resistance (Crucial for human intuition)
            # Earth has air (drag), Space is vacuum (no drag)
            drag_factor = jnp.where(current_mode[..., None] > 0.5, 0.05, 0.0)
            drag_acc = -v * drag_factor
            
            return (interaction_acc + env_acc + drag_acc) * active_mask

        # Simulation Loop
        dt = 0.03 # Slightly finer timestep for collisions

        def sim_step(carry, _):
            p, v = carry
            a = get_acc(p, v, mass, mask, mode)
            
            v_new = v + a * dt
            p_new = p + v_new * dt
            
            # --- BOUNDARIES ---
            # Space: Bouncy box walls (-10 to 10)
            # Earth: Floor at -5.0, Walls at -10/10, Open Ceiling
            
            # 1. Floor Collision (Earth Only)
            floor_y = -5.0
            hit_floor = (p_new[:, :, 1] < floor_y) & (mode[:, :, 0] > 0.5)
            
            # Bounce: Reverse Y velocity, lose energy (inelastic collision)
            v_new = v_new.at[:, :, 1].set(
                jnp.where(hit_floor, -v_new[:, :, 1] * 0.7, v_new[:, :, 1])
            )
            p_new = p_new.at[:, :, 1].set(
                jnp.where(hit_floor, floor_y + 0.01, p_new[:, :, 1])
            )
            
            # Friction on floor (If touching floor, slow down X)
            v_new = v_new.at[:, :, 0].set(
                jnp.where(hit_floor, v_new[:, :, 0] * 0.9, v_new[:, :, 0])
            )

            # 2. General Box Walls (Left/Right/Top/Bottom for Space)
            # Simple clamp for stability
            v_new = jnp.where((p_new > 10) | (p_new < -10), -v_new * 0.8, v_new)
            p_new = jnp.clip(p_new, -10, 10)
            
            # Re-mask
            p_new = p_new * mask
            v_new = v_new * mask
            return (p_new, v_new), None

        (final_p, _), _ = jax.lax.scan(sim_step, (pos, vel), None, length=steps)

        mask_broadcasted = jnp.broadcast_to(mask, (batch_size, MAX_N, 1))
        
        # Flatten for Model
        # We append 'mode' once at the end to match get_input_dim(): (MAX_N * 6) + 1
        flat_particles = jnp.concatenate([pos, vel, mass, mask_broadcasted], axis=-1).reshape(batch_size, -1)
        inputs = jnp.concatenate([flat_particles, mode], axis=-1)
        
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
        
        # A 2-Layer MLP "Brain"
        self.update_fc1 = nnx.Linear(latent_dim + 1, latent_dim * 2, dtype=dtype, rngs=rngs)
        self.update_fc2 = nnx.Linear(latent_dim * 2, latent_dim, dtype=dtype, rngs=rngs)
        
        self.norm = nnx.LayerNorm(latent_dim, dtype=dtype, rngs=rngs)
        self.halt_fc = nnx.Linear(latent_dim, 1, dtype=dtype, rngs=rngs)
        self.complexity_head = nnx.Linear(latent_dim, 1, dtype=dtype, rngs=rngs)

    def __call__(self, raw_input, max_steps=40, training=False, key=None):
        z = nnx.gelu(self.encoder(raw_input.astype(jnp.bfloat16)))
        
        # Calculate batch size from input (e.g., 1024)
        batch_size = z.shape[0] if z.ndim > 1 else 1
        
        predicted_steps = nnx.sigmoid(self.complexity_head(z)) * max_steps
        
        if training and key is not None:
            # Check if key is batched by comparing dimension to input batch size
            is_key_batched = (key.ndim > 0) and (key.shape[0] == batch_size)
            
            if is_key_batched:
                # Case: Batch of keys 
                # Split: (Batch, Steps) -> Swap: (Steps, Batch)
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
            
            # 2. Calculate Update (MLP Block)
            hidden = nnx.gelu(self.update_fc1(combined))
            update = self.update_fc2(hidden) # No activation on final output of residual branch
            next_z_raw = curr_z + update
            
            # 3. Noise (FIXED BLOCK)
            if training:
                # If we have a batch of keys, we must vmap the noise generation
                if step_key_input.ndim > 0:
                    noise = jax.vmap(lambda k: jax.random.normal(k, (next_z_raw.shape[-1],), dtype=curr_z.dtype))(step_key_input)
                else:
                    noise = jax.random.normal(step_key_input, next_z_raw.shape, dtype=curr_z.dtype)
                
                next_z_raw = next_z_raw + (noise * 0.02)
            
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
        
        #Flip from (Steps, Batch) to (Batch, Steps)
        step_probs = jnp.swapaxes(step_probs, 0, 1)
        
        rem = 1.0 - final_prob
        w_out = w_out + (rem * self.decoder(final_z))
        w_z = w_z + (rem * final_z)
        
        return w_out.astype(jnp.float32), w_z.astype(jnp.float32), self.recog_fc(w_z).astype(jnp.float32), step_probs, predicted_steps

# --- 3. TRAINING LOOP ---
@nnx.jit(static_argnums=(3, 4))
def train_step(model, optimizer, subkeys, micro_batch, prediction_horizon, difficulty, baseline_error):
    loss_scale = 1000.0
    
    def loss_fn(model):
        graphdef, state = nnx.split(model)
        
        def scan_body(current_baseline, key):
            # Split key: one for Physics gen, one for Latent Noise
            phys_key, noise_key = jax.random.split(key)
            
            m = nnx.merge(graphdef, state)
            inputs, targets = jax.lax.stop_gradient(
                PhysicsWorld.generate_batch(phys_key, micro_batch, difficulty, prediction_horizon)
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
            G = 4
            expanded_inputs = jnp.repeat(adversarial_inputs, G, axis=0)
            expanded_targets = jnp.repeat(targets, G, axis=0)
            batch_keys = jax.random.split(noise_key, expanded_inputs.shape[0])
            
            # Run model on super-batch
            preds, _, recognition, step_probs, pred_steps = m(expanded_inputs, prediction_horizon, True, batch_keys)
            
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
step = 0  # Manual step counter for infinite loop

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
        difficulty = 0.0 # Reset curriculum to verify mastery from scratch
    print(f"✅ Loaded checkpoint weights | Resetting to Difficulty 0.000")

# --- RESET LOGGING ---
log_file = "training_log.csv"
if os.path.exists(log_file):
    os.remove(log_file) 

print("🔥 Compiling Kernels (This may take 30s)...")

# allow it to run until Mastery
while True:
    step += 1
    key, subkey = jax.random.split(key)
    subkeys = jax.random.split(subkey, ACCUM_STEPS)
    
    # Curriculum: Deeper, not wider
    prediction_horizon = int(5 + (difficulty * 35))
    
    loss_val, step_metrics, baseline_error = train_step(model, optimizer, subkeys, BATCH_SIZE, prediction_horizon, difficulty, baseline_error)
    
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
    difficulty = max(0.0, difficulty)

    # Update state
    integral_error += error
    prev_error = error
    
    # Reset integral if we swing too hard (Anti-windup)
    if abs(error) > 0.1: integral_error = 0.0

    if step % 50 == 0:
        sps = 50 / (time.time() - start_time + 1e-6)
        
        # Calculate how many particles are currently active
        # Based on: active_count = clip(2.0 + difficulty, ...)
        current_active_particles = 2.0 + difficulty
        
        print(f"Step {step} Diff: {difficulty:.3f} (N~{int(current_active_particles)}, Steps: {prediction_horizon}) | Loss: {avg_main_loss:.4f} | Base: {baseline_error:.4f} | PlanErr: {planner_err:.2f} | {sps:.1f} steps/s")
        start_time = time.time()
        
        # Save telemetry
        with open(log_file, "a") as f:
            f.write(f"{step},{difficulty:.4f},{float(loss_val):.4f},{avg_main_loss:.4f},{sps:.1f},{planner_err:.4f},{avg_steps:.2f}\n")
            
        if step % 1000 == 0:
            with open("physics_ckpt.pkl", "wb") as f:
                pickle.dump({'state': nnx.state(model), 'difficulty': float(difficulty)}, f)

        # --- STOPPING CONDITION: HUMAN MASTERY ---
        # 1. Capacity: Can it track 4 objects? (Human MOT limit)
        # 2. Horizon: Can it predict ~2 seconds ahead? (Human Intuition limit)
        #    (2.0 seconds / 0.03 dt = ~67 steps)
        
        is_human_capacity = (difficulty >= 2.0) 
        is_human_accuracy = (avg_main_loss < 0.05)
        
        if is_human_capacity and is_human_accuracy:
            print(f"\n🧠 HUMAN MASTERY ACHIEVED at Step {step}!")
            print(f"   - Capacity: {int(2.0 + difficulty)} Particles (Matched Human MOT)")
            print(f"   - Horizon: {prediction_horizon} Steps (Matched Human Intuition)")
            print("   - Stopping Training (Model is now 'Human-Equivalent').")
            
            # Save Final Model
            with open("physics_human_mastered.pkl", "wb") as f:
                pickle.dump({'state': nnx.state(model), 'difficulty': float(difficulty)}, f)
            break
