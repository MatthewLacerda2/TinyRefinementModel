import os
import pickle
import json
import time
import jax
import jax.numpy as jnp
from flax import nnx
import optax

MAX_N = 64
LATENT_DIM = 512
BATCH_SIZE = 128
ACCUM_STEPS = 2

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

class PhysicsWorld:
    @staticmethod
    def get_input_dim():
        return (MAX_N * 6) + 1 

    @staticmethod
    def get_output_dim():
        return MAX_N * 2

    @staticmethod
    def generate_batch(key, batch_size, difficulty, steps):
        mode_key, init_key, sim_key = jax.random.split(key, 3)
        
        mode = jax.random.bernoulli(mode_key, p=0.5, shape=(batch_size, 1)).astype(jnp.float32)
        
        active_count = jnp.clip(2.0 + difficulty, 2.0, MAX_N).astype(jnp.int32)
        
        pos_space = jax.random.uniform(init_key, (batch_size, MAX_N, 2), minval=-5, maxval=5)
        pos_earth = jax.random.uniform(init_key, (batch_size, MAX_N, 2), minval=-4, maxval=4)
        pos_earth = pos_earth.at[:, :, 1].add(3.0) 
        
        pos = jnp.where(mode[..., None], pos_earth, pos_space)

        vel = jax.random.normal(init_key, (batch_size, MAX_N, 2))
        vel = jnp.where(mode[..., None], vel * 0.5, vel * 1.5) 
        
        mass = jax.random.uniform(init_key, (batch_size, MAX_N, 1), minval=0.5, maxval=3.0)

        indices = jnp.arange(MAX_N)[None, :, None]
        mask = (indices < active_count).astype(jnp.float32)
        
        pos = pos * mask
        vel = vel * mask
        mass = mass * mask

        def get_acc(p, v, m, active_mask, current_mode):
            diff = p[:, :, None, :] - p[:, None, :, :]
            dist_sq = jnp.sum(diff**2, axis=-1, keepdims=True) + 1e-4
            dist = jnp.sqrt(dist_sq)
            
            repulse = 50.0 * jnp.exp(-dist_sq * 2.0)
            
            force_g = (2.0 * m[:, None, :, :] * m[:, :, None, :]) / dist_sq
            
            total_force = (diff / dist) * (force_g - repulse)
            
            valid = active_mask[:, :, None, :] * active_mask[:, None, :, :]
            eye = jnp.eye(MAX_N)[None, :, :, None]
            valid = valid * (1.0 - eye)
            
            interaction_acc = jnp.sum(total_force * valid, axis=2) / (m + 1e-6)
            
            gravity_earth = jnp.array([0.0, -9.8])[None, None, :]
            gravity_space = -0.5 * p
            
            env_acc = jnp.where(current_mode[..., None] > 0.5, gravity_earth, gravity_space)
            
            drag_factor = jnp.where(current_mode[..., None] > 0.5, 0.05, 0.0)
            drag_acc = -v * drag_factor
            
            return (interaction_acc + env_acc + drag_acc) * active_mask

        dt = 0.03

        def sim_step(carry, _):
            p, v = carry
            a = get_acc(p, v, mass, mask, mode)
            
            v_new = v + a * dt
            p_new = p + v_new * dt
            
            floor_y = -5.0
            hit_floor = (p_new[:, :, 1] < floor_y) & (mode > 0.5)
            
            v_new = v_new.at[:, :, 1].set(
                jnp.where(hit_floor, -v_new[:, :, 1] * 0.7, v_new[:, :, 1])
            )
            p_new = p_new.at[:, :, 1].set(
                jnp.where(hit_floor, floor_y + 0.01, p_new[:, :, 1])
            )
            
            v_new = v_new.at[:, :, 0].set(
                jnp.where(hit_floor, v_new[:, :, 0] * 0.9, v_new[:, :, 0])
            )

            v_new = jnp.where((p_new > 10) | (p_new < -10), -v_new * 0.8, v_new)
            p_new = jnp.clip(p_new, -10, 10)
            
            p_new = p_new * mask
            v_new = v_new * mask
            return (p_new, v_new), None

        (final_p, _), _ = jax.lax.scan(sim_step, (pos, vel), None, length=steps)

        mask_broadcasted = jnp.broadcast_to(mask, (batch_size, MAX_N, 1))
        
        flat_particles = jnp.concatenate([pos, vel, mass, mask_broadcasted], axis=-1).reshape(batch_size, -1)
        inputs = jnp.concatenate([flat_particles, mode], axis=-1)
        
        targets = final_p.reshape(batch_size, -1)
        
        return inputs, targets

class LatentReasoningBlock(nnx.Module):
    def __init__(self, latent_dim, num_heads, rngs, dtype=jnp.bfloat16):
        self.attn = nnx.MultiHeadAttention(num_heads=num_heads, in_features=latent_dim, dtype=dtype, rngs=rngs)
        self.norm1 = nnx.LayerNorm(latent_dim, dtype=dtype, rngs=rngs)
        self.fc = nnx.Linear(latent_dim, latent_dim, dtype=dtype, rngs=rngs)
        self.norm2 = nnx.LayerNorm(latent_dim, dtype=dtype, rngs=rngs)

    def __call__(self, z):
        z_norm = self.norm1(z)
        z_attn = self.attn(z_norm, z_norm)
        z = z + z_attn
        
        z_next = self.fc(self.norm2(z))
        z = z + nnx.gelu(z_next)
        return z

class RefineMathPhysics(nnx.Module):
    def __init__(self, latent_dim, rngs):
        self.latent_dim = latent_dim
        dtype = jnp.bfloat16
        
        self.encoder = nnx.Linear(PhysicsWorld.get_input_dim(), latent_dim, dtype=dtype, rngs=rngs)
        self.decoder = nnx.Linear(latent_dim, PhysicsWorld.get_output_dim(), dtype=dtype, rngs=rngs)
        self.recog_fc = nnx.Linear(latent_dim, PhysicsWorld.get_input_dim(), dtype=dtype, rngs=rngs)
        
        self.processor = LatentReasoningBlock(latent_dim, num_heads=4, rngs=rngs, dtype=dtype)
        
        self.norm = nnx.LayerNorm(latent_dim, dtype=dtype, rngs=rngs)
        self.halt_fc = nnx.Linear(latent_dim, 1, dtype=dtype, rngs=rngs)
        self.complexity_head = nnx.Linear(latent_dim, 1, dtype=dtype, rngs=rngs)

    def __call__(self, raw_input, max_steps=40, training=False, key=None):
        z = nnx.gelu(self.encoder(raw_input.astype(jnp.bfloat16)))
        
        batch_size = z.shape[0] if z.ndim > 1 else 1
        
        predicted_steps = nnx.sigmoid(self.complexity_head(z)) * max_steps
        
        if training and key is not None:
            is_key_batched = (key.ndim > 0) and (key.shape[0] == batch_size)
            
            if is_key_batched:
                step_keys = jax.vmap(lambda k: jax.random.split(k, max_steps))(key)
                step_keys = jnp.swapaxes(step_keys, 0, 1)
            else:
                step_keys = jax.random.split(key, max_steps)
        else:
            if z.ndim > 1:
                step_keys = jnp.zeros((max_steps, batch_size, 2), dtype=jnp.uint32)
            else:
                step_keys = jnp.zeros((max_steps, 2), dtype=jnp.uint32)

        def refine_step(carry, step_key_input):
            curr_z, step_idx, run_prob, w_out, w_z = carry
            
            z_seq = curr_z[:, None, :]
            next_z_raw = self.processor(z_seq).squeeze(1)
            
            if training:
                if step_key_input.ndim > 0:
                    noise = jax.vmap(lambda k: jax.random.normal(k, (next_z_raw.shape[-1],), dtype=curr_z.dtype))(step_key_input)
                else:
                    noise = jax.random.normal(step_key_input, next_z_raw.shape, dtype=curr_z.dtype)
                
                next_z_raw = next_z_raw + (noise * 0.02)
            
            next_z = self.norm(next_z_raw)
            
            halt = nnx.sigmoid(self.halt_fc(next_z))
            p = halt * (1.0 - run_prob)
            
            new_out = w_out + (p * self.decoder(next_z))
            new_z = w_z + (p * next_z)
            
            return (next_z, step_idx + 1, run_prob + p, new_out, new_z), p

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
        
        step_probs = jnp.swapaxes(step_probs, 0, 1)
        
        rem = 1.0 - final_prob
        w_out = w_out + (rem * self.decoder(final_z))
        w_z = w_z + (rem * final_z)
        
        return w_out.astype(jnp.float32), w_z.astype(jnp.float32), self.recog_fc(w_z).astype(jnp.float32), step_probs, predicted_steps

@nnx.jit(static_argnums=(3, 4))
def train_step(model, optimizer, subkeys, micro_batch, prediction_horizon, difficulty, baseline_error):
    loss_scale = 1000.0
    
    def loss_fn(model):
        graphdef, state = nnx.split(model)
        
        def scan_body(current_baseline, key):
            phys_key, noise_key = jax.random.split(key)
            
            m = nnx.merge(graphdef, state)
            inputs, targets = jax.lax.stop_gradient(
                PhysicsWorld.generate_batch(phys_key, micro_batch, difficulty, prediction_horizon)
            )
            
            def attack_loss_fn(input_perturbation):
                corrupted_inputs = inputs + input_perturbation
                preds, _, _, _, _ = m(corrupted_inputs, training=False, key=noise_key)
                return jnp.mean((preds - targets) ** 2)

            attack_grad = jax.grad(attack_loss_fn)(jnp.zeros_like(inputs))
            epsilon = 0.05 
            adversarial_inputs = inputs + (epsilon * jnp.sign(attack_grad))
            
            G = 4
            expanded_inputs = jnp.repeat(adversarial_inputs, G, axis=0)
            expanded_targets = jnp.repeat(targets, G, axis=0)
            batch_keys = jax.random.split(noise_key, expanded_inputs.shape[0])
            
            preds, _, recognition, step_probs, pred_steps = m(expanded_inputs, prediction_horizon, True, batch_keys)
            
            sq_err = jnp.mean((preds - expanded_targets) ** 2, axis=-1)
            
            dynamic_threshold = current_baseline * 0.95
            
            winners_mask = (sq_err < dynamic_threshold).astype(jnp.float32)
            
            num_winners = jnp.sum(winners_mask)
            effective_mask = jnp.where(num_winners > 0, winners_mask, jnp.ones_like(winners_mask))
            
            safe_denom = jnp.sum(effective_mask) + 1e-6
            main_loss = jnp.sum(sq_err * effective_mask) / safe_denom
            
            current_batch_avg = jnp.mean(sq_err)
            next_baseline = (0.99 * current_baseline) + (0.01 * current_batch_avg)
            
            recog_loss = jnp.mean((recognition - expanded_inputs) ** 2)
            
            steps_range = jnp.arange(step_probs.shape[1], dtype=jnp.float32)[None, :, None]
            actual_steps = jnp.sum(step_probs * steps_range, axis=1)
            
            diff = pred_steps - actual_steps
            planner_err_sq = jnp.where(diff < 0, 3.0 * (diff**2), diff**2)
            planner_loss = jnp.mean(planner_err_sq)
            
            avg_steps = jnp.mean(actual_steps)
            
            loss = main_loss + (0.5 * recog_loss) + (planner_loss * 0.1) + (avg_steps * 0.005)

            telemetry_loss = jnp.sum(sq_err * effective_mask) / (jnp.sum(effective_mask) + 1e-6)

            metrics = {
                'main_loss': jnp.mean(sq_err),
                'best_loss': telemetry_loss,
                'planner_err': jnp.mean(jnp.abs(diff)),
                'avg_steps': avg_steps,
                'baseline': current_baseline
            }
            return next_baseline, (loss * loss_scale, metrics)

        updated_baseline, (losses_with_scale, all_metrics) = jax.lax.scan(scan_body, baseline_error, subkeys)
        
        avg_metrics = jax.tree.map(jnp.mean, all_metrics)
        return jnp.mean(losses_with_scale), (avg_metrics, updated_baseline)

    (loss_s, (metrics, new_baseline)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    grads = jax.tree.map(lambda g: g / loss_scale, grads)
    optimizer.update(model, grads)
    
    total_loss = loss_s
    return total_loss / loss_scale, metrics, new_baseline

print(f"🚀 Initializing Infinite Physics (Max N={MAX_N})...")
model = RefineMathPhysics(LATENT_DIM, nnx.Rngs(42))
optimizer = nnx.Optimizer(model, optax.adam(3e-4), wrt=nnx.Param)

key = jax.random.key(0)
loss_history = []
difficulty = 0.0
start_time = time.time()
step = 0

integral_error = 0.0
prev_error = 0.0
kp, ki, kd = 0.005, 0.0001, 0.01

baseline_error = 0.1

ckpt_path = "physics_ckpt.pkl"
if os.path.exists(ckpt_path):
    print(f"📂 Loading checkpoint from {ckpt_path}...")
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
        nnx.update(model, ckpt['state'])
        difficulty = 0.0
    print(f"✅ Loaded checkpoint weights | Resetting to Difficulty 0.000")

log_file = "training_log.csv"
if os.path.exists(log_file):
    os.remove(log_file) 

print("🔥 Compiling Kernels (This may take 30s)...")
while True:
    step += 1
    key, subkey = jax.random.split(key)
    subkeys = jax.random.split(subkey, ACCUM_STEPS)
    
    prediction_horizon = int(5 + (difficulty * 35))
    
    loss_val, step_metrics, baseline_error = train_step(model, optimizer, subkeys, BATCH_SIZE, prediction_horizon, difficulty, baseline_error)
    
    loss_history.append(float(step_metrics['best_loss']))
    if len(loss_history) > 50: loss_history.pop(0)
    avg_main_loss = sum(loss_history) / len(loss_history)
    
    planner_err = float(step_metrics['planner_err'])
    avg_steps = float(step_metrics['avg_steps'])

    current_loss = avg_main_loss
    dynamic_target = 0.05 + (0.02 * difficulty) 
    error = dynamic_target - current_loss

    P = kp * error
    I = ki * integral_error
    D = kd * (error - prev_error)
    adjustment = P + I + D

    difficulty += adjustment
    difficulty = max(0.0, difficulty)

    integral_error += error
    prev_error = error
    
    if abs(error) > 0.1: integral_error = 0.0

    if step % 50 == 0:
        sps = 50 / (time.time() - start_time + 1e-6)
        
        current_active_particles = 2.0 + difficulty
        
        print(f"Step {step} Diff: {difficulty:.3f} (N~{int(current_active_particles)}, Steps: {prediction_horizon}) | Loss: {avg_main_loss:.4f} | Base: {baseline_error:.4f} | PlanErr: {planner_err:.2f} | {sps:.1f} steps/s")
        start_time = time.time()
        
        with open(log_file, "a") as f:
            f.write(f"{step},{difficulty:.4f},{float(loss_val):.4f},{avg_main_loss:.4f},{sps:.1f},{planner_err:.4f},{avg_steps:.2f}\n")
            
        if step % 1000 == 0:
            with open("physics_ckpt.pkl", "wb") as f:
                pickle.dump({'state': nnx.state(model), 'difficulty': float(difficulty)}, f)

        is_human_capacity = (difficulty >= 2.0) 
        is_human_accuracy = (avg_main_loss < 0.05)
        
        if is_human_capacity and is_human_accuracy:
            print(f"\n🧠 HUMAN MASTERY ACHIEVED at Step {step}!")
            print(f"   - Capacity: {int(2.0 + difficulty)} Particles (Matched Human MOT)")
            print(f"   - Horizon: {prediction_horizon} Steps (Matched Human Intuition)")
            print("   - Stopping Training (Model is now 'Human-Equivalent').")
            
            with open("physics_human_mastered.pkl", "wb") as f:
                pickle.dump({'state': nnx.state(model), 'difficulty': float(difficulty)}, f)
            break
