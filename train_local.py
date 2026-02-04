import os
import pickle
import json
import time
import jax
import jax.numpy as jnp
from flax import nnx
import optax

# Keep these if they prevent OOM on your specific hardware
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# --- 1. THE PHYSICS GENERATOR (Infinite Gym) ---
class PhysicsWorld:
    @staticmethod
    def get_input_dim():
        return 15 

    @staticmethod
    def get_output_dim():
        return 6

    @staticmethod
    def generate_batch(key, batch_size, difficulty):
        # Helper to clamp difficulty for sub-levels
        d_val = jnp.clip(difficulty, 0.0, 3.0)
        
        def level_0_linear(k, intensity):
            # Intensity 0.0 -> 1.0 scales range and time
            scale = 1.0 + intensity * 9.0 # 1x to 10x range
            t_max = 1.0 + intensity * 4.0 # 1s to 5s
            
            k1, k2, k3 = jax.random.split(k, 3)
            x = jax.random.uniform(k1, (batch_size, 1), minval=-10*scale, maxval=10*scale)
            v = jax.random.uniform(k2, (batch_size, 1), minval=-5*scale, maxval=5*scale)
            t = jnp.zeros((batch_size, 1)) + jax.random.uniform(k3, (batch_size, 1), minval=1.0, maxval=t_max)
            
            target_x = x + v * t
            inputs = jnp.concatenate([x, v, t, jnp.zeros((batch_size, 12))], axis=-1)
            targets = jnp.concatenate([target_x, jnp.zeros((batch_size, 5))], axis=-1)
            return inputs, targets

        def level_1_projectile(k, intensity):
            # Intensity 0.0 -> 1.0 scales Gravity and Velocity complexity
            g_min = 1.0 + intensity * 4.0  # Start with weak gravity (moon-like)
            g_max = 5.0 + intensity * 15.0 # End with heavy gravity
            
            k1, k2, k3, k4, k5, k6 = jax.random.split(k, 6)
            x = jax.random.uniform(k1, (batch_size, 1), minval=-50, maxval=50)
            y = jax.random.uniform(k2, (batch_size, 1), minval=0, maxval=100)
            vx = jax.random.uniform(k3, (batch_size, 1), minval=-20, maxval=20)
            vy = jax.random.uniform(k4, (batch_size, 1), minval=-20, maxval=20)
            g = jax.random.uniform(k5, (batch_size, 1), minval=g_min, maxval=g_max)
            t = jax.random.uniform(k6, (batch_size, 1), minval=1, maxval=5)
            
            tf_x = x + vx * t
            tf_y = y + vy * t - 0.5 * g * t**2
            
            feats = jnp.concatenate([x, y, vx, vy, g, t], axis=-1)
            inputs = jnp.concatenate([feats, jnp.zeros((batch_size, 9))], axis=-1)
            t_feats = jnp.concatenate([tf_x, tf_y], axis=-1)
            targets = jnp.concatenate([t_feats, jnp.zeros((batch_size, 4))], axis=-1)
            return inputs, targets

        def level_2_nbody(k, intensity):
            # Intensity 0.0 -> 1.0 scales PREDICTION HORIZON
            # Start: Predict 1 step ahead (Easy vector math)
            # End: Predict 40 steps ahead (Chaotic simulation)
            steps_float = 1.0 + intensity * 39.0 
            steps = steps_float.astype(jnp.int32)
            
            N = 3
            dt = 0.05
            
            ks = jax.random.split(k, 4)
            pos = jax.random.uniform(ks[0], (batch_size, N, 2), minval=-2, maxval=2)
            vel = jax.random.normal(ks[1], (batch_size, N, 2)) * 0.3
            mass = jax.random.uniform(ks[2], (batch_size, N, 1), minval=0.5, maxval=2.0)
            
            def get_acc(p, m):
                diff = p[:, None, :, :] - p[:, :, None, :]
                dist_sq = jnp.sum(diff**2, axis=-1, keepdims=True) + 1e-2
                force = (diff / jnp.sqrt(dist_sq)) * (m[:, None, :, :] / dist_sq)
                return jnp.sum(force, axis=2)

            def sim_step(carry, _):
                p, v = carry
                a = get_acc(p, mass)
                return (p + v*dt, v + a*dt), None

            (final_p, final_v), _ = jax.lax.scan(sim_step, (pos, vel), None, length=steps)
            
            inputs = jnp.concatenate([pos.reshape(batch_size, -1), vel.reshape(batch_size, -1), mass.reshape(batch_size, -1)], axis=-1)
            targets = final_p.reshape(batch_size, -1)
            return inputs, targets

        # Logic to route float difficulty to specific generator
        # We use lax.cond or switch logic manually since we need to pass the 'remainder' intensity
        
        branch_0 = lambda k: level_0_linear(k, d_val)
        branch_1 = lambda k: level_1_projectile(k, d_val - 1.0)
        branch_2 = lambda k: level_2_nbody(k, d_val - 2.0)

        level_idx = d_val.astype(jnp.int32)
        
        return jax.lax.switch(level_idx, [branch_0, branch_1, branch_2], key)


# --- 2. THE REASONING MODEL ---
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

    def __call__(self, raw_input, max_steps=64): # Increased ceiling to 64
        z = self.encoder(raw_input) 
        z = nnx.gelu(z)
        
        # We need to track the answer at EVERY step to allow early exit
        def refine_step(carry, _):
            current_z, step_idx, running_halt_prob, weighted_out, weighted_z = carry
            
            # 1. Create Time Feature
            step_feat = jnp.full((current_z.shape[0], 1), step_idx, dtype=current_z.dtype)
            combined = jnp.concatenate([current_z, step_feat], axis=-1)
            
            # 2. Thinking Step
            h = nnx.gelu(self.fc1(combined))
            delta = self.fc2(h)
            next_z = self.norm(current_z + 0.1 * delta)
            
            # 3. Halt Decision (Should I stop?)
            # Output is probability (0 to 1)
            halt_val = nnx.sigmoid(self.halt_fc(next_z)) 
            
            # 4. Compute candidate answer NOW (in case we stop here)
            curr_pred = self.decoder(next_z)

            # --- PonderNet Logic ---
            # If we haven't halted yet, this step contributes to the final answer.
            # "p" is the probability we stop at THIS specific step.
            p = halt_val * (1.0 - running_halt_prob)
            
            # Accumulate the weighted answer
            new_weighted_out = weighted_out + (p * curr_pred)
            new_weighted_z = weighted_z + (p * next_z)
            new_running_prob = running_halt_prob + p
            
            return (next_z, step_idx + 1, new_running_prob, new_weighted_out, new_weighted_z), (p, halt_val)

        # Initialize Accumulators
        B = z.shape[0]
        out_dim = PhysicsWorld.get_output_dim()
        
        init_carry = (
            z,                                   # current_z
            0,                                   # step_idx
            jnp.zeros((B, 1)),                   # running_halt_prob
            jnp.zeros((B, out_dim)),             # weighted_out (FINAL ANSWER)
            jnp.zeros((B, self.latent_dim))      # weighted_z (For recog loss)
        )

        (final_z_state, _, final_prob, final_pred, final_z_weighted), (step_probs, halt_vals) = jax.lax.scan(
            refine_step,
            init_carry,
            None,
            length=max_steps
        )
        
        # Safety: If probability didn't sum to 1 (it ran out of steps),
        # force the remainder onto the very last step's answer.
        remainder = 1.0 - final_prob
        final_pred = final_pred + (remainder * self.decoder(final_z_state))
        final_z_weighted = final_z_weighted + (remainder * final_z_state)

        # We return 'step_probs' to penalize the model for thinking too long (optional)
        return final_pred, final_z_weighted, self.recog_fc(final_z_weighted), step_probs


# --- 3. OPTIMIZED TRAINING INFRASTRUCTURE ---

# NOTE: We set static_argnums=(3,) because 'micro_batch' (3rd arg) determines tensor shapes.
# 'difficulty' is NOT static, allowing lax.switch to handle level changes without recompiling.
@nnx.jit(static_argnums=(3,))
def train_step(model, optimizer, subkeys, micro_batch, difficulty):
    loss_scale = 1000.0
    
    def loss_fn(model):
        graphdef, state = nnx.split(model)

        def scan_body(carry, key):
            # Temporarily merge state to run forward pass
            m = nnx.merge(graphdef, state)
            inputs, targets = PhysicsWorld.generate_batch(key, micro_batch, difficulty)
            preds, final_z, recognition, step_probs = m(inputs)
            
            main_loss = jnp.mean((preds - targets) ** 2)
            recog_loss = jnp.mean((recognition - inputs) ** 2)
            stability = jnp.mean(final_z ** 2) * 1e-4

            # --- Ponder Loss (Geometric distribution regularization) ---
            # Encourage halting early (small penalty per step)
            _max_steps = step_probs.shape[0]
            avg_steps = jnp.sum(step_probs * jnp.arange(_max_steps)[:, None, None]) / micro_batch
            ponder_penalty = avg_steps * 0.01 
            
            # Combine loss
            loss = main_loss + (0.5 * recog_loss) + stability + ponder_penalty
            
            # SCALING: Multiply loss by scale here so gradients are large enough
            return None, loss * loss_scale

        # Run accumulation scan
        _, losses = jax.lax.scan(scan_body, None, subkeys)
        
        # Return average scaled loss
        return jnp.mean(losses)

    # Compute gradients
    loss_scaled, grads = nnx.value_and_grad(loss_fn)(model)
    
    # UNSCALING: Divide grads by scale to normalize
    grads = jax.tree.map(lambda g: g / loss_scale, grads)
    
    # Update model
    optimizer.update(model, grads)
    
    return loss_scaled / loss_scale


# --- 4. MAIN EXECUTION ---
latent_dim = 768
accum_steps = 8
micro_batch = 64

print("🚀 initializing model...")
model = RefineMathPhysics(latent_dim, nnx.Rngs(42))
tx = optax.adam(3e-4)
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

print("🚀 JIT Compiling... (First step will be slow)")
key = jax.random.key(0)
loss_history = []
current_level = 0

start_time = time.time()

for step in range(100000):
    key, subkey = jax.random.split(key)
    subkeys = jax.random.split(subkey, accum_steps)
    
    # This runs the JIT-compiled step
    loss = train_step(model, optimizer, subkeys, micro_batch, current_level)
    loss_val = float(loss) # Blocks until calculation is done
    
    loss_history.append(loss_val)
    if len(loss_history) > 100: loss_history.pop(0)
    avg_loss = sum(loss_history) / len(loss_history)
    
    # Auto Level Up
    if len(loss_history) == 100:
        if current_level == 0 and avg_loss < 0.1:
            current_level = 1
            loss_history = []
            print(f"\n🎉 LEVEL UP! Promoted to Level 1: Projectile Motion\n")
        elif current_level == 1 and avg_loss < 0.5:
            current_level = 2
            loss_history = []
            print(f"\n🎉 LEVEL UP! Promoted to Level 2: Three-Body Chaos\n")

    if step % 100 == 0:
        sps = 100 / (time.time() - start_time + 1e-6)
        print(f"Step {step} | Level {current_level} | Loss: {loss_val:.4f} | Avg: {avg_loss:.4f} | Speed: {sps:.2f} steps/s")
        start_time = time.time()
        
        # Save telemetry for plotting
        telemetry = {
            'step': step,
            'level': current_level,
            'loss': loss_val,
            'avg_loss': avg_loss,
            'speed': sps
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
                pickle.dump({'state': nnx.state(model), 'level': current_level}, f)