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
    def generate_batch(key, batch_size, level):
        # ... (Same as your original code) ...
        def level_0_linear(k):
            k1, k2, k3 = jax.random.split(k, 3)
            x = jax.random.uniform(k1, (batch_size, 1), minval=-10, maxval=10)
            v = jax.random.uniform(k2, (batch_size, 1), minval=-5, maxval=5)
            t = jax.random.uniform(k3, (batch_size, 1), minval=1, maxval=5)
            target_x = x + v * t
            inputs = jnp.concatenate([x, v, t, jnp.zeros((batch_size, 12))], axis=-1)
            targets = jnp.concatenate([target_x, jnp.zeros((batch_size, 5))], axis=-1)
            return inputs, targets

        def level_1_projectile(k):
            k1, k2, k3, k4, k5, k6 = jax.random.split(k, 6)
            x = jax.random.uniform(k1, (batch_size, 1), minval=-50, maxval=50)
            y = jax.random.uniform(k2, (batch_size, 1), minval=0, maxval=100)
            vx = jax.random.uniform(k3, (batch_size, 1), minval=-20, maxval=20)
            vy = jax.random.uniform(k4, (batch_size, 1), minval=-20, maxval=20)
            g = jax.random.uniform(k5, (batch_size, 1), minval=5, maxval=15)
            t = jax.random.uniform(k6, (batch_size, 1), minval=1, maxval=5)
            
            tf_x = x + vx * t
            tf_y = y + vy * t - 0.5 * g * t**2
            
            feats = jnp.concatenate([x, y, vx, vy, g, t], axis=-1)
            inputs = jnp.concatenate([feats, jnp.zeros((batch_size, 9))], axis=-1)
            t_feats = jnp.concatenate([tf_x, tf_y], axis=-1)
            targets = jnp.concatenate([t_feats, jnp.zeros((batch_size, 4))], axis=-1)
            return inputs, targets

        def level_2_nbody(k):
            N = 3
            dt = 0.05
            steps = 40
            
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

        return jax.lax.switch(level, [level_0_linear, level_1_projectile, level_2_nbody], key)


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

    def __call__(self, raw_input, max_steps=40):
        z = self.encoder(raw_input) 
        z = nnx.gelu(z)
        
        def refine_step(carry):
            current_z, step = carry
            step_feat = jnp.full((current_z.shape[0], 1), step, dtype=current_z.dtype)
            combined = jnp.concatenate([current_z, step_feat], axis=-1)
            
            h = nnx.gelu(self.fc1(combined))
            delta = self.fc2(h)
            next_z = self.norm(current_z + 0.1 * delta)
            return next_z, step + 1

        (final_z, _), _ = jax.lax.scan(
            lambda c, _: (refine_step(c), None),
            (z, 0),
            None,
            length=max_steps
        )
        
        prediction = self.decoder(final_z)
        recognition = self.recog_fc(final_z)
        return prediction, final_z, recognition


# --- 3. OPTIMIZED TRAINING INFRASTRUCTURE ---

# NOTE: We set static_argnums=(3,) because 'micro_batch' (3rd arg) determines tensor shapes.
# 'level_int' is NOT static, allowing lax.switch to handle level changes without recompiling.
@nnx.jit(static_argnums=(3,))
def train_step(model, optimizer, subkeys, micro_batch, level_int):
    loss_scale = 1000.0
    
    def loss_fn(model):
        graphdef, state = nnx.split(model)

        def scan_body(carry, key):
            # Temporarily merge state to run forward pass
            m = nnx.merge(graphdef, state)
            inputs, targets = PhysicsWorld.generate_batch(key, micro_batch, level_int)
            preds, final_z, recognition = m(inputs)
            
            main_loss = jnp.mean((preds - targets) ** 2)
            recog_loss = jnp.mean((recognition - inputs) ** 2)
            stability = jnp.mean(final_z ** 2) * 1e-4
            
            # Combine loss
            loss = main_loss + (0.5 * recog_loss) + stability
            
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
        
        if step % 1000 == 0 and step > 0:
            with open("physics_ckpt.pkl", "wb") as f:
                pickle.dump({'state': nnx.state(model), 'level': current_level}, f)