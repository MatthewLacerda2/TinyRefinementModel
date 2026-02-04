import os
import pickle
import json
import time
import jax
import jax.numpy as jnp
from flax import nnx
import optax

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# --- 1. THE PHYSICS GENERATOR (Infinite Gym) ---
class PhysicsWorld:
    """
    Procedurally generates physics problems of increasing difficulty.
    """
    @staticmethod
    def get_input_dim():
        return 15 

    @staticmethod
    def get_output_dim():
        return 6

    @staticmethod
    @jax.jit
    def generate_batch(key, batch_size, level):
        """
        Returns: 
           inputs: (B, 15) - The initial conditions (Problem)
           targets: (B, 6) - The ground truth future state (Solution)
        """
        def level_0_linear(k):
            # Level 0: 1D Motion (x_f = x_i + v*t)
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
            
            # Pack: [x, y, vx, vy, g, t, ...zeros...]
            feats = jnp.concatenate([x, y, vx, vy, g, t], axis=-1)
            inputs = jnp.concatenate([feats, jnp.zeros((batch_size, 9))], axis=-1)
            
            # Target: [final_x, final_y, ...zeros...]
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


# --- 2. THE REASONING MODEL (With Recognition Circuit) ---
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
        # 1. Encode
        z = self.encoder(raw_input) 
        z = nnx.gelu(z)
        
        # 2. Refine (Think)
        def refine_step(carry):
            current_z, step = carry
            step_feat = jnp.full((current_z.shape[0], 1), step, dtype=current_z.dtype)
            combined = jnp.concatenate([current_z, step_feat], axis=-1)
            
            h = nnx.gelu(self.fc1(combined))
            delta = self.fc2(h)
            next_z = self.norm(current_z + 0.1 * delta)
            return next_z, step + 1

        final_z, _ = jax.lax.scan(lambda c, _: (refine_step(c)[0], None), (z, 0), None, length=max_steps)
        
        # 3. Decode Answer
        prediction = self.decoder(final_z)
        
        recognition = self.recog_fc(final_z)
        
        return prediction, final_z, recognition


# --- 3. TRAINING INFRASTRUCTURE ---
@nnx.jit
def train_step(model, optimizer, subkeys, micro_batch, level_int):
    loss_scale = 1000.0
    
    def loss_fn(model):
        graphdef, state = nnx.split(model)

        def scan_body(carry, key):
            m = nnx.merge(graphdef, state)
            # Generate Task
            inputs, targets = PhysicsWorld.generate_batch(key, micro_batch, level_int)
            # Forward Pass
            preds, final_z, recognition = m(inputs)
            # A. Physics Loss (Did it get the right answer?)
            main_loss = jnp.mean((preds - targets) ** 2)
            # B. [NEW] Recognition Loss (Did it remember the question?)
            # We force the model to reconstruct the 'inputs' from its 'final_z'.
            # If this loss is high, the model has forgotten the mass/velocity!
            recog_loss = jnp.mean((recognition - inputs) ** 2)
            
            # C. Stability Loss
            stability = jnp.mean(final_z ** 2) * 1e-4
            
            # Combined Loss
            # We weight recognition at 0.5 to prioritize solving over memorizing
            loss = main_loss + (0.5 * recog_loss) + stability
            return None, loss

        _, losses = jax.lax.scan(scan_body, None, subkeys)
        return jnp.mean(losses)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    grads = jax.tree.map(lambda g: g / loss_scale, grads)
    optimizer.update(model, grads)
    return loss / loss_scale

# --- 4. MAIN EXECUTION ---
latent_dim = 768
accum_steps = 8
micro_batch = 64

model = RefineMathPhysics(latent_dim, nnx.Rngs(42))

def param_labels(params):
    def label(p):
        if p.ndim == 2:
            return "fast"
        return "slow"
    return jax.tree_util.tree_map(label, params)

tx = optax.adam(3e-4)
optimizer = nnx.Optimizer(model, tx)

print("🚀 Starting Physics Refinement (With Recognition Circuit)...")
key = jax.random.key(0)
loss_history = []
current_level = 0

for step in range(100000):
    key, subkey = jax.random.split(key)
    subkeys = jax.random.split(subkey, accum_steps)
    
    loss = train_step(model, optimizer, subkeys, micro_batch, current_level)
    loss_val = float(loss)
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
        print(f"Step {step} | Level {current_level} | Loss: {loss_val:.4f} | Avg: {avg_loss:.4f}")
        
        if step % 1000 == 0:
            with open("physics_ckpt.pkl", "wb") as f:
                pickle.dump({'state': nnx.state(model), 'level': current_level}, f)