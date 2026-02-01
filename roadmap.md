The right order (this matters)

Stable dynamics ← you’re here

Generalization pressure (vary z₀ / targets)

Multi-trajectory comparison (proto-GRPO)

Only then: Muon / orthogonalization

- - -

# 2. THE "MUON-LITE" CONSTRAINT
    # We force the weight matrix (W) to be close to orthogonal: W.T @ W ≈ I
    W = model.refine_layer.kernel
    # Identity matrix for comparison
    I = jnp.eye(W.shape[1])
    # Penalty: how much does W.T @ W deviate from Identity?
    ortho_loss = jnp.mean((jnp.dot(W.T, W) - I) ** 2)

    # Return combined loss with a small weight for orthogonality
    return base_loss + 0.01 * ortho_loss

- - - -

from jax import checkpoint

def train_step(model, optimizer, metrics, batch):
    def loss_fn(model):
        z0 = jnp.ones((8, 512)) * 0.01

        # The 'Magic' for Novelty: Rematerialization
        # This keeps VRAM flat even if you do 1000 steps
        @checkpoint
        def recursive_step(z, _):
            return model(z), None

        # Increase length to 64 or 128—now you're 'Thinking' deeper than TRM
        zs, _ = jax.lax.scan(recursive_step, z0, None, length=64)

        # Standard loss logic follows...
        loss_A = jnp.mean((zs[-1] - batch["target"]) ** 2)
        return loss_A

    # ... rest of your train_step