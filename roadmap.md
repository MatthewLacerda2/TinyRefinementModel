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

tell it the target
add latent velocity and denoising
make it read real math
