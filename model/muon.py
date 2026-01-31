import jax.numpy as jnp

def newton_schulz_iteration(A, iterations=5):
    """
    Newton-Schulz iteration to orthogonalize a matrix or stabilize internal latent states.
    Used as the 'Latent Reality Check' mentioned in the README.
    """
    X = A
    for _ in range(iterations):
        XTX = jnp.dot(X.T, X)
        I = jnp.eye(XTX.shape[0])
        X = 0.5 * jnp.dot(X, 3.0 * I - XTX)
    return X
