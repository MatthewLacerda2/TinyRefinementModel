import jax
import jax.numpy as jnp

@jax.jit
def apply_muon_stabilization(A, iterations=5):
    """
    Stabilized Newton-Schulz iteration for JAX.
    Ensures the matrix converges to the nearest orthogonal manifold.
    """
    # 1. Spectral Scaling (Essential for convergence safety)
    # We use the Frobenius norm as a fast proxy for spectral scaling
    scale = jnp.linalg.norm(A) + 1e-7
    X = A / scale
    
    # 2. Iterative Refinement
    # We use a 3rd-order polynomial: X = 0.5 * X * (3I - X^T X)
    for _ in range(iterations):
        # We use @ for matmul to allow XLA to fuse operations
        XTX = X.T @ X
        I = jnp.eye(X.shape[1])
        X = 0.5 * X @ (3.0 * I - XTX)
    
    return X