import random
import sympy as sp
import jax.numpy as jnp

class MathGym:
    """Generates infinite algebra puzzles for the model to 'think' about."""
    
    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def generate_linear_equation(self):
        """Generates a random equation: ax + b = c"""
        x = sp.Symbol('x')
        a = self.rng.randint(1, 10)
        b = self.rng.randint(-20, 20)
        # Ensure c is chosen so x is an integer (easier for early training)
        solution = self.rng.randint(-50, 50)
        c = a * solution + b
        
        equation_str = f"{a}x + ({b}) = {c}"
        return equation_str, float(solution)

    def generate_batch(self, batch_size):
        """Prepares a batch of problems for the JAX trainer."""
        problems = []
        targets = []
        
        for _ in range(batch_size):
            prob_str, target = self.generate_linear_equation()
            # Convert string to fixed-length token IDs (simplified for this chunk)
            problems.append(self._tokenize(prob_str))
            targets.append(target)
            
        return {
            'input': jnp.array(problems),
            'target': jnp.array(targets)
        }

    def _tokenize(self, text, max_len=32):
        """Mock tokenizer: converts chars to IDs."""
        tokens = [ord(c) for c in text[:max_len]]
        return tokens + [0] * (max_len - len(tokens))

def reward_fn(prediction, target, threshold=0.01):
    """
    The Verifier: This is the 'Truth' that GRPO uses.
    Returns 1.0 if the model's converged thought decoded to the right number.
    """
    error = jnp.abs(prediction - target)
    return jnp.where(error < threshold, 1.0, 0.0)