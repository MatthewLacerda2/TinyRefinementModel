#!/bin/bash
# -----------------------------------------------------------------------------
# THE BEAST: TPU Environment Setup (2026 Edition)
# -----------------------------------------------------------------------------

# 1. System-level optimizations for JAX/TPU v5+
# Huge pages reduce memory fragmentation, vital for recursive loops
sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"

# 2. Update Python toolchain (avoiding system-python breakage)
pip install --upgrade pip setuptools wheel

# 3. Install the JAX-AI Stack (Fused Kernels + State Management + Checkpointing)
# - jax[tpu]: The core compiler + TPU runtime
# - flax: For the NNX stateful modules
# - optax: For the AdamW/Muon optimization
# - orbax-checkpoint: For the multi-device saving logic
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax optax orbax-checkpoint sympy

# 4. GCS FUSE Setup (Mounting your bucket like a local drive)
# This lets Orbax save directly to Google Cloud Storage
sudo apt-get update && sudo apt-get install -y gcsfuse
mkdir -p ~/checkpoints
# Replace with your bucket name from Terraform
gcsfuse --implicit-dirs recursive-beast-checkpoints ~/checkpoints

# 5. Clone the Engine
# (Assuming your repo is private, you'd use a Deploy Key here)
git clone https://github.com/your-username/recursive-math-beast.git
cd recursive-math-beast

# 6. Verify Hardware
python3 -c "import jax; print(f'TPU Cores Found: {jax.device_count()}')"

echo "Setup complete. The Beast is ready to think."