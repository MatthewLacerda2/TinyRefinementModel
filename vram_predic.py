import re
import os

# Helper to pull constants from train_local.py without importing (avoids dependency hell)
def get_config():
    config = {}
    path = os.path.join(os.path.dirname(__file__), "train_local.py")
    if not os.path.exists(path):
        return None
    
    with open(path, "r") as f:
        content = f.read()
    
    keys = ["BATCH_SIZE", "MAX_STEPS_LIMIT", "LATENT_DIM", "MAX_SEQ_LEN", "VOCAB_SIZE", "SCRATCH_SLOTS"]
    for key in keys:
        match = re.search(rf"^{key}\s*=\s*(.+)$", content, re.MULTILINE)
        if match:
            # Strip comments and eval the value
            val = match.group(1).split('#')[0].strip()
            config[key] = eval(val)
    return config

DTYPE_SIZE = 4  # float32

def estimate_vram():
    cfg = get_config()
    if not cfg:
        print("❌ Error: Could not find train_local.py to read configuration.")
        return

    # Extract values for readability
    BATCH_SIZE = cfg["BATCH_SIZE"]
    MAX_STEPS_LIMIT = cfg["MAX_STEPS_LIMIT"]
    LATENT_DIM = cfg["LATENT_DIM"]
    MAX_SEQ_LEN = cfg["MAX_SEQ_LEN"]
    VOCAB_SIZE = cfg["VOCAB_SIZE"]
    SCRATCH_SLOTS = cfg["SCRATCH_SLOTS"]
    print("--- 🧠 Refined VRAM Estimation for UniversalReasoner (Linux/CUDA) ---")

    embed_params = VOCAB_SIZE * LATENT_DIM
    time_params = (MAX_STEPS_LIMIT + 1) * LATENT_DIM
    scratch_params = SCRATCH_SLOTS * LATENT_DIM
    
    head_dim = LATENT_DIM // 8
    num_groups = 2
    attn_params = (LATENT_DIM * LATENT_DIM) + (LATENT_DIM * num_groups * head_dim) * 2 + (LATENT_DIM * LATENT_DIM)
    
    mlp_params = (LATENT_DIM * LATENT_DIM * 4) + (LATENT_DIM * 4 * LATENT_DIM)
    norm_params = 4 * LATENT_DIM
    
    processor_params = attn_params + mlp_params + norm_params
    
    heads_params = (LATENT_DIM * 1) + (LATENT_DIM * 1) + (LATENT_DIM * VOCAB_SIZE)
    
    total_params = embed_params + time_params + scratch_params + processor_params + heads_params
    
    param_memory_gb = (total_params * DTYPE_SIZE) / (1024**3)
    
    opt_memory_gb = (total_params * DTYPE_SIZE * 2) / (1024**3)
    
    grad_memory_gb = (total_params * DTYPE_SIZE) / (1024**3)
    
    total_len = MAX_SEQ_LEN + SCRATCH_SLOTS
    
    scan_activations = MAX_STEPS_LIMIT * BATCH_SIZE * total_len * LATENT_DIM
    
    logit_activations = BATCH_SIZE * (MAX_SEQ_LEN - 1) * VOCAB_SIZE
    
    attn_maps_step = BATCH_SIZE * 8 * (total_len**2)
    
    activation_memory_gb = ((scan_activations + logit_activations + attn_maps_step) * DTYPE_SIZE) / (1024**3)

    static_mem = param_memory_gb + opt_memory_gb + grad_memory_gb
    total_mem = static_mem + activation_memory_gb
    
    context_overhead = 0.8
    
    print(f"Total Parameters:      {total_params / 1e6:.2f}M")
    print(f"---")
    print(f"Static (Weights):      {param_memory_gb:.4f} GB")
    print(f"Optimizer (AdamW):     {opt_memory_gb:.4f} GB")
    print(f"Gradients:             {grad_memory_gb:.4f} GB")
    print(f"Scan Activations:      {(scan_activations * DTYPE_SIZE)/(1024**2):.2f} MB")
    print(f"Decoder Logits:        {(logit_activations * DTYPE_SIZE)/(1024**2):.2f} MB")
    print(f"Total Activations:     {activation_memory_gb:.4f} GB")
    print(f"XLA/CUDA Context:      ~{context_overhead} GB")
    print(f"---")
    
    final_est = total_mem + context_overhead
    print(f"🚀 Estimated Total VRAM: {final_est:.2f} GB")
    
    gpu_limit = 6.0 # RTX 2060 Limit
    usage_pct = (final_est / gpu_limit) * 100
    
    print(f"\n📊 GPU Utilization (RTX 2060 6GB): {usage_pct:.1f}%")
    
    if final_est > gpu_limit:
        print(f"🚨 DANGER: Predicted {final_est:.2f}GB exceeds your 6GB limit! You WILL get OOM.")
        print("💡 Suggestion: Reduce BATCH_SIZE or MAX_SEQ_LEN.")
    elif final_est > 5.2:
        print("⚠️ Warning: Very close to 6GB limit. External screens or apps might cause OOM.")
    else:
        print("✅ Safe: You should have enough headroom on your RTX 2060.")

if __name__ == "__main__":
    estimate_vram()