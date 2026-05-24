import csv
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import argparse
from train_local import (
    BATCH_SIZE, 
    MAX_SEQ_LEN, 
    LATENT_DIM, 
    NUM_BLOCKS, 
    VOCAB_SIZE, 
    SHARED_SLOTS, 
    MAX_STEPS_LIMIT,
    NUM_HEADS,
    NUM_GROUPS
)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def smooth(y, box_pts):
    if len(y) < box_pts:
        return np.array(y)
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    
    pad_front = box_pts // 2
    pad_back = len(y) - len(y_smooth) - pad_front
    return np.pad(y_smooth, (pad_front, pad_back), mode='edge')

def calculate_tokens(step):
    return step * BATCH_SIZE * (MAX_SEQ_LEN * 2)

def print_model_stats():
    print("Calculating model parameters & memory footprint...")
    
    head_dim = LATENT_DIM // NUM_HEADS
    
    # --- Block Parameters ---
    # Rotary Attention
    # q, k, v projs are f16, o_proj is f16. Norms are f32.
    q_params = LATENT_DIM * LATENT_DIM + LATENT_DIM
    k_params = LATENT_DIM * (NUM_GROUPS * head_dim) + (NUM_GROUPS * head_dim)
    v_params = LATENT_DIM * (NUM_GROUPS * head_dim) + (NUM_GROUPS * head_dim)
    o_params = LATENT_DIM * LATENT_DIM + LATENT_DIM
    attn_norms = head_dim * 2 # q_norm, k_norm
    
    # MLP (gate, up are f16, down is f16. Norms are f32)
    hidden_dim = int(256 * ((LATENT_DIM * 8 / 3 + 255) // 256))
    gate_params = LATENT_DIM * hidden_dim + hidden_dim
    up_params = LATENT_DIM * hidden_dim + hidden_dim
    down_params = hidden_dim * LATENT_DIM + LATENT_DIM
    
    block_norms = LATENT_DIM * 2 # Standard norm1, norm2
    
    # Pointers to specific precisions (Bytes)
    f16_per_block = (q_params + k_params + v_params + o_params + 
                     gate_params + up_params + down_params)
    f32_per_block = attn_norms + block_norms
    
    total_params_per_block = f16_per_block + f32_per_block
    
    # Unique Physical Blocks
    num_enc_blocks = 1 # Shared
    num_dec_blocks = 1 # Shared
    num_reasoning_blocks = NUM_BLOCKS
    unique_blocks = num_enc_blocks + num_dec_blocks + num_reasoning_blocks
    
    total_layer_params = total_params_per_block * unique_blocks
    total_layer_bytes = (f16_per_block * 2 + f32_per_block * 4) * unique_blocks
    
    # --- Structural Parameters ---
    embed = VOCAB_SIZE * LATENT_DIM
    time_embed = (MAX_STEPS_LIMIT + 1) * LATENT_DIM
    shared_token = SHARED_SLOTS * LATENT_DIM
    seq_norm = LATENT_DIM
    meta_proj = 2 * LATENT_DIM + LATENT_DIM
    extra_norms = LATENT_DIM * 4 # time, forget, signal, hunch
    hunch_gate = LATENT_DIM * LATENT_DIM + LATENT_DIM
    tau_param = 1
    forget_head = LATENT_DIM * LATENT_DIM + LATENT_DIM
    
    # Most structural params are f32 by default in UniversalReasoner
    structure_params = (embed + time_embed + shared_token + seq_norm + meta_proj + 
                        extra_norms + hunch_gate + tau_param + forget_head)
    structure_bytes = structure_params * 4
    
    total_params = total_layer_params + structure_params
    total_weight_bytes = total_layer_bytes + structure_bytes
    
    # Optimizer State (AdamW: 2 f32 moments per parameter)
    optimizer_bytes = total_params * 4 * 2
    
    # Gradients (f32)
    gradient_bytes = total_params * 4
    
    # Activation Memory (Highly heuristic estimation)
    # z_seq + reasoning states + attn scores for a single path
    # (Batch * Seq * Dim * Blocks) + (Batch * Steps * Slots * Dim)
    act_bytes_est = (BATCH_SIZE * MAX_SEQ_LEN * LATENT_DIM * 2 * NUM_BLOCKS * 2) # f16
    act_bytes_est += (BATCH_SIZE * MAX_STEPS_LIMIT * SHARED_SLOTS * LATENT_DIM * 2) # f16
    
    total_vram_gb = (total_weight_bytes + optimizer_bytes + gradient_bytes + act_bytes_est) / (1024**3)

    print(f"Model Parameters: {total_params:,}")
    print(f"  |-- Structure Params  : {structure_params:,}")
    print(f"  |-- Unique Layer Params : {total_layer_params:,} ({unique_blocks} unique blocks)")
    print(f"      |-- Shared Encoder : {total_params_per_block:,}")
    print(f"      |-- Shared Decoder : {total_params_per_block:,}")
    print(f"      |-- Reasoning Stack: {total_params_per_block * NUM_BLOCKS:,} ({NUM_BLOCKS} Blocks)")
    print("-" * 50)
    print(f"ESTIMATED VRAM FOOTPRINT (Training)")
    print(f"  |-- Weights (f16/f32) : {total_weight_bytes / (1024**2):.2f} MB")
    print(f"  |-- Optimizer (AdamW) : {optimizer_bytes / (1024**2):.2f} MB")
    print(f"  |-- Gradients (f32)   : {gradient_bytes / (1024**2):.2f} MB")
    print(f"  |-- Activations (Est) : {act_bytes_est / (1024**2):.2f} MB")
    print(f"  => TOTAL ESTIMATED   : {total_vram_gb:.2f} GB")

def plot_training_history(log_path="training_history.csv"):
    if not os.path.exists(log_path):
        print(f"❌ Error: {log_path} not found.")
        print("Note: The CSV is now stored in your CHECKPOINT_ROOT (default is ./).")
        return

    history = []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    history.append({
                        'step': int(row['step']),
                        'loss':  float(row.get('loss', 0)),
                        'ce': float(row.get('ce', 0)),
                        'first_ce': float(row.get('first_ce', 0)),
                        'diversity': float(row.get('diversity_loss', 0)),
                        'forget_cost': float(row.get('avg_forget_cost', 0)),
                        'grad_norm': float(row.get('grad_norm_avg', 0)),
                        'temporal_drift': float(row.get('temporal_drift', 0)),
                    })
                except ValueError:
                    continue 
    except Exception as e:
        print(f"❌ Error reading {log_path}: {e}")
        return

    if len(history) == 0:
        print("ℹ️ Warning: No valid training data found in CSV.")
        return

    steps = np.array([e['step'] for e in history])
    losses = np.array([e['loss'] for e in history])
    ce = np.array([e['ce'] for e in history])
    first_ce = np.array([e['first_ce'] for e in history])
    diversity = np.array([e['diversity'] for e in history])
    forget = np.array([e['forget_cost'] for e in history])
    grad_norm = np.array([e['grad_norm'] for e in history])
    temporal_drift = np.array([e['temporal_drift'] for e in history])

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ((ax1, ax2), (ax3, ax4)) = axes

    smoothing_window = max(5, min(100, len(steps) // 20))

    # --- 1. CONVERGENCE (CE vs First Step) ---
    ax1.plot(steps, first_ce, color='#ff007b', alpha=0.2, linestyle='--', label='Initial Guess (Step 0)')
    ax1.plot(steps, ce, color='#ff007b', alpha=0.2, label='Final Prediction')
    ax1.plot(steps, smooth(ce, smoothing_window), color='#ff007b', linewidth=2, label='Smoothed Final CE')
    ax1.fill_between(steps, smooth(ce, smoothing_window), smooth(first_ce, smoothing_window), 
                     color='#ff007b', alpha=0.1, label='Refinement Gain')
    ax1.set_title('Convergence & Refinement', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cross Entropy')
    if np.all(ce > 0): ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.1)

    # --- 2. REASONING INTENSITY (Temporal Drift) ---
    ax2.plot(steps, temporal_drift, color='#adff2f', alpha=0.4, label='Drift (Raw)')
    ax2.plot(steps, smooth(temporal_drift, smoothing_window), color='#adff2f', linewidth=2, label='Smoothed Drift')
    ax2.set_title('Latent State Temporal Drift', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Avg Distance per Step', color='#adff2f')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.1)

    # --- 3. RESOURCE COSTS (Forget & Storage) ---
    ax3.plot(steps, smooth(forget, smoothing_window), color='#00ff88', linewidth=2, label='Forget Cost')
    ax3.set_title('Dynamic Resource Penalties', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Cost Value')
    ax3.legend()
    ax3.grid(True, alpha=0.1)

    # --- 4. MODEL HEALTH (Grad Norm & Diversity) ---
    ax4.plot(steps, smooth(grad_norm, smoothing_window), color='#ffffff', linewidth=2, label='Grad Norm')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(steps, smooth(diversity, smoothing_window), color='#ff00ff', linewidth=1, label='Diversity Loss')
    ax4_twin.set_ylabel('Diversity Loss', color='#ff00ff')
    ax4.set_title('Optimization Health', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Gradient Norm')
    if np.any(grad_norm > 0): ax4.set_yscale('log')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.1)

    plt.tight_layout()
    output_image = 'reasoning_analytics.png'
    plt.savefig(output_image, dpi=150)
    print(f"✨ Analytics updated: {output_image}")
    
    # Post-Training Summary
    print("-" * 50)
    print("📊 POST-TRAINING SUMMARY")
    print("-" * 50)
    
    total_tokens = calculate_tokens(steps[-1]) if len(steps) > 0 else 0
    print(f"Total Tokens Trained     : {total_tokens:,}")
    
    if len(ce) > 1:
        print(f"Recent CE Change         : {ce[-1] - ce[-2]:.5f}")
    
    print(f"Lowest CE Observed       : {np.min(ce):.5f}")
    
    valid_ce = ce[~np.isnan(ce)]
    if len(valid_ce) > 0:
        window_size = min(100, len(valid_ce))
        print(f"Final 100-step Avg CE    : {np.mean(valid_ce[-window_size:]):.5f}")

    print("-" * 50)
    print_model_stats()
    print("-" * 50)

    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default="training_history.csv", 
                        help="Path to training_history.csv")
    args = parser.parse_args()
    
    plot_training_history(log_path=args.log)