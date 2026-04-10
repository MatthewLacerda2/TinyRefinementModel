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
    ACCUMULATION_STEPS,
    NUM_HEADS,
    NUM_GROUPS
)

# Ensure UTF-8 encoding for console output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def smooth(y, box_pts):
    """Simple moving average smoothing."""
    if len(y) < box_pts:
        return np.array(y)
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    
    # Pad so original shape is maintained
    pad_front = box_pts // 2
    pad_back = len(y) - len(y_smooth) - pad_front
    return np.pad(y_smooth, (pad_front, pad_back), mode='edge')

def calculate_tokens(step):
    """
    Calculate total tokens seen across all batches and accumulation steps.
    Micro-batches accumulated per step = ACCUMULATION_STEPS
    """
    return step * BATCH_SIZE * MAX_SEQ_LEN * ACCUMULATION_STEPS

def print_model_stats():
    print("Calculating model parameters (mathematical estimation)...")
    
    head_dim = LATENT_DIM // NUM_HEADS
    
    # 1. Block Stack (Per Block)
    q_params = LATENT_DIM * LATENT_DIM + LATENT_DIM
    k_params = LATENT_DIM * (NUM_GROUPS * head_dim) + (NUM_GROUPS * head_dim)
    v_params = LATENT_DIM * (NUM_GROUPS * head_dim) + (NUM_GROUPS * head_dim)
    o_params = LATENT_DIM * LATENT_DIM + LATENT_DIM
    attn_total = q_params + k_params + v_params + o_params
    
    hidden_dim = int(256 * ((LATENT_DIM * 8 / 3 + 255) // 256))
    gate_params = LATENT_DIM * hidden_dim + hidden_dim
    up_params = LATENT_DIM * hidden_dim + hidden_dim
    down_params = hidden_dim * LATENT_DIM + LATENT_DIM
    mlp_total = gate_params + up_params + down_params
    
    block_norms = LATENT_DIM * 2
    params_per_block = attn_total + mlp_total + block_norms
    num_reason_param = params_per_block * NUM_BLOCKS
    
    # 2. Universal Reasoner Extras
    embed = VOCAB_SIZE * LATENT_DIM
    time_embed = (MAX_STEPS_LIMIT + 1) * LATENT_DIM
    shared_token = SHARED_SLOTS * LATENT_DIM
    seq_norm = LATENT_DIM
    
    halt_pre_dim = LATENT_DIM // 4
    halt_pre = LATENT_DIM * halt_pre_dim + halt_pre_dim
    halt_head = halt_pre_dim * 1 + 1
    
    extra_norms = LATENT_DIM * 4 
    hunch_gate = LATENT_DIM * LATENT_DIM + LATENT_DIM
    forget_head = LATENT_DIM * LATENT_DIM + LATENT_DIM 
    
    encoder_params = (embed + time_embed + shared_token + seq_norm + 
                      halt_pre + halt_head + extra_norms + hunch_gate + forget_head)
    
    param_count = num_reason_param + encoder_params
    
    print(f"Model Parameters: {param_count:,}")
    print(f"  |-- Encoder Params : {encoder_params:,}")
    print(f"  |-- Layer Params   : {num_reason_param:,} (across {NUM_BLOCKS} blocks)")

def plot_training_history(log_path="orbax_checkpoints/training_history.csv"):
    if not os.path.exists(log_path):
        print(f"❌ Error: {log_path} not found.")
        print("Note: The CSV is now stored in your CHECKPOINT_ROOT (default is orbax_checkpoints/).")
        print("If you changed CHECKPOINT_ROOT in your .env, use: python plot_history.py --log YOUR_PATH/training_history.csv")
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
                        'avg_ponder': float(row.get('avg_ponder', 0)),
                        'forget_density': float(row.get('forget_density', 0)),
                        'grad_norm_avg': float(row.get('grad_norm_avg', 0)) if row.get('grad_norm_avg') else 0.0,
                    })
                except ValueError:
                    continue # Skip row if it has malformed floats
    except Exception as e:
        print(f"❌ Error reading {log_path}: {e}")
        return

    if len(history) == 0:
        print("ℹ️ Warning: No valid training data found in CSV.")
        return

    steps = np.array([e['step'] for e in history])
    losses = np.array([e['loss'] for e in history])
    ce = np.array([e['ce'] for e in history])
    ponder = np.array([e['avg_ponder'] for e in history])
    forget = np.array([e['forget_density'] for e in history])
    grad_norm = np.array([e['grad_norm_avg'] for e in history])

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ((ax1, ax2), (ax3, ax4)) = axes

    # Pick a smoothing window that doesn't smooth too much at the start
    smoothing_window = max(5, min(100, len(steps) // 20))

    # --- 1. CONVERGENCE (Cross Entropy) ---
    ax1.plot(steps, ce, color='#ff007b', alpha=0.3, label='Raw CE')
    ax1.plot(steps, smooth(ce, smoothing_window), color='#ff007b', linewidth=2, label=f'Smoothed CE (w={smoothing_window})')
    ax1.plot(steps, smooth(losses, smoothing_window), color='#ffcc00', linewidth=1.5, linestyle='--', alpha=0.7, label='Agg Loss (Smoothed)')
    ax1.set_title('Model Convergence', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cross Entropy Loss')
    # Use log scale only if CE > 0 strictly
    if np.all(ce > 0):
        ax1.set_yscale('log')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2)

    # --- 2. REASONING EFFICIENCY (Ponder Dynamics) ---
    ax2.plot(steps, ponder, color='#adff2f', alpha=0.4, label='Avg Pondering Steps')
    ax2.plot(steps, smooth(ponder, smoothing_window), color='#adff2f', linewidth=2, label='Smoothed Ponder')
    ax2.set_title('Reasoning Depth (Ponder Dynamics)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Avg Steps taken per Sequence')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.2)

    # --- 3. MEMORY MANAGEMENT (Forgetting) ---
    ax3.plot(steps, forget, color='#00ff88', alpha=0.4, label='Forget Density (Active Pruning)')
    ax3.plot(steps, smooth(forget, smoothing_window), color='#00ff88', linewidth=2, label='Smoothed Forget')
    ax3.set_title('Memory Dynamics (Information Pruning)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Forget Gate Density')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.2)

    # --- 4. OPTIMIZATION HEALTH (Gradient Norm) ---
    ax4.plot(steps, grad_norm, color='#00f2ff', alpha=0.4, label='Avg Grad Norm')
    ax4.plot(steps, smooth(grad_norm, smoothing_window), color='#00f2ff', linewidth=2, label='Smoothed Grad Norm')
    if np.any(grad_norm > 0):
        ax4.set_yscale('log')
    ax4.set_title('Optimization Health / Stability', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Gradient Norm (Log Scale)')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.2)

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
    parser.add_argument('--log', type=str, default="orbax_checkpoints/training_history.csv", 
                        help="Path to training_history.csv")
    args = parser.parse_args()
    
    plot_training_history(log_path=args.log)