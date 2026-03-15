import csv
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
from train_local import BATCH_SIZE, MAX_SEQ_LEN, UniversalReasoner, LATENT_DIM, NUM_BLOCKS, VOCAB_SIZE, SHARED_SLOTS, MAX_STEPS_LIMIT

# Ensure UTF-8 encoding for console output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s" if hours > 0 else f"{minutes}m {secs}s"

def calculate_tokens(step):
    total = step * BATCH_SIZE * MAX_SEQ_LEN
    return total

def print_model_stats():
    print("Calculating model parameters (mathematical estimation)...")
    
    # Architecture Constants
    num_heads = 8
    num_groups = 2 
    head_dim = LATENT_DIM // num_heads
    
    # 1. Block Stack (Per Block)
    q_params = LATENT_DIM * LATENT_DIM + LATENT_DIM
    k_params = LATENT_DIM * (num_groups * head_dim) + (num_groups * head_dim)
    v_params = LATENT_DIM * (num_groups * head_dim) + (num_groups * head_dim)
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
    print(f"(encoder params: {encoder_params:,}\n Layers params: {num_reason_param:,} across {NUM_BLOCKS} layers}})")

def plot_training_history(log_path="training_history.csv"):
    if not os.path.exists(log_path):
        print(f"❌ Error: {log_path} not found.")
        return

    history = []
    try:
        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append({
                    'step': int(row['step']),
                    'loss': float(row['loss']),
                    'ce': float(row.get('ce', 0)),
                    'avg_ponder': float(row.get('avg_ponder', 0)),
                    'saturation': float(row.get('saturation', 0)),
                    'drift': float(row.get('temporal_drift', 0)),
                    'forget': float(row.get('forget_density', 0)),
                    'spread': float(row.get('logit_spread', 0)),
                    'l_mean': float(row.get('logits_mean', 0)),
                    't_total': float(row.get('t_total', 0)),
                })
    except Exception as e:
        print(f"❌ Error reading {log_path}: {e}")
        return

    steps = np.array([e['step'] for e in history])
    losses = np.array([e['loss'] for e in history])
    ce = np.array([e['ce'] for e in history])
    ponder = np.array([e['avg_ponder'] for e in history])
    drift = np.array([e['drift'] for e in history])
    sat = np.array([e['saturation'] for e in history])
    forget = np.array([e['forget'] for e in history])
    spread = np.array([e['spread'] for e in history])
    l_mean = np.array([e['l_mean'] for e in history])

    plt.style.use('dark_background')
    fig, axes = plt.subplots(5, 1, figsize=(14, 24), sharex=True)
    (ax1, ax2, ax3, ax4, ax5) = axes

    # --- 1. CONVERGENCE (Loss vs CE) ---
    ax1.plot(steps, losses, color='#00f2ff', alpha=0.4, label='Agg Loss (Penalty Included)')
    ax1.plot(steps, ce, color='#ff007b', linewidth=2, label='CE Loss (Pure Accuracy)')
    ax1.set_yscale('log')
    ax1.set_title('Convergence: Accuracy vs. Structural Penalty', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2)

    # --- 2. REASONING EFFICIENCY (Drift / Ponder) ---
    efficiency = drift / (ponder + 1e-6)
    ax2.plot(steps, efficiency, color='#adff2f', linewidth=2, label='Efficiency (Drift/Step)')
    ax2.set_title('Reasoning Efficiency: Knowledge Gained per Thought Step', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Ratio', color='#adff2f')
    ax2.legend(loc='upper left')
    
    # Dual axis for total Drift
    ax2_b = ax2.twinx()
    ax2_b.plot(steps, drift, color='#00ff00', alpha=0.3, label='Raw Temporal Drift')
    ax2_b.set_ylabel('Total Drift', color='#00ff00')
    ax2.grid(True, alpha=0.2)

    # --- 3. EXPERT DYNAMICS (Saturation vs Logits) ---
    ax3.plot(steps, sat, color='#ff00ff', linewidth=2, label='Memory Saturation (Lower = Better)')
    ax3.set_ylabel('Saturation', color='#ff00ff')
    ax3.set_title('Expert Specialization & Halt Decision', fontsize=14, fontweight='bold')
    
    ax3_b = ax3.twinx()
    ax3_b.plot(steps, l_mean, color='#ffcc00', linestyle='--', alpha=0.6, label='Halt Logit Mean')
    ax3_b.set_ylabel('Logit Mean', color='#ffcc00')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.2)

    # --- 4. DECISIVENESS (Logit Spread & Ponder) ---
    ax4.plot(steps, spread, color='#ffffff', linewidth=1.5, label='Logit Spread (Decisiveness)')
    ax4.set_ylabel('Spread', color='#ffffff')
    
    ax4_b = ax4.twinx()
    ax4_b.plot(steps, ponder, color='#ff8800', linewidth=2, label='Avg Ponder Steps')
    ax4_b.set_ylim([0, 17])
    ax4_b.set_ylabel('Steps', color='#ff8800')
    ax4.set_title('Internal Decisiveness vs Reasoning Depth', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Training Step')
    ax4.legend(loc='lower right')

    # --- 5. MEMORY MANAGEMENT (Forgetting) ---
    ax5.plot(steps, forget, color='#00ff88', linewidth=2, label='Forget Density (Active Pruning)')
    ax5.set_ylabel('Density', color='#00ff88')
    ax5.set_title('Memory Management: Information Pruning Intensity', fontsize=14, fontweight='bold')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('reasoning_analytics.png', dpi=150)
    print("✨ Analytics updated: reasoning_analytics.png")
    #make the tokens outputted with x.xxx.xxx so i can clearly see the millions and thousands
    print(f"Amount of tokens trained so far: {calculate_tokens(steps[-1]):,}")

    print(f"Last CE change: {ce[-1] - ce[-2]}")
    print(f"Lowest CE so far: {min(ce)}")

    print_model_stats()

    plt.close()

if __name__ == "__main__":
    plot_training_history()