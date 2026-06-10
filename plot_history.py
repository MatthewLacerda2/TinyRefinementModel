import csv
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import argparse
from config import (
    BATCH_SIZE,
    MAX_SEQ_LEN,
    LATENT_DIM,
    NUM_BLOCKS,
    SHARED_SLOTS,
    MAX_STEPS_LIMIT,
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
    """Parameter/memory stats derived from the live model's nnx.state — counted
    from real shapes and dtypes, never transcribed formulas (which drift; the
    old hand-math here wrongly assumed f16 parameter storage)."""
    print("Calculating model parameters & memory footprint from the live model...")
    import jax
    from flax import nnx
    from model import UniversalReasoner

    with jax.default_device(jax.devices("cpu")[0]):
        model = UniversalReasoner(LATENT_DIM, nnx.Rngs(0))

    params = nnx.state(model, nnx.Param)
    leaves_with_paths = jax.tree_util.tree_flatten_with_path(params)[0]

    def group_of(top_key):
        if top_key in ("encoder_stack", "decoder_stack", "reasoning_stack"):
            return top_key
        if top_key in ("embed", "time_embed", "shared_token"):
            return "embeddings"
        return "heads & norms"

    group_params = {}
    group_bytes = {}
    for path, leaf in leaves_with_paths:
        top = str(getattr(path[0], 'key', path[0]))
        group = group_of(top)
        group_params[group] = group_params.get(group, 0) + leaf.size
        group_bytes[group] = group_bytes.get(group, 0) + leaf.size * leaf.dtype.itemsize

    total_params = sum(group_params.values())
    total_weight_bytes = sum(group_bytes.values())

    # Optimizer state (AdamW: 2 f32 moments per parameter) and f32 gradients
    optimizer_bytes = total_params * 4 * 2
    gradient_bytes = total_params * 4

    # Activation memory (heuristic estimate, f16 compute):
    # z_seq through the blocks + reasoning slot states across steps
    act_bytes_est = (BATCH_SIZE * MAX_SEQ_LEN * LATENT_DIM * 2 * NUM_BLOCKS * 2)
    act_bytes_est += (BATCH_SIZE * MAX_STEPS_LIMIT * SHARED_SLOTS * LATENT_DIM * 2)

    total_vram_gb = (total_weight_bytes + optimizer_bytes + gradient_bytes + act_bytes_est) / (1024**3)

    print(f"Model Parameters: {total_params:,}")
    order = ["embeddings", "encoder_stack", "decoder_stack", "reasoning_stack", "heads & norms"]
    for group in order:
        if group in group_params:
            note = f" (1 physical block, shared across {NUM_BLOCKS} iterations)" if group == "reasoning_stack" else ""
            print(f"  |-- {group:16}: {group_params[group]:>12,}{note}")
    print("-" * 50)
    print("ESTIMATED VRAM FOOTPRINT (Training)")
    print(f"  |-- Weights           : {total_weight_bytes / (1024**2):.2f} MB (f32 storage, f16 compute)")
    print(f"  |-- Optimizer (AdamW) : {optimizer_bytes / (1024**2):.2f} MB")
    print(f"  |-- Gradients (f32)   : {gradient_bytes / (1024**2):.2f} MB")
    print(f"  |-- Activations (Est) : {act_bytes_est / (1024**2):.2f} MB")
    print(f"  => TOTAL ESTIMATED   : {total_vram_gb:.2f} GB")

def plot_training_history(log_path=None):
    if log_path is None:
        # Try to auto-discover the latest metrics.csv from the runs directory
        discovered = None
        if os.path.exists("runs"):
            import glob
            run_dirs = sorted(glob.glob(os.path.join("runs", "run_*")))
            for r_dir in reversed(run_dirs):
                csv_path = os.path.join(r_dir, "metrics.csv")
                if os.path.exists(csv_path):
                    discovered = csv_path
                    break
        if discovered:
            print(f"🔎 Auto-discovered latest metrics CSV: {discovered}")
            log_path = discovered
        else:
            print("❌ Error: No training runs or metrics.csv found inside 'runs/'.")
            return
    elif not os.path.exists(log_path):
        print(f"❌ Error: {log_path} not found.")
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
                        'ponder_cost': float(row.get('ponder_cost', 0)),
                        'mean_halt_step': float(row.get('mean_halt_step', 0)),
                    })
                except ValueError:
                    continue 
    except Exception as e:
        print(f"❌ Error reading {log_path}: {e}")
        return

    if len(history) == 0:
        print("ℹ️ Warning: No valid training data found in CSV.")
        return

    # CSVs written before resume-trimming existed contain replayed (non-monotonic)
    # step ranges; keep only the first occurrence of each advancing step.
    monotonic = []
    last_step = -1
    for entry in history:
        if entry['step'] > last_step:
            monotonic.append(entry)
            last_step = entry['step']
    if len(monotonic) < len(history):
        print(f"ℹ️ Dropped {len(history) - len(monotonic)} non-monotonic (replayed) rows from the CSV.")
    history = monotonic

    steps = np.array([e['step'] for e in history])
    ce = np.array([e['ce'] for e in history])
    first_ce = np.array([e['first_ce'] for e in history])
    diversity = np.array([e['diversity'] for e in history])
    forget = np.array([e['forget_cost'] for e in history])
    grad_norm = np.array([e['grad_norm'] for e in history])
    temporal_drift = np.array([e['temporal_drift'] for e in history])
    mean_halt_step = np.array([e['mean_halt_step'] for e in history])

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
    if np.all(ce > 0):
        ax1.set_yscale('log')
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
    ax3_twin = ax3.twinx()
    ax3_twin.plot(steps, smooth(mean_halt_step, smoothing_window), color='#ffaa00', linewidth=1, label='Mean Halt Step')
    ax3_twin.set_ylabel('Mean Halt Step', color='#ffaa00')
    ax3.set_title('Dynamic Compute & Resource Penalties', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Cost Value')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.1)

    # --- 4. MODEL HEALTH (Grad Norm & Diversity) ---
    ax4.plot(steps, smooth(grad_norm, smoothing_window), color='#ffffff', linewidth=2, label='Grad Norm')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(steps, smooth(diversity, smoothing_window), color='#ff00ff', linewidth=1, label='Diversity Loss')
    ax4_twin.set_ylabel('Diversity Loss', color='#ff00ff')
    ax4.set_title('Optimization Health', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Gradient Norm')
    if np.any(grad_norm > 0):
        ax4.set_yscale('log')
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
    parser.add_argument('--log', type=str, default=None, 
                        help="Path to the metrics CSV file (defaults to auto-discovering the latest run)")
    args = parser.parse_args()
    
    plot_training_history(log_path=args.log)