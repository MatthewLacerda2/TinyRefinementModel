import csv
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"

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
                    'avg_t_cost': float(row.get('avg_t_cost', 0)),
                    't_total': float(row.get('t_total', 0))
                })
    except Exception as e:
        print(f"❌ Error reading {log_path}: {e}")
        return

    if not history:
        print(f"❌ No data in {log_path} to plot.")
        return

    steps = np.array([entry['step'] for entry in history])
    ce_losses = np.array([entry['ce'] for entry in history])
    ponder_steps = np.array([entry['avg_ponder'] for entry in history])

    ppl = np.exp(ce_losses)

    current_step = int(steps[-1])
    current_ce = float(ce_losses[-1])
    current_ponder = float(ponder_steps[-1])

    min_ce_idx = int(np.argmin(ce_losses))
    min_ce = float(ce_losses[min_ce_idx])
    min_ce_step = int(steps[min_ce_idx])
    min_ppl = float(ppl[min_ce_idx])

    plt.style.use('dark_background')
    fig, (ax_ce, ax_ponder) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # CE loss over steps with highlights
    ax_ce.plot(steps, ce_losses, color='#ff007b', linewidth=2.0, label='CE Loss')
    ax_ce.scatter(current_step, current_ce, color='#00ff88', s=60, zorder=3, label='Current Step')
    ax_ce.scatter(min_ce_step, min_ce, color='#ffff00', s=60, zorder=3, label='Lowest CE')
    ax_ce.set_ylabel('CE Loss', color='#ff007b', fontweight='bold', fontsize=11)
    ax_ce.grid(True, linestyle='--', alpha=0.3)
    ax_ce.legend(loc='upper right', fontsize=9)

    # Ponder steps over training with current highlight
    ax_ponder.plot(steps, ponder_steps, color='#adff2f', linewidth=2.0, label='Avg Ponder')
    ax_ponder.scatter(current_step, current_ponder, color='#00f2ff', s=60, zorder=3, label='Current Step')
    ax_ponder.set_ylabel('Avg Ponder', color='#adff2f', fontweight='bold', fontsize=11)
    ax_ponder.set_xlabel('Training Step', fontweight='bold', fontsize=11)
    ax_ponder.grid(True, linestyle='--', alpha=0.3)
    ax_ponder.legend(loc='upper right', fontsize=9)

    # Summary of the requested key metrics
    summary_text = (
        f"Current step: {current_step:,} | "
        f"Lowest CE: {min_ce:.4f} @ step {min_ce_step:,} | "
        f"Lowest perplexity: {min_ppl:.2f} | "
        f"Current avg ponder: {current_ponder:.2f}"
    )
    fig.suptitle(summary_text, fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('training_simple.png', dpi=150, bbox_inches='tight')
    print("✨ Simple training plot updated: training_simple.png")
    print(summary_text)
    plt.close()

if __name__ == "__main__":
    plot_training_history()