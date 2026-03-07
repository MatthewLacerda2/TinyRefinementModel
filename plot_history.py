import csv
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import sys

# Ensure UTF-8 encoding for console output (important for Windows)
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
    scale = 'log' # Hardcoded for filename consistency
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
                    # CSV writes avg_forget_cost; accept avg_t_cost for backward compatibility
                    'avg_t_cost': float(row.get('avg_forget_cost', row.get('avg_t_cost', 0))),
                    't_total': float(row.get('t_total', 0))
                })
    except Exception as e:
        print(f"❌ Error reading {log_path}: {e}")
        return

    if not history:
        print(f"❌ No data in {log_path} to plot.")
        return

    steps = np.array([entry['step'] for entry in history])
    losses = np.array([entry['loss'] for entry in history])
    ce_losses = np.array([entry['ce'] for entry in history])
    ponder_steps = np.array([entry['avg_ponder'] for entry in history])
    avg_t_costs = np.array([entry['avg_t_cost'] for entry in history])
    times = np.array([entry['t_total'] for entry in history])

    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    
    # Helper to apply log scale
    def apply_scaling(ax):
        ax.set_yscale('log')

    # Aggregate Loss
    ax1.plot(steps, losses, color='#00f2ff', linewidth=2.5, label='Agg Loss', marker='o', markersize=4)
    ax1.fill_between(steps, losses, color='#00f2ff', alpha=0.1)
    ax1.set_ylabel('Agg Loss', color='#00f2ff', fontweight='bold', fontsize=11)
    ax1.set_title('Training Progress: Aggregate Loss', fontsize=14, pad=10, color='white', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='upper right')

    # CE Loss
    ax2.plot(steps, ce_losses, color='#ff007b', linewidth=2.5, label='CE Loss', marker='o', markersize=4)
    ax2.fill_between(steps, ce_losses, color='#ff007b', alpha=0.1)
    ax2.set_ylabel('CE Loss', color='#ff007b', fontweight='bold', fontsize=11)
    ax2.set_title('Cross-Entropy (Token Prediction)', fontsize=14, pad=10, color='white', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='upper right')

    # Temporal Cost
    ax3.plot(steps, avg_t_costs, color='#ffcc00', linewidth=2.5, label='Temporal Cost', marker='o', markersize=4)
    ax3.fill_between(steps, avg_t_costs, color='#ffcc00', alpha=0.1)
    ax3.set_ylabel('T-Cost', color='#ffcc00', fontweight='bold', fontsize=11)
    ax3.set_title('Temporal Consistency Cost (Lower = Better)', fontsize=14, pad=10, color='white', fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend(loc='upper right')

    # Ponder Steps
    ax4.plot(steps, ponder_steps, color='#adff2f', linewidth=2.5, label='Avg Ponder Steps', marker='o', markersize=4)
    ax4.fill_between(steps, ponder_steps, color='#adff2f', alpha=0.1)
    ax4.axhline(y=16, color='#adff2f', linestyle='--', alpha=0.5, label='MAX (16 steps)')
    ax4.set_ylabel('Steps', color='#adff2f', fontweight='bold', fontsize=11)
    ax4.set_xlabel('Training Step', fontweight='bold', fontsize=11)
    ax4.set_title('Average Ponder Steps (Model Learning Depth)', fontsize=14, pad=10, color='white', fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.3)
    ax4.legend(loc='upper left')
    
    for ax in [ax1, ax2, ax3, ax4]:
        apply_scaling(ax)

    ax4.set_ylim([0.9, 17.5])

    plt.tight_layout()
    plot_fn = f'training_plot_{scale}.png'
    plt.savefig(plot_fn, dpi=150, bbox_inches='tight')
    print(f"✨ Training analytics updated: {plot_fn}")
    plt.close()

    # Perplexity Plot (more intuitive than CE loss)
    ppl = np.exp(ce_losses)
    fig_ppl, ax_ppl = plt.subplots(1, 1, figsize=(12, 6))
    
    ax_ppl.plot(steps, ppl, color='#00ff88', linewidth=2.5, label='Perplexity', marker='o', markersize=4)
    ax_ppl.fill_between(steps, ppl, color='#00ff88', alpha=0.1)
    ax_ppl.axhline(y=40, color='#ff6b6b', linestyle='--', alpha=0.7, linewidth=2, label='GPT-2 Target (~40)')
    ax_ppl.set_ylabel('Perplexity', color='#00ff88', fontweight='bold', fontsize=12)
    ax_ppl.set_xlabel('Training Step', fontweight='bold', fontsize=12)
    ax_ppl.set_title('Perplexity Over Training (Lower is Better)', fontsize=14, pad=10, color='white', fontweight='bold')
    ax_ppl.grid(True, linestyle='--', alpha=0.3)
    ax_ppl.legend(loc='upper right', fontsize=11)
    
    apply_scaling(ax_ppl)

    plt.tight_layout()
    ppl_fn = f'training_perplexity_{scale}.png'
    plt.savefig(ppl_fn, dpi=150, bbox_inches='tight')
    print(f"✨ Perplexity plot updated: {ppl_fn}")
    plt.close()


    current_step = steps[-1]
    
    # Calculate average time per step from recent data
    recent_time_window = times[-20:] if len(times) >= 20 else times
    avg_step_time = np.mean(recent_time_window)
    
    elapsed_time = np.sum(times)  # Use actual total time, not estimated

    print(f"\n{'='*60}")
    print(f"📊 TRAINING STATUS")
    print(f"{'='*60}")
    print(f"📍 Current Step: {current_step:,}")
    print(f"⏱️  Total Elapsed Time: {format_time(elapsed_time)}")
    print(f"⚡ Avg Time per Step: {avg_step_time:.2f}s")
    print(f"📈 Current CE Loss: {ce_losses[-1]:.4f}")
    print(f"📉 Current Perplexity: {ppl[-1]:.2f}")
    print(f"🧠 Current Avg Ponder Steps: {ponder_steps[-1]:.2f}")
    print(f"⚙️  Current Temporal Cost: {avg_t_costs[-1]:.4f}")
    
    # Calculate learning dynamics
    if len(steps) > 10:
        recent_ce = ce_losses[-10:]
        ce_improvement_per_step = (recent_ce[0] - recent_ce[-1]) / 10
        print(f"\n📊 Recent Learning Rate:")
        print(f"   CE Loss improvement per step: {ce_improvement_per_step:.5f}")
        print(f"   (Recent 10 steps: {recent_ce[0]:.4f} → {recent_ce[-1]:.4f})")
    
    # Simple linear extrapolation (more realistic than power law for early training)
    print(f"\n{'='*60}")
    print(f"🎯 SIMPLE CONVERGENCE ESTIMATE")
    print(f"{'='*60}")
    
    try:
        if len(steps) > 20:
            # Use linear fit on last 20% of data
            recent_steps = steps[-max(10, len(steps)//5):]
            recent_ce_losses = ce_losses[-max(10, len(steps)//5):]
            
            # Linear regression
            coeffs = np.polyfit(recent_steps, recent_ce_losses, 1)
            slope, intercept = coeffs[0], coeffs[1]
            
            current_ce = ce_losses[-1]
            current_ppl = np.exp(current_ce)
            
            print(f"📈 Current Perplexity: {current_ppl:.2f}")
            print(f"📉 Recent CE Loss Trend: {slope:.6f} per step")
            
            if slope >= 0:
                print(f"⚠️  Loss is not decreasing! Training may have plateaued.")
            else:
                # Project to target PPL = 40
                target_ce = np.log(40)
                if current_ce > target_ce:
                    steps_to_target = (current_ce - target_ce) / abs(slope)
                    time_to_target = steps_to_target * avg_step_time
                    
                    print(f"\n🎯 Projected Path to PPL=40 (GPT-2 level):")
                    print(f"   Current PPL: {current_ppl:.2f}")
                    print(f"   Target PPL: 40")
                    print(f"   Estimated Steps Needed: {int(steps_to_target):,}")
                    print(f"   Estimated Time: {format_time(time_to_target)}")
                    print(f"   Expected Completion: {format_time(elapsed_time + time_to_target)} total")
                else:
                    print(f"\n🎉 GOAL REACHED! Current PPL {current_ppl:.2f} < Target 40")
            
            # Also estimate when loss might plateau
            if slope < 0:
                print(f"\n📌 Estimated Plateau Behavior:")
                ppl_per_1000_steps = np.exp(current_ce + slope * 1000) - current_ppl
                print(f"   In 1000 more steps: PPL would be ~{np.exp(current_ce + slope * 1000):.2f}")
                
        else:
            print(f"⏳ Not enough data yet (need >20 points for reliable estimate)")
            
    except Exception as e:
        print(f"⚠️  Could not calculate convergence projection: {e}")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    plot_training_history()