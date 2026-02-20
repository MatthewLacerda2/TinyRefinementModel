import csv
import matplotlib.pyplot as plt
import os
import numpy as np
import math

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"

def plot_training_history(log_path="training_history.csv"):
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
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
                    't_total': float(row.get('t_total', 0))
                })
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return

    if not history:
        print(f"No data in {log_path} to plot.")
        return

    steps = [entry['step'] for entry in history]
    losses = [entry['loss'] for entry in history]
    ce_losses = [entry['ce'] for entry in history]
    ponder_steps = [entry['avg_ponder'] for entry in history]
    times = [entry['t_total'] for entry in history]

    # Create a clean, modern plot
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Aggregate Loss
    ax1.plot(steps, losses, color='#00f2ff', linewidth=2, label='Agg Loss')
    ax1.fill_between(steps, losses, color='#00f2ff', alpha=0.1)
    ax1.set_ylabel('Agg Loss', color='#00f2ff', fontweight='bold')
    ax1.set_title('Training Progress: Aggregate Loss', fontsize=14, pad=10, color='white')
    ax1.grid(True, linestyle='--', alpha=0.2)
    ax1.legend()

    # 2. CE Loss
    ax2.plot(steps, ce_losses, color='#ff007b', linewidth=2, label='CE Loss')
    ax2.fill_between(steps, ce_losses, color='#ff007b', alpha=0.1)
    ax2.set_ylabel('CE Loss', color='#ff007b', fontweight='bold')
    ax2.set_title('Cross-Entropy (Token Prediction)', fontsize=14, pad=10, color='white')
    ax2.grid(True, linestyle='--', alpha=0.2)
    ax2.legend()

    # 3. Ponder (Average Steps)
    ax3.plot(steps, ponder_steps, color='#adff2f', linewidth=2, label='Avg Ponder')
    ax3.fill_between(steps, ponder_steps, color='#adff2f', alpha=0.1)
    ax3.set_ylabel('Steps', color='#adff2f', fontweight='bold')
    ax3.set_xlabel('Training Step', fontweight='bold')
    ax3.set_title('Average Ponder Steps', fontsize=14, pad=10, color='white')
    ax3.grid(True, linestyle='--', alpha=0.2)
    ax3.legend()

    plt.tight_layout()
    plt.savefig('training_plot.png', dpi=120)
    print("✨ Training analytics updated: training_plot.png")

    # --- Prediction & Time Logic ---
    target_ppl = 40
    target_ce = math.log(target_ppl)
    
    current_step = steps[-1]
    elapsed_time = sum(times)
    
    # Calculate average time per step (recent window for accuracy)
    recent_time_window = times[-20:] if len(times) >= 20 else times
    avg_step_time = sum(recent_time_window) / len(recent_time_window)

    print(f"\n📊 --- Training Status ---")
    print(f"📍 Current Step: {current_step}")
    print(f"⏱️  Current Time: {format_time(elapsed_time)}")
    
    if len(steps) > 10:
        try:
            # Fit CE loss to a Logarithmic curve: CE = a + b * log(step)
            valid_idx = [i for i, s in enumerate(steps) if s > 0]
            log_steps = np.log([steps[i] for i in valid_idx])
            log_ce = [ce_losses[i] for i in valid_idx]
            
            b, a = np.polyfit(log_steps, log_ce, 1)
            
            current_ce = ce_losses[-1]
            current_ppl = math.exp(current_ce)
            
            print(f"📈 Current Perplexity: {current_ppl:.2f}")
            
            if current_ppl <= target_ppl:
                print(f"✅ Goal Reached! Target perplexity {target_ppl} achieved.")
            elif b >= 0:
                print(f"⚠️  Warning: CE loss is not strictly decreasing. Prediction unreliable.")
            else:
                # target_ce = a + b * log(target_step)
                target_step = math.exp((target_ce - a) / b)
                steps_remaining = target_step - current_step
                additional_time = steps_remaining * avg_step_time
                total_expected_time = elapsed_time + additional_time
                
                print(f"\n🎯 --- Goal Prediction (PPL {target_ppl}) ---")
                print(f"🔭 Projected Final Step: ~{int(target_step)}")
                print(f"⏳ Steps Remaining: ~{int(steps_remaining)}")
                print(f"🕰️  Total Expected Time: {format_time(total_expected_time)}")
                print(f"⌛ Remaining Time: {format_time(additional_time)}")
        except Exception as e:
            print(f"Could not calculate projection: {e}")

if __name__ == "__main__":
    plot_training_history()
