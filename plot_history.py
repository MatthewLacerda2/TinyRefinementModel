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
                    'avg_t_cost': float(row.get('avg_t_cost', 0)),
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
    avg_t_costs = [entry.get('avg_t_cost', 0) for entry in history]
    times = [entry['t_total'] for entry in history]

    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
    
    # Aggregate Loss
    ax1.plot(steps, losses, color='#00f2ff', linewidth=2, label='Agg Loss')
    ax1.fill_between(steps, losses, color='#00f2ff', alpha=0.1)
    ax1.set_ylabel('Agg Loss', color='#00f2ff', fontweight='bold')
    ax1.set_title('Training Progress: Aggregate Loss', fontsize=14, pad=10, color='white')
    ax1.grid(True, linestyle='--', alpha=0.2)
    ax1.legend()

    # CE Loss
    ax2.plot(steps, ce_losses, color='#ff007b', linewidth=2, label='CE Loss')
    ax2.fill_between(steps, ce_losses, color='#ff007b', alpha=0.1)
    ax2.set_ylabel('CE Loss', color='#ff007b', fontweight='bold')
    ax2.set_title('Cross-Entropy (Token Prediction)', fontsize=14, pad=10, color='white')
    ax2.grid(True, linestyle='--', alpha=0.2)
    ax2.legend()

    # Temporal Cost
    ax3.plot(steps, avg_t_costs, color='#ffcc00', linewidth=2, label='Temporal Cost')
    ax3.fill_between(steps, avg_t_costs, color='#ffcc00', alpha=0.1)
    ax3.set_ylabel('T-Cost', color='#ffcc00', fontweight='bold')
    ax3.set_title('Temporal Consistency Cost', fontsize=14, pad=10, color='white')
    ax3.grid(True, linestyle='--', alpha=0.2)
    ax3.legend()

    # Ponder Steps
    ax4.plot(steps, ponder_steps, color='#adff2f', linewidth=2, label='Avg Ponder')
    ax4.fill_between(steps, ponder_steps, color='#adff2f', alpha=0.1)
    ax4.set_ylabel('Steps', color='#adff2f', fontweight='bold')
    ax4.set_xlabel('Training Step', fontweight='bold')
    ax4.set_title('Average Ponder Steps', fontsize=14, pad=10, color='white')
    ax4.grid(True, linestyle='--', alpha=0.2)
    ax4.legend()

    plt.tight_layout()
    plt.savefig('training_plot.png', dpi=120)
    print("✨ Training analytics updated: training_plot.png")
    plt.close()

    # Log Scale Plot
    fig_log, (ax_l1, ax_l2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Agg Loss Log-Log
    ax_l1.plot(steps, losses, color='#00f2ff', linewidth=2, label='Agg Loss (Log)')
    ax_l1.fill_between(steps, losses, color='#00f2ff', alpha=0.1)
    ax_l1.set_xscale('log')
    ax_l1.set_yscale('log')
    ax_l1.set_ylabel('Agg Loss (Log)', color='#00f2ff', fontweight='bold')
    ax_l1.set_title('Training Progress: Agg Loss (Log-Log Scale)', fontsize=14, pad=10, color='white')
    ax_l1.grid(True, which="both", ls="-", alpha=0.1)
    ax_l1.legend()

    # CE Loss Log-Log
    ax_l2.plot(steps, ce_losses, color='#ff007b', linewidth=2, label='CE Loss (Log)')
    ax_l2.fill_between(steps, ce_losses, color='#ff007b', alpha=0.1)
    ax_l2.set_xscale('log')
    ax_l2.set_yscale('log')
    ax_l2.set_ylabel('CE Loss (Log)', color='#ff007b', fontweight='bold')
    ax_l2.set_xlabel('Training Step (Log)', fontweight='bold')
    ax_l2.set_title('Cross-Entropy (Log-Log Scale)', fontsize=14, pad=10, color='white')
    ax_l2.grid(True, which="both", ls="-", alpha=0.1)
    ax_l2.legend()

    plt.tight_layout()
    plt.savefig('training_plot_log.png', dpi=120)
    plt.close()
    print("✨ Log analytics updated: training_plot_log.png")

    target_ppl = 40
    target_ce = math.log(target_ppl)
    
    current_step = steps[-1]
    
    recent_time_window = times[-20:] if len(times) >= 20 else times
    avg_step_time = sum(recent_time_window) / len(recent_time_window)
    
    elapsed_time = current_step * avg_step_time

    print(f"\n📊 --- Training Status ---")
    print(f"📍 Current Step: {current_step}")
    print(f"⏱️  Current Time: {format_time(elapsed_time)} (est.)")
    
    if len(steps) > 10:
        print(f"\n🎯 --- Convergence Prediction ---")
        try:
            from scipy.optimize import curve_fit
            
            valid_idx = [i for i, s in enumerate(steps) if s > 0 and ce_losses[i] > 0]
            x_data = np.array([steps[i] for i in valid_idx], dtype=float)
            y_data = np.array([ce_losses[i] for i in valid_idx], dtype=float)
            
            def power_law(x, a, b, c):
                return a * np.power(x, -b) + c
                
            min_y = np.min(y_data)
            p0 = [y_data[0] - min_y, 0.5, min_y * 0.9]
            
            bounds = ([0, 0, 0], [np.inf, 5.0, min_y])
            
            popt, _ = curve_fit(power_law, x_data, y_data, p0=p0, bounds=bounds, maxfev=10000)
            a, b, c = popt
            
            asymptotic_ppl = math.exp(c)
            current_ce = ce_losses[-1]
            current_ppl = math.exp(current_ce)
            
            print(f"📈 Current Perplexity: {current_ppl:.2f}")
            print(f"🧱 Absolute PPL Floor (Theoretical Asymptote): ~{asymptotic_ppl:.2f}")
            
            pred_step = current_step
            while pred_step < 1_000_000:
                current_loss = power_law(pred_step, a, b, c)
                future_loss = power_law(pred_step + 2000, a, b, c)
                
                if (current_loss - future_loss) < 0.01:
                    break
                pred_step += 50
                
            final_ce = power_law(pred_step, a, b, c)
            final_ppl = math.exp(final_ce)
            
            steps_remaining = pred_step - current_step
            
            if steps_remaining <= 0:
                print("\n🛑 Model is projected to be converging right now based on the monitor thresholds!")
            else:
                additional_time = steps_remaining * avg_step_time
                total_expected_time = elapsed_time + additional_time
                
                print(f"\n🔭 Projected Halt Step (LossMonitor Trigger): ~{int(pred_step)}")
                print(f"🎯 Projected Final PPL at Halt: ~{final_ppl:.2f}")
                print(f"⏳ Steps Remaining: ~{int(steps_remaining)}")
                print(f"🕰️  Total Expected Time: {format_time(total_expected_time)}")
                print(f"⌛ Remaining Time: {format_time(additional_time)}")
                
        except ImportError:
            print("⚠️  Scipy is required for asymptotic projection. Please run `pip install scipy`.")
        except Exception as e:
            print(f"⚠️  Could not calculate convergence projection: {e}")

if __name__ == "__main__":
    plot_training_history()
