import csv
import matplotlib.pyplot as plt
import os

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
                    'halt_loss': float(row.get('halt_loss', 0))
                })
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return

    if not history:
        print(f"No data in {log_path} to plot.")
        return

    steps = [entry['step'] for entry in history]
    losses = [entry['loss'] for entry in history]
    halt_losses = [entry['halt_loss'] for entry in history]

    # Create a clean, modern plot
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 1. Total Loss
    ax1.plot(steps, losses, color='#00f2ff', linewidth=2, label='Total Loss')
    ax1.fill_between(steps, losses, color='#00f2ff', alpha=0.1)
    ax1.set_ylabel('Loss', color='#00f2ff', fontweight='bold')
    ax1.set_title('Training Progress: Total Loss', fontsize=14, pad=15, color='white')
    ax1.grid(True, linestyle='--', alpha=0.2)
    ax1.legend()

    # 2. Halt Loss (Targeting timing/efficiency)
    ax2.plot(steps, halt_losses, color='#ff007b', linewidth=2, label='Halt Loss')
    ax2.fill_between(steps, halt_losses, color='#ff007b', alpha=0.1)
    ax2.set_ylabel('Halt Loss', color='#ff007b', fontweight='bold')
    ax2.set_xlabel('Training Step', fontweight='bold')
    ax2.set_title('Guided Halting Progress (Stability)', fontsize=14, pad=15, color='white')
    ax2.grid(True, linestyle='--', alpha=0.2)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_plot.png', dpi=120)
    print("✨ Training analytics updated: training_plot.png")

if __name__ == "__main__":
    plot_training_history()