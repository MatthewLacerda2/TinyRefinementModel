import json
import matplotlib.pyplot as plt
import os

def plot_training_history(history_path="training_history.json"):
    if not os.path.exists(history_path):
        print(f"Error: {history_path} not found.")
        return

    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
    except Exception as e:
        print(f"Error reading {history_path}: {e}")
        return

    if not history:
        print("No data to plot.")
        return

    steps = [item['step'] for item in history]
    losses = [item['loss'] for item in history]
    avg_losses = [item.get('avg_loss', item['loss']) for item in history]
    difficulties = [item.get('difficulty', 0) for item in history]

    # Create a 3-panel plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

    # 1. Loss (Linear)
    ax1.plot(steps, losses, color='blue', alpha=0.3, label='Step Loss')
    ax1.plot(steps, avg_losses, color='blue', linewidth=2, label='Avg Loss (Smoothed)')
    ax1.set_ylabel('Loss (Linear)')
    ax1.set_title('Training Loss - Linear Scale')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Loss (Log)
    ax2.plot(steps, losses, color='blue', alpha=0.3)
    ax2.plot(steps, avg_losses, color='blue', linewidth=2)
    ax2.set_ylabel('Loss (Log)')
    ax2.set_yscale('log')
    ax2.set_title('Training Loss - Log Scale')
    ax2.grid(True, which="both", alpha=0.3)

    # 3. Difficulty
    ax3.plot(steps, difficulties, color='red', linewidth=2)
    ax3.set_ylabel('Difficulty / N-Particles')
    ax3.set_xlabel('Step')
    ax3.set_title('Simulation Difficulty Over Time')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_plot.png')
    print("Full training plot saved to training_plot.png")

if __name__ == "__main__":
    plot_training_history()