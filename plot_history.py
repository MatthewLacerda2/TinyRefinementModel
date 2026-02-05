import csv
import matplotlib.pyplot as plt
import os

def plot_training_history(log_path="training_log.csv"):
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    steps = []
    losses = []
    avg_losses = []
    difficulties = []
    avg_steps = []

    try:
        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(float(row['step'])))
                losses.append(float(row['loss']))
                avg_losses.append(float(row['avg_loss']))
                difficulties.append(float(row['difficulty']))
                avg_steps.append(float(row['avg_steps']))
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return

    if not steps:
        print("No data to plot.")
        return

    # Create a 3-panel plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

    # 1. Loss (Log Scale)
    ax1.plot(steps, losses, color='#3498db', alpha=0.3, label='Step Loss')
    ax1.plot(steps, avg_losses, color='#2980b9', linewidth=2, label='Avg Loss (Smoothed)')
    ax1.set_ylabel('Loss (Log)')
    ax1.set_yscale('log')
    ax1.set_title('Training Loss - Log Scale')
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    # 2. Difficulty
    ax2.plot(steps, difficulties, color='#e74c3c', linewidth=2)
    ax2.set_ylabel('Difficulty')
    ax2.set_title('Simulation Difficulty Over Time')
    ax2.grid(True, alpha=0.3)
    
    # 3. Avg Steps (Complexity)
    ax3.plot(steps, avg_steps, color='#f39c12', linewidth=2)
    ax3.set_ylabel('Horizon / Steps')
    ax3.set_xlabel('Training Step')
    ax3.set_title('Prediction Horizon (Avg Steps)')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_plot.png')
    print("Full training plot saved to training_plot.png")

if __name__ == "__main__":
    plot_training_history()