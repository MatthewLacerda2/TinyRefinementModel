import json
import matplotlib.pyplot as plt
import os

def plot_training_history(history_path="training_history.json"):
    if not os.path.exists(history_path):
        print(f"Error: {history_path} not found.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    steps = [item['step'] for item in history]
    losses = [item['loss'] for item in history]
    difficulties = [item.get('difficulty', 0) for item in history]
    speeds = [item.get('speed', 0) for item in history]

    if not steps:
        print("No data to plot.")
        return

    # --- Plot 1: Loss & Difficulty ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss', color='blue')
    ax1.plot(steps, losses, color='blue', alpha=0.5, label='Step Loss')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_yscale('log') # Loss is better viewed in log for this scale

    ax2 = ax1.twinx()
    ax2.set_ylabel('Difficulty / N-Particles', color='red')
    ax2.plot(steps, difficulties, color='red', linewidth=2, label='Difficulty')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Training Progress: Loss (Log) and Difficulty (Linear)')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_progress.png')
    print("Full progress plot saved to training_progress.png")

    # --- Plot 2: Speed (Linear) ---
    plt.figure(figsize=(10, 5))
    plt.plot(steps, speeds, label='Steps/s', color='green', alpha=0.7)
    plt.title('Training Speed (steps/s)')
    plt.xlabel('Step')
    plt.ylabel('Speed (steps/s)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('speed_linear.png')
    print("Speed plot saved to speed_linear.png")

    plt.show()

if __name__ == "__main__":
    plot_training_history()