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
    speeds = [item.get('speed', 0) for item in history]

    if not steps:
        print("No data to plot.")
        return

    # --- Plot 1: Loss (Linear) ---
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, label='Loss', color='blue', alpha=0.7)
    plt.title('Training Loss (Linear Scale)')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('loss_linear.png')
    print("Loss plot saved to loss_linear.png")

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