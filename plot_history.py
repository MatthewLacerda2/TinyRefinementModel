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

    if not steps:
        print("No data to plot.")
        return

    # --- Plot 1: Linear Scale ---
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, label='Loss', color='blue', alpha=0.7)
    plt.title('Training Loss (Linear Scale)')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig('loss_linear.png')
    print("Linear plot saved to loss_linear.png")

    # --- Plot 2: Log Scale ---
    plt.figure(figsize=(10, 5))
    plt.semilogy(steps, losses, label='Loss (Log)', color='red', alpha=0.7)
    plt.title('Training Loss (Log Scale)')
    plt.xlabel('Step')
    plt.ylabel('Loss (log)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig('loss_log.png')
    print("Log plot saved to loss_log.png")

    # Display both windows
    plt.show()

if __name__ == "__main__":
    plot_training_history()