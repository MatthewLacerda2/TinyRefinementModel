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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Linear Scale
    ax1.plot(steps, losses, label='Loss', color='blue', alpha=0.7)
    ax1.set_title('Training Loss (Linear Scale)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.legend()

    # Log Scale
    ax2.semilogy(steps, losses, label='Loss (Log)', color='red', alpha=0.7)
    ax2.set_title('Training Loss (Log Scale)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss (log)')
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_plot.png')
    print("Plot saved to training_plot.png")
    plt.show()

if __name__ == "__main__":
    plot_training_history()
