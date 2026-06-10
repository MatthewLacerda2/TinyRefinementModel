# Tiny Refinement Model

This repository implements the **Universal Reasoner**, a latent reasoning language model designed to demonstrate that continuous latent-space reasoning is highly effective compared to traditional token-level chain-of-thought scratchpads.

## Architecture

The Universal Reasoner separates sequence-level token processing from reasoning-level representation. Instead of generating human-readable reasoning tokens, the model processes thoughts in a high-dimensional continuous latent space. 

The architecture is divided into three specialized transformer blocks:

1. **Encoder Stack**: Processes raw input tokens into rich contextual representations using causal self-attention.
2. **Latent Reasoning Loop**: Operates over a set of private latent slot tokens, representing the model's continuous 'internal scratchpad'. This stack recursively processes the scratchpad over several steps. A hunch gating mechanism carries memory across sequence segments, while a forget mechanism dynamically updates the scratchpad by blending fresh thoughts with historical memory at each iteration.
3. **Decoder Stack**: Combines the output of the encoder stack with the final state of the latent scratchpad, merging the sequence context with the model's completed reasoning before mapping the representations back to predict the next token.

### Information Flow

```
  [Input Tokens] 
        │
        ▼
  [Embedding]
        │
        ▼
┌──────────────────┐
│  Encoder Stack   │ (Process input context causally)
└────────┬─────────┘
         │
         ├───► [Hunch Gate] ───► [Initialize Latent Scratchpad]
         │                                   │
         │                                   ▼
         │                      ┌─────────────────────────┐
         ├─────────────────────►│  Reasoning Loop Stack   │◄─────────┐
         │  (Attend to context) │  (Runs for N steps)     │          │
         │                      └────────────┬────────────┘          │ (Recursive
         │                                   │                       │  feedback 
         │                                   ▼                       │  loop)
         │                             [Forget Head]                 │
         │                        (Project current & new)            │
         │                                   │                       │
         │                                   ▼                       │
         │                             [Forget Gate] ────────────────┘
         │                        (Sigmoid-gated blend)
         ▼                                   │
┌──────────────────┐                         ▼
│  Decoder Stack   │◄────────────────────────┘
│ (Blend & Decode) │ (Read final scratchpad states)
└────────┬─────────┘
         │
         ▼
    [LM Head]
         │
         ▼
  [Output logits]
```

---

## Tech stack

* **JAX**: High-performance numerical computing library used for compiling and executing optimized array operations.
* **Flax NNX**: Modern, module-based neural network library for JAX that simplifies state management and parameter tracking.
* **Optax**: JAX-native optimization library used to build gradient processing pipelines and decay schedules.
* **AdamW**: Our primary optimization algorithm (configured with learning rate warmup and weight decay) which updates the model's parameters stably.
* **Orbax**: High-performance checkpointing library for JAX used to save and load training states and metadata asynchronously.
* **Hugging Face**: Streamed dataset pipelines used for curriculum training and evaluation.
* **Tiktoken**: Fast byte-pair encoding tokenizer using the `cl100k_base` encoding vocabulary.

---

## How to Run

Before running the scripts, activate your virtual environment and ensure you have copied the environment file template:
```bash
source venv/bin/activate
cp .env.example .env  # Update HF_TOKEN inside .env
```

### 1. Prefill Data Tokenization
Download and pre-tokenize the FineWeb-Edu, CodeParrot-clean, FineMath, and UltraChat datasets. This processes and chunks tokens into `runs/data/` for high-throughput training:
```bash
python prefill.py
```

### 2. Start or Resume Training
Initiate the model training loop. The script automatically handles data mixing, curriculum learning, and SFT plateau transitions:
* **Auto-Resume (Default)**: Automatically detects and resumes the latest training run directory (reusing checkpoints if saved, or beginning from step 1 appending to the same `metrics.csv` if checkpoints do not exist yet):
  ```bash
  python start_training.py
  ```
* **Force Brand New Run**: Ignore previous checkpoints/runs and start entirely from scratch:
  ```bash
  python start_training.py --new-run
  ```
* **Custom Checkpoint Folder**: Point to a specific directory containing Orbax checkpoint segments:
  ```bash
  python start_training.py --checkpoint-path runs/run_xxxxx/checkpoints
  ```

### 3. Plot Training History
Visualize the metrics, resource costs, and optimization health:
* **Auto-Discover Latest Run**: Analyzes the most recent run's `metrics.csv` inside `runs/` and outputs `reasoning_analytics.png`:
  ```bash
  python plot_history.py
  ```
* **Specific Run**: Plot a custom log file:
  ```bash
  python plot_history.py --log runs/run_xxxxx/metrics.csv
  ```

### 4. Run Local Inference CLI
Chat interactively with the trained Universal Reasoner. The CLI automatically loads the latest saved checkpoint weights:
```bash
python infer_local.py
```
