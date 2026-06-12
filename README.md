# Tiny Refinement Model

This repository implements the **Universal Reasoner**, a latent reasoning language model designed to demonstrate that continuous latent-space reasoning is highly effective compared to traditional token-level chain-of-thought scratchpads.

## Architecture

The Universal Reasoner separates sequence-level token processing from reasoning-level representation. Instead of generating human-readable reasoning tokens, the model processes thoughts in a high-dimensional continuous latent space. 

The architecture is divided into three specialized transformer blocks:

1. **Encoder Stack**: Processes raw input tokens into rich contextual representations using causal self-attention.
2. **Latent Reasoning Loop**: Operates over a set of private latent slot tokens, representing the model's continuous 'internal scratchpad'. This stack recursively processes the scratchpad over several steps (the depth is randomly sampled during training, fixed at inference). A forget mechanism dynamically updates the scratchpad by blending fresh thoughts with historical memory at each iteration.
3. **Decoder Stack**: Combines the output of the encoder stack with the scratchpad state the window *started* with — the hunch carried over from previous windows through a gating mechanism. The current window's completed reasoning is never shown to its own decoder (that would leak future tokens into past predictions); it is stored in the hunch cache and benefits the *next* window. Reasoning about what was just read helps predict what comes next.

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
         │   [Hunch Cache] ──► [Hunch Gate] ──► [Initial Scratchpad Slots]
         │  (from previous      (blend with               │
         │      window)          fresh slots)             ├──────────────┐
         │                                                ▼              │
         │                                  ┌─────────────────────────┐  │
         ├─────────────────────────────────►│  Reasoning Loop Stack   │  │
         │     (Attend to context)          │ (N steps; depth random  │  │
         │                                  │  during training)       │  │
         │                                  └────────────┬────────────┘  │
         │                                               │               │
         │                                  [Forget Gate]│(blend per     │
         │                                               │ iteration)    │
         │                                               ▼               │
         │                                        [Hunch Cache] ──► (next window)
         ▼                                                               │
┌──────────────────┐                                                     │
│  Decoder Stack   │◄────────────────────────────────────────────────────┘
│ (Blend & Decode) │ (Reads the slots this window STARTED with — strictly
└────────┬─────────┘  past information; this window's reasoning output
         │            only reaches the NEXT window via the hunch cache)
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

### 5. Tests and Diagnostics
The test suite runs on CPU by default, so it works even while a training run owns the GPU (set `RUN_TESTS_ON_GPU=1` to exercise the real half-precision path on a free GPU):
```bash
python -m pytest tests/
```
Offline diagnostics, each answering one question against the latest checkpoint:
```bash
PYTHONPATH=. python tools/eval_depth_curve.py    # does reasoning about the previous window help the next one?
PYTHONPATH=. python tools/overfit_smoke.py       # can the training path overfit one batch? (pre-run gate)
PYTHONPATH=. python tools/dump_transcripts.py    # fixed-prompt transcripts, archived per checkpoint
```
