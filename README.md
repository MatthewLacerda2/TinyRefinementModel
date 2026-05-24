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
