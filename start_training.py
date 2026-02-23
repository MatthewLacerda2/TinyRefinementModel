import jax
import jax.numpy as jnp
from flax import nnx
import pickle
import csv
import time
import tiktoken
import os
from datasets import load_dataset, interleave_datasets
from train_local import (
    UniversalReasoner,
    TrainingManager,
    optimizer_chain,
    LATENT_DIM, MAX_SEQ_LEN, BATCH_SIZE, ACCUMULATION_STEPS, PAD_TOKEN_ID
)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

CHECKPOINT_INTERVAL = 1000

class TextDataGenerator:
    def __init__(self, max_seq_len=MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len
        self.enc = tiktoken.get_encoding("cl100k_base")
        
        token = os.environ.get("HF_TOKEN")
        print(f"🚀 Preparing SmolLM-Corpus mix (Auth: {'Yes' if token else 'No'})...")
        
        ds_cosmo = load_dataset("HuggingFaceTB/smollm-corpus", 
            "cosmopedia-v2", 
            split="train", 
            streaming=True,
            token=token
        ).select_columns(["text"])
        
        ds_fineweb = load_dataset(
            "HuggingFaceTB/smollm-corpus", 
            "fineweb-edu-dedup", 
            split="train", 
            streaming=True,
            token=token
        ).select_columns(["text"])
        
        self.dataset = interleave_datasets([ds_cosmo, ds_fineweb], stopping_strategy="all_exhausted")
        self.iterator = iter(self.dataset)
        self.exhausted = False

    def get_batch(self, batch_size):
        if self.exhausted: return None
        batch_ids = []
        while len(batch_ids) < batch_size:
            try:
                item = next(self.iterator)
                tokens = self.enc.encode(item['text'])
                if len(tokens) < self.max_seq_len:
                    tokens = tokens + [PAD_TOKEN_ID] * (self.max_seq_len - len(tokens))
                else:
                    tokens = tokens[:self.max_seq_len]
                batch_ids.append(tokens)
            except StopIteration: self.exhausted = True; break
            except Exception as e: continue
        return jnp.array(batch_ids, dtype=jnp.int32)

class LossMonitor:
    def __init__(self, patience=2000, window=500, max_ponder_limit=7.5):
        self.patience = patience
        self.window = window
        self.max_ponder_limit = max_ponder_limit
        
        self.ce_history = []
        self.best_ce = float('inf')
        self.last_improvement_step = 0

    def push(self, step, ce_loss, avg_ponder):
        self.ce_history.append(ce_loss)
        if len(self.ce_history) > self.window: 
            self.ce_history.pop(0)
            
        avg_ce = sum(self.ce_history) / len(self.ce_history)
        
        # Condition 1: Check for actual learning (CE loss dropping)
        if avg_ce < (self.best_ce - 0.01):
            self.best_ce = avg_ce
            self.last_improvement_step = step
            return False
            
        # Condition 2: Out of patience for CE improvement
        if (step - self.last_improvement_step) > self.patience:
            print(f"\n🛑 Plateau detected: No CE improvement > 0.01 for {self.patience} steps.")
            return True
            
        # Condition 3: Saturation detected (Pondering maxed out)
        if avg_ponder >= self.max_ponder_limit:
            print(f"\n🛑 Saturation detected: Avg ponder steps maxed out at {avg_ponder:.2f}.")
            return True
            
        return False

if __name__ == "__main__":
    print(f"🚀 Initializing Dynamic Latent Reasoner (Dim={LATENT_DIM})...")
    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(42), dtype=jnp.bfloat16)
    manager = TrainingManager(model, optimizer_chain)
    
    data_gen = TextDataGenerator(MAX_SEQ_LEN)
    history_file = "training_history.csv"
    monitor = LossMonitor()
    
    checkpoint_path = "checkpoint.pkl"
    if os.path.exists(checkpoint_path):
        print(f"📖 Loading checkpoint from {checkpoint_path}...")
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        
        if "model_state" in checkpoint:
            nnx.update(model, checkpoint["model_state"])
            nnx.update(manager.optimizer, checkpoint["optim_state"])
        else:
            print("❌ Error: Valid checkpoint state not found.")
            exit(1)
        
        start_step = checkpoint.get("step", 0) + 1
        
        if "monitor_state" in checkpoint:
            m_state = checkpoint["monitor_state"]
            monitor.ce_history = m_state.get("ce_history", [])
            monitor.best_ce = m_state.get("best_ce", float('inf'))
            monitor.last_improvement_step = m_state.get("last_improvement_step", 0)
        else:
            monitor.last_improvement_step = start_step
            
        print(f"✅ Resuming from step {start_step}")
    else:
        print("🆕 No checkpoint found, starting from scratch...")
        start_step = 1

    key = jax.random.key(start_step)
    step = start_step
    accum_loss, accum_ce, accum_p = 0.0, 0.0, 0.0

    while True:
        t0 = time.time()
        
        for i in range(ACCUMULATION_STEPS):
            key, subkey = jax.random.split(key)
            batch = data_gen.get_batch(BATCH_SIZE)
            if batch is None: break

            manager.grad_buffer, loss, aux = manager.accumulate_grad_step(
                batch, subkey, manager.grad_buffer
            )

            token_loss, ponder_val = aux
            accum_loss += float(loss) / ACCUMULATION_STEPS
            accum_ce += float(token_loss) / ACCUMULATION_STEPS
            accum_p += float(ponder_val) / ACCUMULATION_STEPS

        if batch is None: break

        manager.apply_updates()
        
        t_total = time.time() - t0

        if monitor.push(step, accum_ce, accum_p): break
        
        if step % CHECKPOINT_INTERVAL == 0:
            with open("checkpoint.pkl", "wb") as f:
                checkpoint_data = {
                    "model_state": nnx.state(model),
                    "optim_state": nnx.state(manager.optimizer), 
                    "step": step,
                    "monitor_state": {
                        "ce_history": monitor.ce_history,
                        "best_ce": monitor.best_ce,
                        "last_improvement_step": monitor.last_improvement_step
                    }
                }
                pickle.dump(checkpoint_data, f)
            
            print(f"Step {step:04d} | Agg Loss: {accum_loss:.4f} | Avg Steps: {accum_p:.2f} | Time: {t_total:.2f}s")
            
            with open(history_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["step", "loss", "ce", "avg_ponder", "t_total"])
                if f.tell() == 0: writer.writeheader()
                writer.writerow({
                    "step": int(step), 
                    "loss": f"{accum_loss:.4f}", 
                    "ce": f"{accum_ce:.4f}", 
                    "avg_ponder": f"{accum_p:.2f}", 
                    "t_total": f"{t_total:.2f}"
                })
            
        step += 1
