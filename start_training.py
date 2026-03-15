import os

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
#os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from flax import nnx
import orbax.checkpoint as ocp
import csv
import time
import tiktoken
import threading
import queue
import multiprocessing as mp
import concurrent.futures
from functools import partial
from dotenv import load_dotenv
import numpy as np
from datasets import load_dataset
from train_local import (
    UniversalReasoner,
    train_step,
    optimizer_chain,
    LATENT_DIM, MAX_SEQ_LEN, BATCH_SIZE, PAD_TOKEN_ID, FORGET_LAMBDA, HUNCH_REFRESH_EVERY
)

load_dotenv()

CHECKPOINT_INTERVAL = 50
SORT_BUFFER_SIZE = 1000
PREFETCH_SIZE = 16

GLOBAL_POOL = None

def get_global_pool():
    global GLOBAL_POOL
    if GLOBAL_POOL is None:
        GLOBAL_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
    return GLOBAL_POOL

def global_tokenize_item(text, max_seq_len, enc_name):
    import tiktoken
    enc = tiktoken.get_encoding(enc_name)
    tokens = enc.encode(text)
    if len(tokens) < max_seq_len:
        tokens = tokens + [PAD_TOKEN_ID] * (max_seq_len - len(tokens))
    else:
        tokens = tokens[: max_seq_len]
    return tokens

class TextDataGenerator:
    def __init__(self, directory, max_seq_len=MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len
        self.directory = directory
        self.files = sorted([f for f in os.listdir(directory) if f.endswith('.npy')])
        self.current_file_idx = 0
        self.data = None
        self.pointer = 0
        self.exhausted = False
        self.skip_count = 0

    def _load_next_file(self):
        if self.current_file_idx >= len(self.files):
            self.exhausted = True
            return False
        
        file_path = os.path.join(self.directory, self.files[self.current_file_idx])
        print(f"📖 Streaming {file_path} into TPU memory...")
        self.data = np.load(file_path)
        self.pointer = 0
        
        if self.skip_count > 0:
            # Handle resume-from-checkpoint skipping
            tokens_to_skip = self.skip_count * self.max_seq_len
            if tokens_to_skip < len(self.data):
                self.pointer = tokens_to_skip
                self.skip_count = 0
            else:
                self.skip_count -= (len(self.data) // self.max_seq_len)
                self.current_file_idx += 1
                return self._load_next_file()
                
        self.current_file_idx += 1
        return True

    def get_batch(self, batch_size):
        if self.exhausted: return None
        if self.data is None or self.pointer + (batch_size * self.max_seq_len) > len(self.data):
            if not self._load_next_file(): return None
            
        # Reshape the flat token stream into [batch, seq_len]
        total_tokens = batch_size * self.max_seq_len
        batch = self.data[self.pointer : self.pointer + total_tokens]
        self.pointer += total_tokens
        return jnp.array(batch.reshape(batch_size, self.max_seq_len), dtype=jnp.int32)

class DataMixer:
    def __init__(self, sources, weights):
        self.sources = list(sources)
        self.weights = list(weights)
        
    def get_batch(self, batch_size):
        while len(self.sources) > 0:
            counts = np.random.multinomial(batch_size, self.weights)
            batch_list = []
            exhausted_indices = []
            
            for i, (source, count) in enumerate(zip(self.sources, counts)):
                if count > 0:
                    b = source.get_batch(count)
                    if getattr(source, "exhausted", False) or b is None:
                        exhausted_indices.append(i)
                    else:
                        batch_list.append(b)
            
            if exhausted_indices:
                new_sources = []
                new_weights = []
                for i in range(len(self.sources)):
                    if i not in exhausted_indices:
                        new_sources.append(self.sources[i])
                        new_weights.append(self.weights[i])
                
                self.sources = new_sources
                if len(self.sources) == 0:
                    return None
                    
                total_w = sum(new_weights)
                self.weights = [w / total_w for w in new_weights]
                continue
                
            if batch_list:
                return jnp.concatenate(batch_list, axis=0)
                
        return None

def start_prefetch_worker(data_gen, batch_size, q):
    def worker():
        while True:
            batch = data_gen.get_batch(batch_size)
            if batch is None:
                q.put(None)
                return
            q.put(batch)
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t

class LossMonitor:
    def __init__(self, patience=500, window=500):
        self.patience = patience
        self.window = window
        self.ce_history = []
        self.best_ce = float("inf")
        self.last_improvement_step = 0

    def push(self, step, ce_loss):
        self.ce_history.append(ce_loss)
        if len(self.ce_history) > self.window: self.ce_history.pop(0)
        avg_ce = sum(self.ce_history) / len(self.ce_history)
        if avg_ce < (self.best_ce - 0.01):
            self.best_ce = avg_ce
            self.last_improvement_step = step
            return False
        return (step - self.last_improvement_step) > self.patience

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"🚀 Initializing Dynamic Latent Reasoner (Dim={LATENT_DIM})...")
    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(42))
    optimizer = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)

    mesh = Mesh(jax.devices(), ('batch',))

    # PartitionSpec('batch') tells JAX to slice the 0th dimension across cores
    data_sharding = NamedSharding(mesh, PartitionSpec('batch'))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    # Replicate the initial weights and optimizer state to all 8 cores
    state = nnx.state((model, optimizer))
    state = jax.tree.map(lambda x: jax.device_put(x, replicated_sharding), state)
    nnx.update((model, optimizer), state)

    history_file = "training_history.csv"
    monitor = LossMonitor()

    mngr = ocp.CheckpointManager(
        os.path.abspath("orbax_checkpoints"),
        item_names=("model", "optimizer", "monitor_state", "step"),
        options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True),
    )

    if mngr.latest_step() is not None:
        latest_step = mngr.latest_step()
        print(f"📖 Loading Orbax checkpoint from step {latest_step}...")
        restored = mngr.restore(
            latest_step,
            args=ocp.args.Composite(
                model=ocp.args.StandardRestore(nnx.state(model)),
                optimizer=ocp.args.StandardRestore(nnx.state(optimizer)),
                monitor_state=ocp.args.JsonRestore(),
                step=ocp.args.JsonRestore(),
            ),
        )
        nnx.update(model, restored["model"])
        nnx.update(optimizer, restored["optimizer"])
        start_step = restored["step"] + 1
        m_state = restored["monitor_state"]
        monitor.ce_history = m_state["ce_history"]
        monitor.best_ce = m_state["best_ce"]
        monitor.last_improvement_step = m_state["last_improvement_step"]
        print(f"✅ Resuming from step {start_step}")
        del restored # Free up the temporary dictionary
        import gc; gc.collect()
    else:
        print("🆕 No checkpoint found, starting from scratch...")
        start_step = 1

    print("🚀 Initializing Dynamic Data Phases...")
    
    pretrain_sources = [
        TextDataGenerator("npy_dir/pretrain/fineweb-edu"),
        TextDataGenerator("npy_dir/pretrain/code_instructions"),
        TextDataGenerator("npy_dir/pretrain/finemath"),
    ]
    pretrain_mixer = DataMixer(pretrain_sources, [0.60, 0.25, 0.15])

    chat_mixer = TextDataGenerator("npy_dir/chat/ultrachat")

    if start_step > 1:
        if start_step < 25000:
            total_pretrain_seen = (start_step - 1) * BATCH_SIZE
            weights = [0.60, 0.25, 0.15]
            for gen, weight in zip(pretrain_sources, weights):
                gen.skip_count = int(total_pretrain_seen * weight)
        else:
            total_chat_seen = (start_step - 25000) * BATCH_SIZE
            chat_mixer.skip_count = total_chat_seen


    step = start_step
    hunch = None
    while True:
        t0 = time.time()
        step_loss = step_ce = step_p = step_forget_cost = 0.0
        step_data_wait = step_compute_time = 0.0
        step_diag = {k: 0.0 for k in [
            'logits_mean', 'logits_std', 'logits_min', 'logits_max', 
            'prob_mean', 'prob_std', 'saturation', 'temporal_drift', 
            'forget_density', 'logit_spread', 'diversity_loss'
        ]}

        last_loss = None
        t_data_start = time.time()
        if step < 25000:
            current_batch = pretrain_mixer.get_batch(BATCH_SIZE)
        else:
            if step == 25000:
                print("🚀 PHASE SHIFT: Transitioning to Ultrachat Fine-tuning...")
            current_batch = chat_mixer.get_batch(BATCH_SIZE)
            
        if current_batch is None: break
        
        current_batch = jax.device_put(current_batch, data_sharding)
        
        if hunch is not None:
            hunch = jax.device_put(hunch, data_sharding)
        
        t_data_end = time.time()
        step_data_wait = t_data_end - t_data_start
        
        loss, (ce, p, forget_cost, halt_diag), hunch = train_step(
            model, optimizer, current_batch, step, FORGET_LAMBDA, prev_hunch=hunch,
            should_truncate=False
        )
        
        loss.block_until_ready()
        t_compute_end = time.time()
        step_compute_time = t_compute_end - t_data_end

        for k in step_diag: step_diag[k] = float(halt_diag[k])

        step_loss = float(loss)
        step_ce = float(ce)
        step_p = float(p)
        step_forget_cost = float(forget_cost)
        last_loss = loss

        step_loss = float(step_loss) 
        step_ce = float(step_ce)
        step_p = float(step_p)
        step_forget_cost = float(step_forget_cost)
        step_diag = {k: float(jnp.mean(v)) for k, v in step_diag.items()}

        t_total = time.time() - t0

        if monitor.push(step, step_ce): 
            print("🛑 Training halted: No improvement in CE.")
            break

        if step % CHECKPOINT_INTERVAL == 0:
            mngr.save(
                step,
                args=ocp.args.Composite(
                    model=ocp.args.StandardSave(nnx.state(model)),
                    optimizer=ocp.args.StandardSave(nnx.state(optimizer)),
                    monitor_state=ocp.args.JsonSave({
                        "ce_history": monitor.ce_history,
                        "best_ce": monitor.best_ce,
                        "last_improvement_step": monitor.last_improvement_step,
                    }),
                    step=ocp.args.JsonSave(step),
                ),
            )
            mngr.wait_until_finished()

            print(
                f"Step {step:04d} | CE: {step_ce:.4f} | Agg Loss: {step_loss:.4f} | "
                f"Avg Steps: {step_p:.2f} | Forget: {step_forget_cost:.4f} | Time: {t_total:.2f}s\n"
                f"      Wait: {step_data_wait:.3f}s | Compute: {step_compute_time:.3f}s\n"
                f"      Logits [μ:{step_diag['logits_mean']:.2f}, σ:{step_diag['logits_std']:.2f}, min:{step_diag['logits_min']:.2f}, max:{step_diag['logits_max']:.2f}] | Spread: {step_diag['logit_spread']:.2f}\n"
                f"      Prob [μ:{step_diag['prob_mean']:.3f}, σ:{step_diag['prob_std']:.3f}] | Sat: {step_diag['saturation']:.3f} | Drift: {step_diag['temporal_drift']:.3f} | Forget: {step_diag['forget_density']:.3f}\n"
            )

            write_header = not os.path.exists(history_file) or os.path.getsize(history_file) == 0
            with open(history_file, "a", newline="") as f:
                fields = ["step", "loss", "ce", "avg_ponder", "avg_forget_cost", "t_total", "data_wait", "compute_time",
                          "logits_mean", "logits_std", "logits_min", "logits_max", 
                          "prob_mean", "prob_std", "saturation", "temporal_drift", 
                          "forget_density", "logit_spread", "diversity_loss"]
                writer = csv.DictWriter(f, fieldnames=fields)
                if write_header: writer.writeheader()
                
                row = {
                    "step": int(step), "loss": f"{step_loss:.4f}", "ce": f"{step_ce:.4f}",
                    "avg_ponder": f"{step_p:.2f}", "avg_forget_cost": f"{step_forget_cost:.4f}", "t_total": f"{t_total:.2f}",
                    "data_wait": f"{step_data_wait:.4f}", "compute_time": f"{step_compute_time:.4f}",
                }
                row.update({k: f"{v:.4f}" for k, v in step_diag.items() if k in fields})
                writer.writerow(row)

        step += 1