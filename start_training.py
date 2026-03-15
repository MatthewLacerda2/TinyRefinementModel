import os

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
#os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax.numpy as jnp
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
    def __init__(self, dataset_name, max_seq_len=MAX_SEQ_LEN, config_name=None):
        self.max_seq_len = max_seq_len
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.enc_name = "cl100k_base"
        self.skip_count = 0
        self.items_yielded = 0
        
        self.iterator = None
        self.pool = get_global_pool()
        self.exhausted = False

    def _ensure_iterator(self):
        if self.iterator is not None:
            return

        token = os.environ.get("HF_TOKEN")
        print(f"🚀 Preparing {self.dataset_name} (Config: {self.config_name})...")

        ds_kwargs = {
            "path": self.dataset_name,
            "split": "train",
            "streaming": True,
            "token": token,
        }

        if self.config_name:
            ds_kwargs["name"] = self.config_name
        elif "fineweb" in self.dataset_name and "edu" not in self.dataset_name:
            ds_kwargs["name"] = "default"
        
        ds = load_dataset(**ds_kwargs)
        
        total_skip = self.skip_count + self.items_yielded
        if total_skip > 0:
            print(f"⏩ Fast-forwarding {total_skip} items in {self.dataset_name}...")
            ds = ds.skip(total_skip)

        self.iterator = self._curriculum_iterator(ds)

    def _curriculum_iterator(self, ds):
        buffer = []
        for example in ds:
            buffer.append(example)
            if len(buffer) >= SORT_BUFFER_SIZE:
                if "score" in buffer[0]:
                    buffer.sort(key=lambda x: x.get("score", 0))
                yield from buffer
                buffer = []
        if buffer:
            if "score" in buffer[0]:
                buffer.sort(key=lambda x: x.get("score", 0))
            yield from buffer

    def get_batch(self, batch_size):
        if getattr(self, "exhausted", False):
            return None
        self._ensure_iterator()
        
        batch_texts = []
        text_column = None

        while len(batch_texts) < batch_size:
            try:
                item = next(self.iterator)
                self.items_yielded += 1
                
                if text_column is None:
                    # Find the best column for content, avoiding metadata/IDs
                    options = ["prompt", "text", "content", "code", "markdown"]
                    text_column = next((col for col in options if col in item), None)
                    if text_column is None:
                        # Fallback: find the longest string column that isn't an ID
                        string_cols = [k for k, v in item.items() if isinstance(v, str)]
                        text_column = next((k for k in string_cols if "id" not in k.lower()), string_cols[0])

                batch_texts.append(item[text_column])
            except StopIteration:
                print(f"🔄 Dataset {self.dataset_name} exhausted! Stopping iteration for this dataset.")
                self.exhausted = True
                self.iterator = None
                return None
            except Exception as e:
                print(f"⚠️ Stream error in {self.dataset_name}: {e}. Reconnecting...")
                self.iterator = None
                self._ensure_iterator()
                continue
        
        if not batch_texts: return None
            
        tokenize_func = partial(global_tokenize_item, 
                                 max_seq_len=self.max_seq_len, 
                                 enc_name=self.enc_name)
        
        batch_ids = list(self.pool.map(tokenize_func, batch_texts))
        return jnp.array(batch_ids, dtype=jnp.int32)

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

    print("🚀 Preparing Data Mixture (Balanced Reasoning & STEM)...")
    sources = [
        TextDataGenerator("HuggingFaceFW/fineweb-edu"),                   # 50% Primary High-Quality
        TextDataGenerator("HuggingFaceFW/fineweb"),                       # 15% General Web
        TextDataGenerator("iamtarun/python_code_instructions_18k_alpaca"), # 15% Python Instructions
        TextDataGenerator("HuggingFaceTB/finemath", config_name="finemath-4plus"), # 15% Math
        TextDataGenerator("HuggingFaceTB/cosmopedia-v2", config_name="cosmopedia-v2"), # 5% Synthetic 
    ]
    weights = [0.50, 0.15, 0.15, 0.15, 0.05]
    
    # Calculate skip counts for resume-safety
    total_samples_seen = (start_step - 1) * BATCH_SIZE
    if total_samples_seen > 0:
        for gen, weight in zip(sources, weights):
            gen.skip_count = int(total_samples_seen * weight)

    data_gen = DataMixer(sources, weights)
    batch_queue = queue.Queue(maxsize=PREFETCH_SIZE)
    start_prefetch_worker(data_gen, BATCH_SIZE, batch_queue)

    step = start_step
    while True:
        t0 = time.time()
        step_loss = step_ce = step_p = step_forget_cost = 0.0
        step_diag = {k: 0.0 for k in [
            'logits_mean', 'logits_std', 'logits_min', 'logits_max', 
            'prob_mean', 'prob_std', 'saturation', 'temporal_drift', 
            'forget_density', 'logit_spread', 'diversity_loss'
        ]}

        last_loss = None
        current_batch = None
        hunch = None

        current_batch = batch_queue.get()
        if current_batch is None: break
        
        # We no longer use should_truncate within an accumulation loop.
        # train_step handles refreshing every HUNCH_REFRESH_EVERY global steps.
        loss, (ce, p, forget_cost, halt_diag), hunch = train_step(
            model, optimizer, current_batch, step, FORGET_LAMBDA, prev_hunch=hunch,
            should_truncate=False
        )
        for k in step_diag: step_diag[k] = float(halt_diag[k])

        step_loss = float(loss)
        step_ce = float(ce)
        step_p = float(p)
        step_forget_cost = float(forget_cost)
        last_loss = loss
        
        if last_loss is not None: last_loss.block_until_ready()

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
                f"      Logits [μ:{step_diag['logits_mean']:.2f}, σ:{step_diag['logits_std']:.2f}, min:{step_diag['logits_min']:.2f}, max:{step_diag['logits_max']:.2f}] | Spread: {step_diag['logit_spread']:.2f}\n"
                f"      Prob [μ:{step_diag['prob_mean']:.3f}, σ:{step_diag['prob_std']:.3f}] | Sat: {step_diag['saturation']:.3f} | Drift: {step_diag['temporal_drift']:.3f} | Forget: {step_diag['forget_density']:.3f}\n"
            )

            write_header = not os.path.exists(history_file) or os.path.getsize(history_file) == 0
            with open(history_file, "a", newline="") as f:
                fields = ["step", "loss", "ce", "avg_ponder", "avg_forget_cost", "t_total",
                          "logits_mean", "logits_std", "logits_min", "logits_max", 
                          "prob_mean", "prob_std", "saturation", "temporal_drift", 
                          "forget_density", "logit_spread", "diversity_loss"]
                writer = csv.DictWriter(f, fieldnames=fields)
                if write_header: writer.writeheader()
                
                row = {
                    "step": int(step), "loss": f"{step_loss:.4f}", "ce": f"{step_ce:.4f}",
                    "avg_ponder": f"{step_p:.2f}", "avg_forget_cost": f"{step_forget_cost:.4f}", "t_total": f"{t_total:.2f}",
                }
                row.update({k: f"{v:.4f}" for k, v in step_diag.items() if k in fields})
                writer.writerow(row)

        step += 1