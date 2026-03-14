import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
import csv
import time
import tiktoken
import os
import threading
import queue
import multiprocessing as mp
import concurrent.futures
from functools import partial
from dotenv import load_dotenv
import jax.lax as lax
import numpy as np
from datasets import load_dataset
from train_local import (
    UniversalReasoner,
    train_step,
    optimizer_chain,
    LATENT_DIM, MAX_SEQ_LEN, BATCH_SIZE, ACCUMULATION_STEPS, PAD_TOKEN_ID, FORGET_LAMBDA
)

load_dotenv()

CHECKPOINT_INTERVAL = 10
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
                print(f"🔄 Dataset {self.dataset_name} exhausted (Epoch complete)! Restarting from beginning...")
                self.skip_count = 0
                self.items_yielded = 0
                self.iterator = None
                self._ensure_iterator()
                continue
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
        self.sources = sources
        self.weights = weights
        
    def get_batch(self, batch_size):
        counts = np.random.multinomial(batch_size, self.weights)
        batch_list = []
        for source, count in zip(self.sources, counts):
            if count > 0:
                b = source.get_batch(count)
                if b is not None and b.shape[0] > 0:
                    batch_list.append(b)
        if not batch_list:
            # Fallback if datasets strangely completely fail
            return self.sources[0].get_batch(batch_size) 
        return jnp.concatenate(batch_list, axis=0)

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
    total_samples_seen = (start_step - 1) * BATCH_SIZE * ACCUMULATION_STEPS
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

        for i in range(ACCUMULATION_STEPS):
            current_batch = batch_queue.get()
            if current_batch is None: break

            loss, (ce, p, forget_cost, halt_diag) = train_step(
                model, optimizer, current_batch, step, FORGET_LAMBDA
            )
            for k in step_diag: step_diag[k] += halt_diag[k]

            step_loss += loss
            step_ce += ce
            step_p += p
            step_forget_cost += forget_cost
            last_loss = loss

        if current_batch is None: break
        if last_loss is not None: last_loss.block_until_ready()

        step_loss = float(step_loss) 
        step_ce = float(step_ce) / ACCUMULATION_STEPS
        step_p = float(step_p) / ACCUMULATION_STEPS
        step_forget_cost = float(step_forget_cost) / ACCUMULATION_STEPS
        step_diag = {k: float(v) / ACCUMULATION_STEPS for k, v in step_diag.items()}

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