import os

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
#os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
import time
import threading
import queue
import multiprocessing as mp
from dotenv import load_dotenv
import numpy as np
import fsspec
from train_local import (
    UniversalReasoner,
    train_step,
    LATENT_DIM, MAX_SEQ_LEN, BATCH_SIZE, PAD_TOKEN_ID, SHARED_SLOTS
)
from schedulers import optimizer_chain
from metrics_logger import LossMonitor, MetricsLogger

load_dotenv()

CHECKPOINT_INTERVAL = 100
SORT_BUFFER_SIZE = 1000
PREFETCH_SIZE = 16
PHASE_STEP = 2000

DATA_ROOT = os.path.abspath(os.environ.get("DATA_ROOT", ""))
CHECKPOINT_ROOT = os.path.abspath(os.environ.get("CHECKPOINT_ROOT", "orbax_checkpoints"))

if not DATA_ROOT:
    print(f"⚠️ Warning: DATA_ROOT is not set. Data loading will likely fail unless provided via environment.")
    
print(f"📁 Checkpoints will be saved to: {CHECKPOINT_ROOT}")
if not CHECKPOINT_ROOT.startswith("gs://"):
    print(f"ℹ️ Note: Saving locally. You will need to manually sync to GCS using: gsutil -m cp -r {CHECKPOINT_ROOT} gs://YOUR_BUCKET/")


class TextDataGenerator:
    def __init__(self, directory, max_seq_len=MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len
        self.directory = directory
        
        self.fs, self.path_prefix = fsspec.core.url_to_fs(directory)
        
        all_files = self.fs.ls(directory)
        self.files = sorted([f for f in all_files if f.endswith('.npy')])
        
        self.current_file_idx = 0
        self.data = None
        self.pointer = 0
        self.exhausted = False
        self.skip_count = 0
        self.is_new_file = False

    def _load_next_file(self):
        if self.current_file_idx >= len(self.files):
            self.exhausted = True
            return False
        
        file_path = self.files[self.current_file_idx]
        print(f"📖 Streaming {file_path} into TPU memory...")
        
        with self.fs.open(file_path, 'rb') as f:
            self.data = np.load(f)
            
        self.pointer = 0
        
        if self.skip_count > 0:
            tokens_to_skip = self.skip_count * self.max_seq_len
            if tokens_to_skip < len(self.data):
                self.pointer = tokens_to_skip
                self.skip_count = 0
            else:
                self.skip_count -= (len(self.data) // self.max_seq_len)
                self.current_file_idx += 1
                return self._load_next_file()
                
        self.current_file_idx += 1
        self.is_new_file = True
        return True

    def get_batch(self, batch_size):
        if self.exhausted: return None, None
        total_tokens = batch_size * self.max_seq_len
        
        if self.data is None or self.pointer + total_tokens > len(self.data):
            if not self._load_next_file():
                return None, None
            # If the new file is also exhausted or too small, retry
            if self.exhausted or self.pointer + total_tokens > len(self.data):
                return self.get_batch(batch_size)

        batch = self.data[self.pointer : self.pointer + total_tokens]
        self.pointer += total_tokens
        
        reset_mask = np.zeros((batch_size,), dtype=bool)
        if self.is_new_file:
            reset_mask[:] = True
            self.is_new_file = False
            
        return jnp.array(batch.reshape(batch_size, self.max_seq_len), dtype=jnp.int32), jnp.array(reset_mask)

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
                    res = source.get_batch(count)
                    if res is None or getattr(source, "exhausted", False):
                        exhausted_indices.append(i)
                    else:
                        batch_list.append(res)
            
            if exhausted_indices:
                new_sources, new_weights = [], []
                for i, (s, w) in enumerate(zip(self.sources, self.weights)):
                    if i not in exhausted_indices:
                        new_sources.append(s); new_weights.append(w)
                self.sources = new_sources
                if not self.sources: return None, None
                total_w = sum(new_weights)
                self.weights = [w / total_w for w in new_weights]
                continue
                
            if batch_list:
                batches, masks = zip(*batch_list)
                return jnp.concatenate(batches, axis=0), jnp.concatenate(masks, axis=0)
        return None, None


# LossMonitor moved to metrics_logger.py


def init_model_and_optimizer():
    print(f"🚀 Initializing Dynamic Latent Reasoner (Dim={LATENT_DIM})...")
    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(42))
    optimizer = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)
    
    return model, optimizer

def load_or_create_checkpoint(model, optimizer):
    monitor = LossMonitor()
    mngr = ocp.CheckpointManager(
        CHECKPOINT_ROOT,
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
        graphdef, _ = nnx.split(optimizer)
        optimizer = nnx.merge(graphdef, restored["optimizer"])
        
        start_step = restored["step"] + 1
        m_state = restored["monitor_state"]
        monitor.ce_history = m_state["ce_history"]
        monitor.best_ce = m_state["best_ce"]
        monitor.last_improvement_step = m_state["last_improvement_step"]
        
        print(f"✅ Resuming from step {start_step}")
        del restored 
        import gc; gc.collect()
    else:
        print("🆕 No checkpoint found, starting from scratch...")
        start_step = 1

    return mngr, monitor, start_step

def setup_data_pipeline(start_step):
    print("🚀 Initializing Dynamic Data Phases...")
    pretrain_sources = [
        TextDataGenerator(f"{DATA_ROOT}/pretrain/fineweb-edu"),
        TextDataGenerator(f"{DATA_ROOT}/pretrain/code_instructions"),
        TextDataGenerator(f"{DATA_ROOT}/pretrain/finemath"),
    ]
    pretrain_mixer = DataMixer(pretrain_sources, [0.60, 0.25, 0.15])
    chat_mixer = TextDataGenerator(f"{DATA_ROOT}/chat/ultrachat")

    if start_step > 1:
        if start_step < PHASE_STEP:
            total_pretrain_seen = (start_step - 1) * BATCH_SIZE
            weights = [0.60, 0.25, 0.15]
            for gen, weight in zip(pretrain_sources, weights):
                gen.skip_count = int(total_pretrain_seen * weight)
        else:
            total_chat_seen = (start_step - PHASE_STEP) * BATCH_SIZE
            chat_mixer.skip_count = total_chat_seen

    data_queue = queue.Queue(maxsize=PREFETCH_SIZE)

    def data_wrapper():
        current_step = start_step
        while True:
            if current_step < PHASE_STEP:
                res = pretrain_mixer.get_batch(BATCH_SIZE)
            else:
                res = chat_mixer.get_batch(BATCH_SIZE)
            
            if res[0] is None:
                data_queue.put((None, None))
                break
            
            data_queue.put(res)
            current_step += 1

    threading.Thread(target=data_wrapper, daemon=True).start()
    return data_queue

def train_loop(model, optimizer, data_queue, mngr, monitor, start_step):
    history_file = f"{CHECKPOINT_ROOT}/training_history.csv"
    logger = MetricsLogger(history_file)
    step = start_step
    hunch = jnp.zeros((BATCH_SIZE, SHARED_SLOTS, LATENT_DIM))
    
    while True:
        t0 = time.time()
        t_data_start = time.time()
        
        current_batch, reset_mask = data_queue.get() 
        if current_batch is None: 
            print("🏁 Data stream exhausted.")
            break
        
        if step == PHASE_STEP:
            print("🚀 PHASE SHIFT: Transitioning to Chat Fine-tuning...")
        
        t_data_end = time.time()
        step_data_wait = t_data_end - t_data_start
        
        loss, (ce, p, forget_cost, halt_diag), hunch = train_step(
            model, optimizer, current_batch, jnp.array(step), prev_hunch=hunch,
            should_truncate=reset_mask
        )
        
        loss.block_until_ready()
        t_compute_end = time.time()
        step_compute_time = t_compute_end - t_data_end

        step_diag = logger.extract_diags(halt_diag, jnp.mean)
        step_loss = float(loss)
        step_ce = float(ce)
        step_p = float(p)
        step_forget_cost = float(forget_cost)

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
            logger.log(step, step_loss, step_ce, step_p, step_forget_cost, t_total, step_data_wait, step_compute_time, step_diag)
        step += 1

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    model, optimizer = init_model_and_optimizer()
    
    mngr, monitor, start_step = load_or_create_checkpoint(model, optimizer)
    
    data_queue = setup_data_pipeline(start_step)
    
    train_loop(model, optimizer, data_queue, mngr, monitor, start_step)