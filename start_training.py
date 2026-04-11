import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
import time
import threading
import queue
import multiprocessing as mp
from dotenv import load_dotenv
from train_local import (
    UniversalReasoner,
    compute_grad_step,
    apply_grads,
    LATENT_DIM, BATCH_SIZE, ACCUMULATION_STEPS, SHARED_SLOTS,
    optimizer_chain
)

from metrics_logger import LossMonitor, MetricsLogger

load_dotenv()

LOG_INTERVAL = 5
CHECKPOINT_INTERVAL = 20
SORT_BUFFER_SIZE = 1000
PREFETCH_SIZE = 128
PHASE_STEP = 1000

DATA_ROOT = os.path.abspath(os.environ.get("DATA_ROOT", ""))
CHECKPOINT_ROOT = os.path.abspath(os.environ.get("CHECKPOINT_ROOT", "orbax_checkpoints"))

if not DATA_ROOT:
    print(f"⚠️ Warning: DATA_ROOT is not set. Data loading will likely fail unless provided via environment.")
    
#print(f"📁 Checkpoints will be saved to: {CHECKPOINT_ROOT}")
#if not CHECKPOINT_ROOT.startswith("gs://"):
#    print(f"ℹ️ Note: Saving locally. You will need to manually sync to GCS using: gsutil -m cp -r {CHECKPOINT_ROOT} gs://YOUR_BUCKET/")

from data_loaders import TextDataGenerator, DataMixer

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
        monitor.ce_history = m_state.get("ce_history", [])
        monitor.best_ce = m_state.get("best_ce", float("inf"))
        monitor.best_loss = m_state.get("best_loss", float("inf"))
        monitor.best_avg_ce = m_state.get("best_avg_ce", monitor.best_ce)
        monitor.last_improvement_step = m_state.get("last_improvement_step", 0)
        
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
            total_pretrain_seen = (start_step - 1) * ACCUMULATION_STEPS * BATCH_SIZE
            weights = [0.60, 0.25, 0.15]
            for gen, weight in zip(pretrain_sources, weights):
                gen.skip_count = int(total_pretrain_seen * weight)
        else:
            total_chat_seen = (start_step - PHASE_STEP) * ACCUMULATION_STEPS * BATCH_SIZE
            chat_mixer.skip_count = total_chat_seen

    data_queue = queue.Queue(maxsize=PREFETCH_SIZE)

    def data_wrapper():
        macro_step = start_step
        micro_counter = 0
        
        while True:
            # Switch phases based on the MACRO step
            if macro_step < PHASE_STEP:
                res = pretrain_mixer.get_batch(BATCH_SIZE)
            else:
                res = chat_mixer.get_batch(BATCH_SIZE)
            
            if res[0] is None:
                data_queue.put((None, None))
                break
            
            data_queue.put(res)
            
            # Only increment the macro_step when a full accumulated batch is assembled
            micro_counter += 1
            if micro_counter % ACCUMULATION_STEPS == 0:
                macro_step += 1

    threading.Thread(target=data_wrapper, daemon=True).start()
    return data_queue

def train_loop(model, optimizer, data_queue, mngr, monitor, start_step):
    history_file = f"{CHECKPOINT_ROOT}/training_history.csv"
    logger = MetricsLogger(history_file)
    step = start_step
    hunch = jnp.zeros((BATCH_SIZE, SHARED_SLOTS, LATENT_DIM))
    
    while True:
        t0 = time.time()
        step_data_wait = 0.0
        
        accum_loss = 0.0
        accum_ce = 0.0
        accum_p = 0.0
        accum_forget_cost = 0.0
        last_halt_diag = None
        first_halt_diag = None
        first_ce = None

        # Grad norm tracking across accumulation window
        grad_norm_min = float('inf')
        grad_norm_max = float('-inf')
        grad_norm_sum = 0.0
        
        if step == PHASE_STEP:
            print("🚀 PHASE SHIFT: Transitioning to Chat Fine-tuning...")

        # Accumulation loop
        for micro_step in range(ACCUMULATION_STEPS):
            t_data_start = time.time()
            current_batch, reset_mask = data_queue.get() 
            if current_batch is None: 
                break
            
            step_data_wait += (time.time() - t_data_start)
            
            # The optax.apply_every(128) in schedulers.py will handle the accumulation
            loss, (ce, p, forget_cost, halt_diag), hunch, grads, grad_norm = compute_grad_step(
                model, current_batch, jnp.array(step), prev_hunch=hunch,
                should_truncate=reset_mask
            )
            
            apply_grads(optimizer, grads, model)

            # If any batch was truncated (file boundary), we must zero out the values, not just stop gradients
            hunch = jnp.where(reset_mask[:, None, None], jnp.zeros_like(hunch), hunch)

            gn = float(grad_norm)
            grad_norm_min = min(grad_norm_min, gn)
            grad_norm_max = max(grad_norm_max, gn)
            grad_norm_sum += gn

            accum_loss += float(loss)
            accum_ce += float(ce) / ACCUMULATION_STEPS
            accum_p += float(p) / ACCUMULATION_STEPS
            accum_forget_cost += float(forget_cost) / ACCUMULATION_STEPS
            last_halt_diag = halt_diag

            if micro_step == 0:
                first_halt_diag = halt_diag
                first_ce = float(ce)
            
        if current_batch is None:
            print("🏁 Data stream exhausted.")
            break
            
        loss_val = accum_loss / ACCUMULATION_STEPS
        
        t_compute_end = time.time()
        step_compute_time = t_compute_end - t0 - step_data_wait

        step_diag = logger.extract_diags(last_halt_diag, jnp.mean)
        step_loss = float(loss_val)
        step_ce = float(accum_ce)
        step_p = float(accum_p)
        step_forget_cost = float(accum_forget_cost)

        grad_norm_avg = grad_norm_sum / ACCUMULATION_STEPS

        # Detect frozen metrics: compare first vs last micro-step halt logit mean
        first_logit_mu = float(jnp.mean(jnp.array(first_halt_diag.get('logits_mean', 0.0)))) if first_halt_diag else float('nan')
        last_logit_mu  = float(jnp.mean(jnp.array(last_halt_diag.get('logits_mean', 0.0)))) if last_halt_diag else float('nan')
        logit_drift = abs(last_logit_mu - first_logit_mu)

        t_total = time.time() - t0

        if monitor.push(step, step_ce, step_loss): 
            print("🛑 Training halted: No improvement in CE.")
            break

        # Save if we hit a new record or every CHECKPOINT_INTERVAL steps
        if monitor.is_new_best or (step % CHECKPOINT_INTERVAL == 0):
            mngr.save(
                step,
                args=ocp.args.Composite(
                    model=ocp.args.StandardSave(nnx.state(model)),
                    optimizer=ocp.args.StandardSave(nnx.state(optimizer)),
                    monitor_state=ocp.args.JsonSave({
                        "ce_history": monitor.ce_history,
                        "best_ce": monitor.best_ce,
                        "best_loss": monitor.best_loss,
                        "best_avg_ce": monitor.best_avg_ce,
                        "last_improvement_step": monitor.last_improvement_step,
                    }),
                    step=ocp.args.JsonSave(step),
                ),
            )
            mngr.wait_until_finished()

        # CSV logging continues on the fixed interval
        if step % LOG_INTERVAL == 0:
            logger.log(step, step_loss, step_ce, step_p, step_forget_cost, t_total, step_compute_time, step_diag,
                       grad_norm_avg=grad_norm_avg, logit_drift=logit_drift, first_ce=first_ce)
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