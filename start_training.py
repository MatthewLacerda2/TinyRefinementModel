import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax
import jax.numpy as jnp
import optax
from flax import nnx
import orbax.checkpoint as ocp
import time
import threading
import queue
import multiprocessing as mp
from dotenv import load_dotenv
from train_local import (
    UniversalReasoner,
    LATENT_DIM,
    MAX_STEPS_LIMIT,
    PAD_TOKEN_ID,
    ACCUMULATION_STEPS,
    HUNCH_REFRESH_EVERY,
    SHARED_SLOTS,
    BATCH_SIZE,
    compute_grad_step,
    apply_grads,
)
from schedules import (
    learning_schedule,
    weight_decay_schedule,
    ponder_lambda_schedule,
    forget_lambda_schedule,
    storage_lambda_schedule
)

from metrics_logger import LossMonitor, MetricsLogger

load_dotenv()

LOG_EVERY = 10
CHECKPOINT_INTERVAL = 20
SORT_BUFFER_SIZE = 1000
PREFETCH_SIZE = 128
PHASE_STEP = 1000

DATA_ROOT = os.path.abspath(os.environ.get("DATA_ROOT", ""))
CHECKPOINT_ROOT = os.path.abspath(os.environ.get("CHECKPOINT_ROOT", "orbax_checkpoints"))

if not DATA_ROOT:
    print(f"⚠️ Warning: DATA_ROOT is not set. Data loading will likely fail unless provided via environment.")

from data_loaders import TextDataGenerator, DataMixer

def init_model_and_optimizer():
    print(f"🚀 Initializing Dynamic Latent Reasoner (Dim={LATENT_DIM})...")
    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(42))
    optimizer = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)
    
    return model, optimizer

def weight_decay_mask(params):
    return jax.tree_util.tree_map(lambda x: x.ndim >= 2, params)

optimizer_chain = optax.MultiSteps(
    optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=learning_schedule,
            weight_decay=weight_decay_schedule,
            mask=weight_decay_mask,
        ),
    ),
    every_k_schedule=ACCUMULATION_STEPS,
    use_grad_mean=True
)

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
            total_pretrain_seen = (start_step - 1) * ACCUMULATION_STEPS * 1 # BATCH_SIZE
            weights = [0.60, 0.25, 0.15]
            for gen, weight in zip(pretrain_sources, weights):
                gen.skip_count = int(total_pretrain_seen * weight)
        else:
            total_chat_seen = (start_step - PHASE_STEP) * ACCUMULATION_STEPS * 1
            chat_mixer.skip_count = total_chat_seen

    data_queue = queue.Queue(maxsize=PREFETCH_SIZE)

    def data_wrapper():
        macro_step = start_step
        micro_counter = 0
        
        while True:
            if macro_step < PHASE_STEP:
                res = pretrain_mixer.get_batch(BATCH_SIZE)
            else:
                res = chat_mixer.get_batch(BATCH_SIZE)
            
            if res[0] is None:
                data_queue.put((None, None))
                break
            
            data_queue.put(res)
            
            micro_counter += 1
            if micro_counter % ACCUMULATION_STEPS == 0:
                macro_step += 1

    threading.Thread(target=data_wrapper, daemon=True).start()
    return data_queue

def train_loop(model, optimizer, data_queue, mngr, monitor, start_step):
    history_file = f"{CHECKPOINT_ROOT}/training_history.csv"
    logger = MetricsLogger(history_file)
    step = start_step
    
    accum_loss = 0.0
    accum_token_loss = 0.0
    accum_p = 0.0
    accum_forget_cost = 0.0
    accum_storage_cost = 0.0
    accum_grad_norm = 0.0
    t0_batch = time.time()
    t_compute = 0.0
    
    while True:
        t_data_start = time.time()
        batch, should_truncate = data_queue.get() 
        if batch is None: 
            break
        
        t_compute_start = time.time()
        
        loss, out, grads, grad_norm = compute_grad_step(
            model, batch, jnp.array(step), should_truncate=should_truncate
        )
        
        apply_grads(optimizer, grads, model)
        
        t_compute += (time.time() - t_compute_start)

        accum_loss += loss / LOG_EVERY
        accum_token_loss += out.halt_diag.get('token_loss', loss) / LOG_EVERY
        accum_p += out.ponder_cost / LOG_EVERY
        accum_forget_cost += out.forget_cost / LOG_EVERY
        accum_storage_cost += out.storage_cost / LOG_EVERY
        accum_grad_norm += grad_norm / LOG_EVERY
            
        if (step + 1) % LOG_EVERY == 0:
            t_total = time.time() - t0_batch
            logger.log(
                step + 1, 
                float(accum_token_loss),
                out,
                t_total, 
                t_compute / LOG_EVERY,
                grad_norm_avg=float(accum_grad_norm),
                logit_drift=float(out.halt_diag.get('temporal_drift', 0)),
                first_ce=float(out.halt_diag.get('first_ce', 0))
            )
            
            if monitor.push(step, float(accum_token_loss), float(accum_loss)): 
                print("🛑 Training halted: No improvement in CE.")
                break

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
            
            accum_loss = 0.0
            accum_token_loss = 0.0
            accum_p = 0.0
            accum_forget_cost = 0.0
            accum_storage_cost = 0.0
            accum_grad_norm = 0.0
            t0_batch = time.time()
            t_compute = 0.0

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