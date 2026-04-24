import os

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
#os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax
import jax.numpy as jnp
import optax
from flax import nnx
import orbax.checkpoint as ocp
import time
from jax.sharding import NamedSharding, PartitionSpec
import threading
import queue
import multiprocessing as mp
from dotenv import load_dotenv
from train_local import (
    UniversalReasoner,
    LATENT_DIM,
    ACCUMULATION_STEPS,
    BATCH_SIZE,
    compute_grad_step,
    apply_grads,
)
from schedules import (
    learning_schedule,
    weight_decay_schedule,
)

from metrics_logger import LossMonitor, MetricsLogger

load_dotenv()

LOG_EVERY = 250
PREFETCH_SIZE = 128

def get_env_path(var_name, default=""):
    val = os.environ.get(var_name, default)
    if val.startswith(("gs://", "s3://", "https://")):
        return val
    return os.path.abspath(val)

DATA_ROOT = get_env_path("DATA_ROOT", "")
CHECKPOINT_ROOT = get_env_path("CHECKPOINT_ROOT", "orbax_checkpoints")

if not DATA_ROOT:
    print(f"⚠️ Warning: DATA_ROOT is not set. Data loading will likely fail unless provided via environment.")

from data_loaders import TextDataGenerator, DataMixer

devices = jax.devices()
num_devices = len(devices)
print(f"🌍 Found {num_devices} JAX devices: {devices}")

mesh = jax.sharding.Mesh(devices, ('fsdp',))
fsdp_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('fsdp'))
replicate_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

def shard_state(state):
    def apply_sharding(x):
        if not hasattr(x, "shape") or x.ndim == 0:
            return x
        
        if x.ndim >= 2:
            spec = jax.sharding.PartitionSpec(None, 'fsdp') 
        else:
            spec = jax.sharding.PartitionSpec() 
            
        sharding = jax.sharding.NamedSharding(mesh, spec)
        return jax.device_put(x, sharding)

    return jax.tree_util.tree_map(apply_sharding, state)

def init_model_and_optimizer():
    print(f"🚀 Initializing Dynamic Latent Reasoner (Dim={LATENT_DIM})...")
    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(42), batch_size=BATCH_SIZE * num_devices)
    optimizer = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)
    
    print(f"💎 Sharding model & optimizer across {num_devices} devices (FSDP)...")
    nnx.update(model, shard_state(nnx.state(model)))
    nnx.update(optimizer, shard_state(nnx.state(optimizer)))
    
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

def create_sft_optimizer(model, old_opt=None):
    print("📉 Recreating optimizer with 10x LR penalty for SFT phase...")
    sft_lr_schedule = lambda step: learning_schedule(step) * 0.1
    
    sft_chain = optax.MultiSteps(
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=sft_lr_schedule,
                weight_decay=weight_decay_schedule,
                mask=weight_decay_mask,
            ),
        ),
        every_k_schedule=ACCUMULATION_STEPS,
        use_grad_mean=True
    )
    new_opt = nnx.Optimizer(model, sft_chain, wrt=nnx.Param)
    if old_opt is not None:
        nnx.update(new_opt, nnx.state(old_opt))
    return new_opt

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

        nnx.update(model, shard_state(restored["model"]))
        
        graphdef, _ = nnx.split(optimizer)
        sharded_opt_state = shard_state(restored["optimizer"])
        optimizer = nnx.merge(graphdef, sharded_opt_state)
        
        start_step = restored["step"] + 1
        m_state = restored["monitor_state"]
        monitor.ce_history = m_state.get("ce_history", [])
        monitor.best_ce = m_state.get("best_ce", float("inf"))
        monitor.best_loss = m_state.get("best_loss", float("inf"))
        monitor.best_avg_ce = m_state.get("best_avg_ce", monitor.best_ce)
        monitor.last_improvement_step = m_state.get("last_improvement_step", 0)
        monitor.sft_start_step = m_state.get("sft_start_step", None)
        
        print(f"✅ Resuming from step {start_step} (Sharded)")
        del restored 
        import gc; gc.collect()
    else:
        print("🆕 No checkpoint found, starting from scratch...")
        start_step = 1
        monitor.sft_start_step = None

    return mngr, monitor, start_step

def setup_data_pipeline(start_step, sft_phase_event, sft_start_step=None):
    print("🚀 Initializing Dynamic Data Phases...")
    global_batch_size = BATCH_SIZE * num_devices
    print(f"📦 Global Batch Size: {global_batch_size} (Per-device: {BATCH_SIZE})")
    
    pretrain_sources = [
        TextDataGenerator(f"{DATA_ROOT}/pretrain/fineweb-edu"),
        TextDataGenerator(f"{DATA_ROOT}/pretrain/code_instructions"),
        TextDataGenerator(f"{DATA_ROOT}/pretrain/finemath"),
    ]
    pretrain_weights = [0.60, 0.25, 0.15]
    pretrain_mixer = DataMixer(pretrain_sources, pretrain_weights)
    
    sft_sources = [
        TextDataGenerator(f"{DATA_ROOT}/chat/ultrachat"),
        pretrain_sources[0],
        pretrain_sources[1],
        pretrain_sources[2],
    ]
    sft_weights = [0.70, 0.15, 0.10, 0.05]
    sft_mixer = DataMixer(sft_sources, sft_weights)

    if start_step > 1:
        if sft_start_step is None or start_step < sft_start_step:
            total_pretrain_seen = (start_step - 1) * ACCUMULATION_STEPS * num_devices
            for gen, weight in zip(pretrain_sources, pretrain_weights):
                gen.skip_count = int(total_pretrain_seen * weight)
        else:
            # 1. Catch up pretrain sources to the point where pretraining ended
            total_pre_pretrain_seen = (sft_start_step - 1) * ACCUMULATION_STEPS * num_devices
            for gen, weight in zip(pretrain_sources, pretrain_weights):
                gen.skip_count = int(total_pre_pretrain_seen * weight)
            
            # 2. Add SFT usage for all blended sources (Chat + Replay)
            total_sft_seen = (start_step - sft_start_step) * ACCUMULATION_STEPS * num_devices
            for gen, weight in zip(sft_sources, sft_weights):
                gen.skip_count += int(total_sft_seen * weight)

    data_queue = queue.Queue(maxsize=PREFETCH_SIZE)

    def data_wrapper():
        while True:
            if not sft_phase_event.is_set():
                res = pretrain_mixer.get_batch(global_batch_size)
            else:
                res = sft_mixer.get_batch(global_batch_size)
            
            if res[0] is None:
                data_queue.put((None, None))
                break
            
            data_queue.put(res)

    threading.Thread(target=data_wrapper, daemon=True).start()
    return data_queue

def train_loop(model, optimizer, data_queue, mngr, monitor, start_step, sft_phase_event):
    history_file = "training_history.csv"
    logger = MetricsLogger(history_file)
    step = start_step
    
    accum_loss = 0.0
    accum_token_loss = 0.0
    accum_forget_cost = 0.0
    accum_storage_cost = 0.0
    accum_grad_norm = 0.0
    t_compute = 0.0
    
    while True:
        batch, should_truncate = data_queue.get() 
        if batch is None: 
            break
        
        t_compute_start = time.time()
        
        data_s = NamedSharding(mesh, PartitionSpec('fsdp', None))
        
        batch_sharded = jax.device_put(batch, data_s)
        step_replicated = jax.device_put(jnp.array(step), replicate_sharding)
        truncate_replicated = jax.device_put(jnp.array(should_truncate), replicate_sharding)

        loss, out, grads, grad_norm = compute_grad_step(
            model, batch_sharded, step_replicated, should_truncate=truncate_replicated
        )
        
        apply_grads(optimizer, grads, model)
        
        t_compute += (time.time() - t_compute_start)

        accum_loss += loss / LOG_EVERY
        accum_token_loss += out.halt_diag.get('token_loss', loss) / LOG_EVERY
        accum_forget_cost += out.forget_cost / LOG_EVERY
        accum_storage_cost += out.storage_cost / LOG_EVERY
        accum_grad_norm += grad_norm / LOG_EVERY
            
        if (step + 1) % LOG_EVERY == 0:
            logger.log(
                step + 1, 
                float(accum_token_loss),
                float(accum_loss),
                out,
                t_compute / LOG_EVERY,
                grad_norm_avg=float(accum_grad_norm),
                first_ce=float(out.halt_diag.get('ce1', 0))
            )
            
            if monitor.push(step, float(accum_token_loss), float(accum_loss)): 
                if not sft_phase_event.is_set():
                    sft_phase_event.set()
                    monitor.sft_start_step = step
                    
                    # Apply 10x Learning Rate Penalty for SFT
                    optimizer = create_sft_optimizer(model, optimizer)
                    
                    print("\n" + "🔄"*30)
                    print("🔄 CE Plateau Detected! Triggering SFT Chat Phase and decaying Learning Rate!")
                    print("🔄"*30 + "\n")
                    
                    # Reset LossMonitor for fresh SFT baseline
                    monitor.ce_history = []
                    monitor.best_ce = float("inf")
                    monitor.best_loss = float("inf")
                    monitor.best_avg_ce = float("inf")
                    monitor.last_improvement_step = step
                else:
                    print("🛑 Training halted: No improvement in CE during SFT phase.")
                    break

            if monitor.is_new_best:
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
                            "sft_active": sft_phase_event.is_set(),
                            "sft_start_step": monitor.sft_start_step,
                        }),
                        step=ocp.args.JsonSave(step),
                    ),
                )
                mngr.wait_until_finished()
            
            accum_loss = 0.0
            accum_token_loss = 0.0
            accum_forget_cost = 0.0
            accum_storage_cost = 0.0
            accum_grad_norm = 0.0
            t_compute = 0.0

        step += 1

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    sft_phase_event = threading.Event()

    model, optimizer = init_model_and_optimizer()
    
    mngr, monitor, start_step = load_or_create_checkpoint(model, optimizer)
    
    if getattr(monitor, "sft_start_step", None) is not None:
        print(f"🔄 Resuming in SFT phase (started at step {monitor.sft_start_step})")
        sft_phase_event.set()
        optimizer = create_sft_optimizer(model, optimizer)

    data_queue = setup_data_pipeline(start_step, sft_phase_event, getattr(monitor, "sft_start_step", None))
    
    train_loop(model, optimizer, data_queue, mngr, monitor, start_step, sft_phase_event)