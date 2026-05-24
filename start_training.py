import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

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
from layers import (
    LATENT_DIM,
    NUM_BLOCKS,
    SHARED_SLOTS,
    MAX_SEQ_LEN,
    VOCAB_SIZE,
    MAX_STEPS_LIMIT,
    BATCH_SIZE,
    ACCUMULATION_STEPS,
    PAD_TOKEN_ID,
    NUM_HEADS,
    NUM_GROUPS,
)
from model import UniversalReasoner
from train_local import (
    compute_grad_step,
    apply_grads,
)
from schedules import (
    learning_schedule,
    weight_decay_schedule,
)

from metrics_logger import LossMonitor, MetricsLogger
import json
import datetime
import subprocess
import sys

class RunTracker:
    def __init__(self, runs_root="runs"):
        self.runs_root = runs_root
        self.run_id = None
        self.run_dir = None
        self.start_time = None
        self.session_index = None

    @staticmethod
    def get_git_metadata():
        metadata = {
            "commit": "unknown",
            "branch": "unknown",
            "dirty": False
        }
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            metadata["commit"] = commit
        except Exception:
            pass

        try:
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            metadata["branch"] = branch
        except Exception:
            pass

        try:
            status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
            metadata["dirty"] = len(status) > 0
        except Exception:
            pass

        return metadata

    @staticmethod
    def get_hyperparameters():
        return {
            "LATENT_DIM": LATENT_DIM,
            "NUM_BLOCKS": NUM_BLOCKS,
            "SHARED_SLOTS": SHARED_SLOTS,
            "MAX_SEQ_LEN": MAX_SEQ_LEN,
            "VOCAB_SIZE": VOCAB_SIZE,
            "MAX_STEPS_LIMIT": MAX_STEPS_LIMIT,
            "BATCH_SIZE": BATCH_SIZE,
            "ACCUMULATION_STEPS": ACCUMULATION_STEPS,
            "PAD_TOKEN_ID": PAD_TOKEN_ID,
            "NUM_HEADS": NUM_HEADS,
            "NUM_GROUPS": NUM_GROUPS,
        }

    def _check_compatibility(self, metadata_path):
        if not os.path.exists(metadata_path):
            return
        
        try:
            with open(metadata_path, "r") as f:
                old_meta = json.load(f)
            
            old_params = old_meta.get("parameters", {})
            current_params = self.get_hyperparameters()
            
            critical_keys = [
                "LATENT_DIM", "NUM_BLOCKS", "SHARED_SLOTS", "MAX_SEQ_LEN", 
                "VOCAB_SIZE", "NUM_HEADS", "NUM_GROUPS"
            ]
            mismatches = [
                f"  - {k}: run used {old_params[k]}, current code uses {current_params[k]}"
                for k in critical_keys
                if k in old_params and old_params[k] != current_params[k]
            ]
            
            if mismatches:
                print("\n" + "🛑"*20)
                print("🛑 ERROR: Parameter Mismatch Detected! Cannot resume this training run:")
                print("\n".join(mismatches))
                print("\n💡 Options:")
                print("  1. Revert your code parameters back to match the run's parameters.")
                print("  2. Start a brand new training run with: python start_training.py --new-run")
                print("  3. Point to a different checkpoint folder with: python start_training.py --checkpoint-path <path>")
                print("🛑"*20 + "\n")
                sys.exit(1)
        except SystemExit:
            sys.exit(1)
        except Exception:
            pass

    def start_session(self, run_id=None):
        os.makedirs(self.runs_root, exist_ok=True)
        self.start_time = time.time()
        start_timestamp = datetime.datetime.now().astimezone().isoformat()

        if run_id is None:
            # Generate a new unique run ID
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"run_{timestamp_str}"
            self.run_dir = os.path.join(self.runs_root, self.run_id)
            os.makedirs(self.run_dir, exist_ok=True)

            git_meta = self.get_git_metadata()
            params = self.get_hyperparameters()

            metadata = {
                "run_id": self.run_id,
                "git_commit": git_meta["commit"],
                "git_branch": git_meta["branch"],
                "git_dirty": git_meta["dirty"],
                "parameters": params,
                "sections": [
                    {
                        "start_time": start_timestamp,
                        "end_time": None,
                        "duration_seconds": None
                    }
                ]
            }
            self.session_index = 0
            self.save_metadata(metadata)
            print(f"📁 Created new training run folder: {self.run_dir}")
        else:
            # Resume existing run
            self.run_id = run_id
            self.run_dir = os.path.join(self.runs_root, self.run_id)
            os.makedirs(self.run_dir, exist_ok=True)

            metadata_path = os.path.join(self.run_dir, "run_metadata.json")
            self._check_compatibility(metadata_path)

            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                except Exception:
                    metadata = None
            else:
                metadata = None

            if metadata is None:
                git_meta = self.get_git_metadata()
                params = self.get_hyperparameters()
                metadata = {
                    "run_id": self.run_id,
                    "git_commit": git_meta["commit"],
                    "git_branch": git_meta["branch"],
                    "git_dirty": git_meta["dirty"],
                    "parameters": params,
                    "sections": []
                }

            metadata["sections"].append({
                "start_time": start_timestamp,
                "end_time": None,
                "duration_seconds": None
            })
            self.session_index = len(metadata["sections"]) - 1
            self.save_metadata(metadata)
            print(f"🔄 Resumed training run folder: {self.run_dir}")

        return self.run_id

    def update_session_duration(self):
        if self.run_dir is None or self.session_index is None:
            return
        metadata_path = os.path.join(self.run_dir, "run_metadata.json")
        if not os.path.exists(metadata_path):
            return
        
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            end_timestamp = datetime.datetime.now().astimezone().isoformat()
            duration = time.time() - self.start_time
            
            metadata["sections"][self.session_index]["end_time"] = end_timestamp
            metadata["sections"][self.session_index]["duration_seconds"] = round(duration, 2)
            
            self.save_metadata(metadata)
        except Exception:
            pass

    def save_metadata(self, metadata):
        metadata_path = os.path.join(self.run_dir, "run_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

load_dotenv()

LOG_REAL_STEPS = 5
PREFETCH_SIZE = 128

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

def create_sft_optimizer(model, old_state=None):
    import gc
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
    if old_state is not None:
        nnx.update(new_opt, old_state)
        
    gc.collect()
    return new_opt

def discover_latest_checkpoint_run(runs_root="runs"):
    if not os.path.exists(runs_root):
        return None, None
    
    import glob
    run_dirs = sorted(glob.glob(os.path.join(runs_root, "run_*")))
    
    for r_dir in reversed(run_dirs):
        chk_dir = os.path.join(r_dir, "checkpoints")
        if os.path.exists(chk_dir):
            try:
                mngr = ocp.CheckpointManager(
                    chk_dir,
                    item_names=("model", "optimizer", "monitor_state", "step"),
                )
                if mngr.latest_step() is not None:
                    run_id = os.path.basename(r_dir)
                    return chk_dir, run_id
            except Exception:
                pass
    return None, None

def load_or_create_checkpoint(model, optimizer, checkpoint_path, force_new_run=False):
    monitor = LossMonitor()
    mngr = ocp.CheckpointManager(
        checkpoint_path,
        item_names=("model", "optimizer", "monitor_state", "step"),
        options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True),
    )

    run_id = None
    if not force_new_run and mngr.latest_step() is not None:
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
        monitor.ce_history = m_state.get("ce_history", [])
        monitor.best_ce = m_state.get("best_ce", float("inf"))
        monitor.best_loss = m_state.get("best_loss", float("inf"))
        monitor.best_avg_ce = m_state.get("best_avg_ce", monitor.best_ce)
        monitor.last_improvement_step = m_state.get("last_improvement_step", 0)
        monitor.sft_start_step = m_state.get("sft_start_step", None)
        run_id = m_state.get("run_id", None)
        
        print(f"✅ Resuming from step {start_step}")
        del restored 
        import gc; gc.collect()
    else:
        if force_new_run:
            print("🆕 Force New Run specified, starting from scratch...")
        else:
            print("🆕 No checkpoint found, starting from scratch...")
        start_step = 1
        monitor.sft_start_step = None

    return mngr, monitor, start_step, optimizer, run_id

def setup_data_pipeline(start_step, sft_phase_event, sft_start_step=None):
    print("🚀 Initializing Dynamic Data Phases...")
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
            total_pretrain_seen = (start_step - 1)
            for gen, weight in zip(pretrain_sources, pretrain_weights):
                gen.skip_count = int(total_pretrain_seen * weight)
        else:
            # 1. Catch up pretrain sources to the point where pretraining ended
            total_pre_pretrain_seen = (sft_start_step - 1)
            for gen, weight in zip(pretrain_sources, pretrain_weights):
                gen.skip_count = int(total_pre_pretrain_seen * weight)
            
            # 2. Add SFT usage for all blended sources (Chat + Replay)
            total_sft_seen = (start_step - sft_start_step)
            for gen, weight in zip(sft_sources, sft_weights):
                gen.skip_count += int(total_sft_seen * weight)

    data_queue = queue.Queue(maxsize=PREFETCH_SIZE)

    def data_wrapper():
        while True:
            if not sft_phase_event.is_set():
                res = pretrain_mixer.get_batch(BATCH_SIZE)
            else:
                res = sft_mixer.get_batch(BATCH_SIZE)
            
            if res[0] is None:
                data_queue.put((None, None))
                break
            
            data_queue.put(res)

    threading.Thread(target=data_wrapper, daemon=True).start()
    return data_queue

def train_loop(model, optimizer, data_queue, mngr, monitor, start_step, sft_phase_event, run_tracker):
    history_file = os.path.join(run_tracker.run_dir, "metrics.csv")
    logger = MetricsLogger(history_file)
    step = start_step
    
    accum_loss = 0.0
    accum_token_loss = 0.0
    accum_forget_cost = 0.0
    accum_grad_norm = 0.0
    t_compute = 0.0
    
    try:
        while True:
            batch, should_truncate = data_queue.get() 
            if batch is None: 
                break
            
            t_compute_start = time.time()
            
            loss, out, grads, grad_norm = compute_grad_step(
                model, batch, jnp.array(step), should_truncate=should_truncate
            )
            
            apply_grads(optimizer, grads, model)
            
            t_compute += (time.time() - t_compute_start)

            current_loss = float(loss)
            current_token_loss = float(out.halt_diag.get('token_loss', loss))
            current_forget = float(out.forget_cost)
            current_grad_norm = float(grad_norm)

            divisor = ACCUMULATION_STEPS * LOG_REAL_STEPS
            accum_loss += current_loss / divisor
            accum_token_loss += current_token_loss / divisor
            accum_forget_cost += current_forget / divisor
            accum_grad_norm += current_grad_norm / divisor
                
            if (step + 1) % (ACCUMULATION_STEPS * LOG_REAL_STEPS) == 0:
                opt_step = (step + 1) // ACCUMULATION_STEPS
                
                logger.log(
                    opt_step, 
                    float(accum_token_loss),
                    float(accum_loss),
                    out,
                    t_compute,
                    grad_norm_avg=float(accum_grad_norm),
                    first_ce=float(out.halt_diag.get('ce1', 0))
                )
                
                # Periodically update session duration to capture active timings
                run_tracker.update_session_duration()
                
                if monitor.push(opt_step, float(accum_token_loss), float(accum_loss)): 
                    if not sft_phase_event.is_set():
                        sft_phase_event.set()
                        monitor.sft_start_step = step
                        
                        import gc
                        old_state = nnx.state(optimizer)
                        del optimizer
                        gc.collect()

                        optimizer = create_sft_optimizer(model, old_state)
                        del old_state
                        gc.collect()
                        
                        print("\n" + "🔄"*30)
                        print("🔄 CE Plateau Detected! Triggering SFT Chat Phase and decaying Learning Rate!")
                        print("🔄"*30 + "\n")
                        
                        monitor.ce_history = []
                        monitor.best_ce = float("inf")
                        monitor.best_loss = float("inf")
                        monitor.best_avg_ce = float("inf")
                        monitor.last_improvement_step = opt_step
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
                                "run_id": run_tracker.run_id,  # Save run_id inside checkpoint metadata
                            }),
                            step=ocp.args.JsonSave(step),
                        ),
                    )
                    mngr.wait_until_finished()
                
                accum_loss = 0.0
                accum_token_loss = 0.0
                accum_forget_cost = 0.0
                accum_grad_norm = 0.0
                t_compute = 0.0

            step += 1
    finally:
        # Guarantee run metadata is finalized on exit
        run_tracker.update_session_duration()

if __name__ == "__main__":
    import argparse

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Train the Dynamic Latent Reasoner")
    parser.add_argument("--new-run", action="store_true", help="Force starting a brand new training run from scratch (ignores existing checkpoints)")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Custom folder for Orbax checkpoints")
    args = parser.parse_args()

    # 1. Resolve checkpoint path and run_id to resume (if any)
    checkpoint_run_id = None
    active_checkpoint_path = None
    
    if args.checkpoint_path is not None:
        active_checkpoint_path = os.path.abspath(args.checkpoint_path)
        # Try to extract run_id from path if it follows runs/run_xxx/checkpoints
        parts = active_checkpoint_path.split(os.sep)
        for part in parts:
            if part.startswith("run_"):
                checkpoint_run_id = part
                break
        print(f"📁 Using custom checkpoint path: {active_checkpoint_path}")
    elif not args.new_run:
        # Auto-discover the latest checkpointed run
        discovered_path, discovered_run_id = discover_latest_checkpoint_run()
        if discovered_path is not None:
            active_checkpoint_path = discovered_path
            checkpoint_run_id = discovered_run_id
            print(f"🔎 Auto-discovered latest checkpointed run: {checkpoint_run_id}")
    
    # 2. Start/Resume Run Tracker session
    run_tracker = RunTracker()
    if checkpoint_run_id is None:
        # Starting a brand new run
        run_tracker.start_session(run_id=None)
        if active_checkpoint_path is None:
            active_checkpoint_path = os.path.join(run_tracker.run_dir, "checkpoints")
    else:
        # Resuming existing run
        run_tracker.start_session(run_id=checkpoint_run_id)
        if active_checkpoint_path is None:
            active_checkpoint_path = os.path.join(run_tracker.run_dir, "checkpoints")

    sft_phase_event = threading.Event()

    model, optimizer = init_model_and_optimizer()
    
    mngr, monitor, start_step, optimizer, checkpoint_run_id_from_meta = load_or_create_checkpoint(
        model, optimizer, active_checkpoint_path, force_new_run=args.new_run
    )
    
    # Set event if resuming in SFT phase
    if getattr(monitor, "sft_start_step", None) is not None:
        print(f"🔄 Resuming in SFT phase (started at step {monitor.sft_start_step})")
        sft_phase_event.set()
        
        import gc
        old_state = nnx.state(optimizer)
        del optimizer
        gc.collect()
        
        optimizer = create_sft_optimizer(model, old_state)
        del old_state
        gc.collect()

    data_queue = setup_data_pipeline(start_step, sft_phase_event, getattr(monitor, "sft_start_step", None))
    
    train_loop(model, optimizer, data_queue, mngr, monitor, start_step, sft_phase_event, run_tracker)