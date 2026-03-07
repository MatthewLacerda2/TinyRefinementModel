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
from dotenv import load_dotenv
from datasets import load_dataset
from train_local import (
    UniversalReasoner,
    train_step,
    optimizer_chain,
    LATENT_DIM, MAX_SEQ_LEN, BATCH_SIZE, ACCUMULATION_STEPS, PAD_TOKEN_ID, FORGET_LAMBDA, PONDER_LAMBDA
)

load_dotenv()

if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

CHECKPOINT_INTERVAL = 10
SORT_BUFFER_SIZE = 10_000
PREFETCH_SIZE = 4


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


class TextDataGenerator:
    def __init__(self, max_seq_len=MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len
        self.enc = tiktoken.get_encoding("cl100k_base")

        token = os.environ.get("HF_TOKEN")
        print(f"🚀 Preparing FineWeb-Edu (sorted by difficulty) (Auth: {'Yes' if token else 'No'})...")

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="default",
            split="train",
            streaming=True,
            token=token,
        ).select_columns(["text", "score"])

        self.iterator = self._curriculum_iterator(ds)
        self.exhausted = False

    def _curriculum_iterator(self, ds):
        buffer = []
        for example in ds:
            buffer.append(example)
            if len(buffer) >= SORT_BUFFER_SIZE:
                buffer.sort(key=lambda x: x["score"])
                yield from buffer
                buffer = []
        if buffer:
            buffer.sort(key=lambda x: x["score"])
            yield from buffer

    def get_batch(self, batch_size):
        if self.exhausted:
            return None
        batch_ids = []
        while len(batch_ids) < batch_size:
            try:
                item = next(self.iterator)
                tokens = self.enc.encode(item["text"])
                if len(tokens) < self.max_seq_len:
                    tokens = tokens + [PAD_TOKEN_ID] * (self.max_seq_len - len(tokens))
                else:
                    tokens = tokens[: self.max_seq_len]
                batch_ids.append(tokens)
            except StopIteration:
                self.exhausted = True
                break
            except Exception:
                continue
        if not batch_ids:
            return None
        return jnp.array(batch_ids, dtype=jnp.int32)


class LossMonitor:
    def __init__(self, patience=500, window=500, max_ponder_limit=15):
        self.patience = patience
        self.window = window
        self.max_ponder_limit = max_ponder_limit

        self.ce_history = []
        self.best_ce = float("inf")
        self.last_improvement_step = 0

    def push(self, step, ce_loss, avg_ponder):
        self.ce_history.append(ce_loss)
        if len(self.ce_history) > self.window:
            self.ce_history.pop(0)

        avg_ce = sum(self.ce_history) / len(self.ce_history)

        if avg_ce < (self.best_ce - 0.01):
            self.best_ce = avg_ce
            self.last_improvement_step = step
            return False

        if (step - self.last_improvement_step) > self.patience:
            print(f"\n🛑 Plateau detected: No CE improvement > 0.01 for {self.patience} steps.")
            return True

        return False


if __name__ == "__main__":
    print(f"🚀 Initializing Dynamic Latent Reasoner (Dim={LATENT_DIM})...")
    model = UniversalReasoner(LATENT_DIM, nnx.Rngs(42), dtype=jnp.float32)
    optimizer = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)

    data_gen = TextDataGenerator(MAX_SEQ_LEN)
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

    batch_queue = queue.Queue(maxsize=PREFETCH_SIZE)
    start_prefetch_worker(data_gen, BATCH_SIZE, batch_queue)

    step = start_step
    while True:
        t0 = time.time()
        # Accumulate as JAX arrays — no float() sync inside the loop
        step_loss = step_ce = step_p = step_forget_cost = 0.0
        step_diag = {k: 0.0 for k in ['logits_mean', 'logits_std', 'logits_min', 'logits_max', 'prob_mean', 'prob_std']}
        last_loss = None
        batch = None

        for i in range(ACCUMULATION_STEPS):
            batch = batch_queue.get()
            if batch is None:
                break

            loss, (ce, p, forget_cost, halt_diag) = train_step(
                model, optimizer, batch, step, FORGET_LAMBDA
            )
            for k in step_diag:
                step_diag[k] += halt_diag[k]

            # Accumulate without forcing device sync each iteration
            step_loss += loss
            step_ce += ce
            step_p += p
            step_forget_cost += forget_cost
            last_loss = loss

        if batch is None:
            break

        # Single sync point for the whole macro-step
        if last_loss is not None:
            last_loss.block_until_ready()

        step_loss = float(step_loss) / ACCUMULATION_STEPS
        step_ce = float(step_ce) / ACCUMULATION_STEPS
        step_p = float(step_p) / ACCUMULATION_STEPS
        step_forget_cost = float(step_forget_cost) / ACCUMULATION_STEPS
        step_diag = {k: float(v) / ACCUMULATION_STEPS for k, v in step_diag.items()}

        t_total = time.time() - t0

        if monitor.push(step, step_ce, step_p):
            break

        if step % CHECKPOINT_INTERVAL == 0:
            mngr.save(
                step,
                args=ocp.args.Composite(
                    model=ocp.args.StandardSave(nnx.state(model)),
                    optimizer=ocp.args.StandardSave(nnx.state(optimizer)),
                    monitor_state=ocp.args.JsonSave(
                        {
                            "ce_history": monitor.ce_history,
                            "best_ce": monitor.best_ce,
                            "last_improvement_step": monitor.last_improvement_step,
                        }
                    ),
                    step=ocp.args.JsonSave(step),
                ),
            )
            mngr.wait_until_finished()

            print(
                f"Step {step:04d} | CE: {step_ce:.4f} | Agg Loss: {step_loss:.4f} | "
                f"Avg Steps: {step_p:.2f} | Forget: {step_forget_cost:.4f} | Time: {t_total:.2f}s"
            )
            print(
                f"      Logits [μ:{step_diag['logits_mean']:.2f}, σ:{step_diag['logits_std']:.2f}, min:{step_diag['logits_min']:.2f}, max:{step_diag['logits_max']:.2f}] | "
                f"Prob [μ:{step_diag['prob_mean']:.3f}, σ:{step_diag['prob_std']:.3f}]"
            )

            write_header = not os.path.exists(history_file) or os.path.getsize(history_file) == 0
            with open(history_file, "a", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["step", "loss", "ce", "avg_ponder", "avg_forget_cost", "t_total"]
                )
                if write_header:
                    writer.writeheader()
                writer.writerow(
                    {
                        "step": int(step),
                        "loss": f"{step_loss:.4f}",
                        "ce": f"{step_ce:.4f}",
                        "avg_ponder": f"{step_p:.2f}",
                        "avg_forget_cost": f"{step_forget_cost:.4f}",
                        "t_total": f"{t_total:.2f}",
                    }
                )

        step += 1