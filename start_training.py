import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
import csv
import time
import tiktoken
import os
from dotenv import load_dotenv
from datasets import load_dataset
from train_local import (
    UniversalReasoner,
    train_step,
    optimizer_chain,
    LATENT_DIM, MAX_SEQ_LEN, BATCH_SIZE, ACCUMULATION_STEPS, PAD_TOKEN_ID
)

load_dotenv()

if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

CHECKPOINT_INTERVAL = 10

# How many examples to buffer before sorting by difficulty score.
# Larger = better global ordering but more RAM. 10k is a good balance.
SORT_BUFFER_SIZE = 10_000


class TextDataGenerator:
    def __init__(self, max_seq_len=MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len
        self.enc = tiktoken.get_encoding("cl100k_base")

        token = os.environ.get("HF_TOKEN")
        print(f"🚀 Preparing FineWeb-Edu (sorted by difficulty) (Auth: {'Yes' if token else 'No'})...")

        # HuggingFaceFW/fineweb-edu contains an 'score' column (1-5, educational quality).
        # We use it as a difficulty proxy: low score = simple/easy, high score = complex/hard.
        # Streaming doesn't allow a global sort, so we do chunk-level curriculum sorting:
        # buffer SORT_BUFFER_SIZE examples, sort ascending by score, then yield in order.
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
        """Buffer chunks of examples and yield them sorted by score (easy → hard)."""
        buffer = []
        for example in ds:
            buffer.append(example)
            if len(buffer) >= SORT_BUFFER_SIZE:
                buffer.sort(key=lambda x: x["score"])
                yield from buffer
                buffer = []
        # Flush remaining
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
    def __init__(self, patience=2000, window=500, max_ponder_limit=14):
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
            return False

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

    step = start_step
    while True:
        t0 = time.time()
        step_loss, step_ce, step_p, step_t_cost = 0.0, 0.0, 0.0, 0.0

        for i in range(ACCUMULATION_STEPS):
            batch = data_gen.get_batch(BATCH_SIZE)
            if batch is None:
                break

            loss, (ce, p, t_cost) = train_step(model, optimizer, batch)

            loss.block_until_ready()

            step_loss += float(loss)
            step_ce += float(ce)
            step_p += float(p)
            step_t_cost += float(t_cost)

        if batch is None:
            break

        step_ce /= ACCUMULATION_STEPS
        step_p /= ACCUMULATION_STEPS
        step_t_cost /= ACCUMULATION_STEPS

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
                f"Step {step:04d} | Agg Loss: {step_loss:.4f} | Avg Steps: {step_p:.2f} | T-Cost: {step_t_cost:.4f} | Time: {t_total:.2f}s"
            )

            with open(history_file, "a", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["step", "loss", "ce", "avg_ponder", "avg_t_cost", "t_total"]
                )
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(
                    {
                        "step": int(step),
                        "loss": f"{step_loss:.4f}",
                        "ce": f"{step_ce:.4f}",
                        "avg_ponder": f"{step_p:.2f}",
                        "avg_t_cost": f"{step_t_cost:.4f}",
                        "t_total": f"{t_total:.2f}",
                    }
                )

        step += 1