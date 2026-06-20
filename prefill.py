import os
import numpy as np
import tiktoken
import sys
from datasets import load_dataset
from multiprocessing import Pool, cpu_count
import json
import glob
import time
import threading
import queue
from config import MAX_SEQ_LEN, TOKENIZER_NAME, resolve_root
from dotenv import load_dotenv

# Load environment variables (such as HF_TOKEN) before datasets loads
load_dotenv()

# Config
ENC_NAME = TOKENIZER_NAME
OUTPUT_DIR = resolve_root(os.environ.get("DATA_ROOT", "runs/data"))
TOKENS_PER_FILE = 125_000_000  # ~500MB per chunk
PREFETCH_BUFFER = 15000        # raw text records buffered ahead of tokenization
TOKENIZE_BATCH_ITEMS = 4000    # records per parallel tokenization round
FINEWEB_MIN_SCORE = 4.0        # educational-score floor. Raised 3.0→4.0: the model is
                               # tiny (~78M), so we trade volume for per-token quality
                               # density (the phi / TinyStories regime). FineWeb-Edu's
                               # released set is already ≥3; ≥4 is far denser yet still
                               # leaves 100B+ tokens — ample for our 4.5B target.

# Targets: 8.05B total for high-performance SOTA corpus
MIXTURE = [
    {
        "path": "HuggingFaceFW/fineweb-edu",
        "target_tokens": 4_500_000_000,
        "folder": "pretrain",
        "alias": "fineweb-edu"
    },
    {
        "path": "codeparrot/codeparrot-clean",
        "target_tokens": 2_000_000_000,
        "folder": "pretrain",
        "alias": "codeparrot"
    },
    {
        "path": "HuggingFaceTB/finemath",
        "config": "finemath-4plus",
        "target_tokens": 1_200_000_000,
        "folder": "pretrain",
        "alias": "finemath"
    },
    {
        "path": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "target_tokens": 350_000_000,
        "folder": "chat",
        "alias": "ultrachat"
    }
]


def tokenize_batch_parallel(text):
    """Worker function for parallel tokenization."""
    enc = tiktoken.get_encoding(ENC_NAME)
    return enc.encode(text, allowed_special="all") + [enc.eot_token]


def extract_text(item, alias):
    """Per-dataset text extraction. Returns None for filtered or empty items."""
    if alias == "fineweb-edu":
        score = item.get("score", item.get("educational_score", 0.0))
        if score < FINEWEB_MIN_SCORE:
            return None

    if alias == "ultrachat" and "messages" in item:
        msg_list = item["messages"]
        if isinstance(msg_list, list):
            return "\n\n".join(
                f"{msg.get('role', 'unknown').capitalize()}: {msg.get('content', '')}"
                for msg in msg_list
            )
        return str(msg_list)

    txt = item.get("text") or item.get("content") or item.get("prompt")

    if not txt and "data" in item:
        if isinstance(item["data"], list):
            txt = "\n".join(str(x) for x in item["data"])
        else:
            txt = str(item["data"])

    return txt or None


def load_progress(save_path, name):
    """Returns (file_idx, total_tokens, items_processed) for a dataset, reading
    status.json or — recovery mode — scanning existing chunk files."""
    status_file = os.path.join(save_path, "status.json")
    file_idx = 0
    total_tokens = 0
    items_processed = 0

    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status = json.load(f)
            file_idx = status.get("file_idx", 0)
            total_tokens = status.get("total_tokens", 0)
            items_processed = status.get("items_processed", 0)
        print(f"🔄 Resuming {name} from status.json: item {items_processed:,} (Tokens: {total_tokens/1e6:.1f}M, Chunk: {file_idx})")
        return file_idx, total_tokens, items_processed

    existing_chunks = glob.glob(os.path.join(save_path, "chunk_*.npy"))
    if existing_chunks:
        try:
            indices = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing_chunks]
            file_idx = max(indices) + 1

            for f in existing_chunks:
                data = np.load(f, mmap_mode='r')
                total_tokens += len(data)

            print(f"🔎 Auto-discovered progress for {name}: {total_tokens/1e6:.1f}M tokens in {len(existing_chunks)} chunks.")
            print(f"⚠️ Note: Starting stream from beginning as row count is unknown (Next chunk: {file_idx})")
        except (OSError, ValueError) as e:
            print(f"⚠️ Could not recover progress for {name}: {e}")
            file_idx = 0
            total_tokens = 0

    return file_idx, total_tokens, items_processed


def save_progress(save_path, file_idx, total_tokens, items_processed):
    with open(os.path.join(save_path, "status.json"), 'w') as f:
        json.dump({
            "file_idx": file_idx,
            "total_tokens": total_tokens,
            "items_processed": items_processed
        }, f)


def stream_with_retries(ds_cfg, name, start_offset, out_queue, stop_event):
    """Producer: streams raw text records into out_queue, reconnecting on network
    failures and resuming from the last consumed item. Ends with a None sentinel."""
    split_name = ds_cfg.get('split', 'train')

    def open_stream(offset):
        ds = load_dataset(ds_cfg['path'], name=ds_cfg.get('config'), split=split_name, streaming=True)
        return ds.skip(offset) if offset > 0 else ds

    current_offset = start_offset
    active_ds = open_stream(current_offset)

    while not stop_event.is_set():
        consumed = 0
        try:
            for item in active_ds:
                if stop_event.is_set():
                    break
                consumed += 1
                txt = extract_text(item, name)
                if txt:
                    out_queue.put(txt)
            # Natural exit means dataset is completed
            break
        except Exception as e:
            # Catch connection timeouts, name resolution failures, or closed HTTPX client errors
            current_offset += consumed
            t_now = time.strftime('%H:%M:%S', time.localtime())
            print(f"\n[{t_now}] 🔌 Prefetcher connection dropped: {e}")
            print(f"[{t_now}] 🔄 Re-connecting and resuming dataset from item {current_offset:,} in 5 seconds...")
            time.sleep(5)

            try:
                active_ds = open_stream(current_offset)
            except Exception as conn_err:
                print(f"\n⚠️ Re-connection failed: {conn_err}. Will retry shortly...")

    out_queue.put(None)


def tokenize_buffer(pool, buffer):
    """Tokenizes a list of raw texts in parallel into one flat int32 array."""
    token_lists = pool.map(tokenize_batch_parallel, buffer)
    return np.array([t for sub in token_lists for t in sub], dtype=np.int32)


def write_chunk(save_path, file_idx, token_acc):
    """Saves accumulated tokens as a stride-aligned chunk. Returns the next chunk
    index and the unaligned remainder (carried into the next chunk)."""
    chunk_data = np.concatenate(token_acc)
    stride = 2 * MAX_SEQ_LEN + 1
    valid_len = (len(chunk_data) // stride) * stride
    if valid_len == 0:
        return file_idx, token_acc

    t_save = time.strftime('%H:%M:%S', time.localtime())
    np.save(os.path.join(save_path, f"chunk_{file_idx}.npy"), chunk_data[:valid_len])
    print(f"\n[{t_save}] 💾 Saved chunk_{file_idx}.npy ({valid_len/1e6:.1f}M tokens)")

    remainder = chunk_data[valid_len:]
    return file_idx + 1, [remainder] if len(remainder) > 0 else []


def process_dataset(pool, ds_cfg):
    name = ds_cfg.get('alias') or ds_cfg['path'].split('/')[-1]
    target = ds_cfg['target_tokens']
    save_path = os.path.join(OUTPUT_DIR, ds_cfg['folder'], name)
    os.makedirs(save_path, exist_ok=True)

    print(f"\n🚀 Processing {name} | Target: {target/1e9:.2f}B tokens")

    file_idx, total_tokens_ds, items_processed = load_progress(save_path, name)
    if total_tokens_ds >= target:
        print(f"⏩ {name} already completed. Skipping.")
        return

    prefetch_queue = queue.Queue(maxsize=PREFETCH_BUFFER)
    stop_event = threading.Event()
    threading.Thread(
        target=stream_with_retries,
        args=(ds_cfg, name, items_processed, prefetch_queue, stop_event),
        daemon=True,
    ).start()

    buffer = []
    token_acc = []
    t_start = time.time()
    initial_tokens = total_tokens_ds

    try:
        while True:
            txt = prefetch_queue.get()
            if txt is None:  # Sentinel: stream completed
                break

            buffer.append(txt)
            if len(buffer) < TOKENIZE_BATCH_ITEMS:
                continue

            flat_batch = tokenize_buffer(pool, buffer)
            token_acc.append(flat_batch)
            total_tokens_ds += len(flat_batch)
            items_processed += len(buffer)
            buffer = []

            elapsed = time.time() - t_start
            tokens_sec = (total_tokens_ds - initial_tokens) / max(1e-3, elapsed)
            progress = (total_tokens_ds / target) * 100
            sys.stdout.write(
                f"\rProgress: {total_tokens_ds/1e6:.1f}M/{target/1e6:.0f}M tokens ({progress:.1f}%) | "
                f"Speed: {tokens_sec/1e3:.1f}k tok/s | Elapsed: {elapsed/60:.1f}m"
            )
            sys.stdout.flush()

            if sum(len(x) for x in token_acc) >= TOKENS_PER_FILE:
                file_idx, token_acc = write_chunk(save_path, file_idx, token_acc)
                save_progress(save_path, file_idx, total_tokens_ds, items_processed)

            if total_tokens_ds >= target:
                print(f"\n✅ {name} target reached.")
                break
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Cleaning up background threads...")
        stop_event.set()
        # Empty queue to unblock the producer thread
        while not prefetch_queue.empty():
            try:
                prefetch_queue.get_nowait()
            except queue.Empty:
                break
        raise
    finally:
        stop_event.set()

    # Final flush for this dataset
    if (token_acc or buffer) and total_tokens_ds < target:
        if buffer:
            flat_batch = tokenize_buffer(pool, buffer)
            token_acc.append(flat_batch)
            total_tokens_ds += len(flat_batch)
            items_processed += len(buffer)
        if token_acc:
            file_idx, _ = write_chunk(save_path, file_idx, token_acc)
            save_progress(save_path, file_idx, total_tokens_ds, items_processed)
            print(f"\n🏁 Finished {name}. Total: {total_tokens_ds/1e9:.2f}B tokens")


def run_prefill():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Leave one core free to keep the system responsive
    num_workers = max(1, int(cpu_count() - 1))

    with Pool(num_workers) as pool:
        for ds_cfg in MIXTURE:
            process_dataset(pool, ds_cfg)


if __name__ == "__main__":
    run_prefill()
