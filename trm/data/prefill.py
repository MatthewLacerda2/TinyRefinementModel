import argparse
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
from trm.config import TOKENIZER_NAME, resolve_root
from trm.data.dedup import DedupParams, NearDedup
from trm.settings import CONFIG, location
from dotenv import load_dotenv

# Near-dedup (#486): `--dedup` drops each source's near-duplicate documents before they
# are tokenized (trm/data/dedup.py, knobs DEDUP_* in trm/settings.py) and records what it
# removed in the source's status.json. A deduped corpus is a NEW corpus: point DATA_ROOT
# at a new folder. A folder is only ever resumed with the dedup settings it was started
# with, so the tokenized corpus in runs/data/ (written without) refuses `--dedup`.
# `--measure N` only counts: it judges a source's first N documents and writes nothing.
#
# Shard order (#364, measured by reading the code below, not assumed): each source is
# streamed with load_dataset(..., streaming=True) and never shuffled, so the chunk
# files hold documents in the Hugging Face stream order, and the loader reads them
# sequentially. Consecutive micro-steps from one source are neighbours in that order.

# Load environment variables (such as HF_TOKEN) before datasets loads
load_dotenv()

# Config
ENC_NAME = TOKENIZER_NAME
OUTPUT_DIR = resolve_root(location("DATA_ROOT", "runs/data"))
TOKENS_PER_FILE = 125_000_000  # ~500MB per chunk
PREFETCH_BUFFER = 15000        # raw text records buffered ahead of tokenization
TOKENIZE_BATCH_ITEMS = 4000    # records per parallel tokenization round
FINEWEB_MIN_SCORE = 4.0        # educational-score floor. Raised 3.0→4.0: the model is
                               # small (~139M), so we trade volume for per-token quality
                               # density (the phi / TinyStories regime). FineWeb-Edu's
                               # released set is already ≥3; ≥4 is far denser yet still
                               # leaves 100B+ tokens — ample for our 4.0B fineweb target.

# Targets: ~10.85B total (10.5B pretrain + 0.35B chat). Doubled from ~5.5B so the
# 138.7M model (dim960, r50k) trains at ~76 tokens/param — small models reward
# over-training well past Chinchilla, so we feed it generously rather than starve it.
MIXTURE = [
    {
        "path": "HuggingFaceFW/fineweb-edu",
        # The ≥4 score floor keeps only ~7% of docs (~95k tok/s), so this is the slow
        # pole — ~12h to stream 4.0B. Worth it: highest-quality general text and the
        # largest single share of the pretrain mix.
        "target_tokens": 4_000_000_000,
        "folder": "pretrain",
        "alias": "fineweb-edu"
    },
    {
        "path": "codeparrot/codeparrot-clean",
        "target_tokens": 4_000_000_000,
        "folder": "pretrain",
        "alias": "codeparrot"
    },
    {
        "path": "HuggingFaceTB/finemath",
        "config": "finemath-4plus",
        "target_tokens": 2_500_000_000,
        "folder": "pretrain",
        "alias": "finemath"
    },
    {
        # General web, the slice the first base run lacked (#489). Streamed in the same
        # order the modded-nanogpt speedrun reads it, so its FIRST 100M tokens ARE the
        # speedrun FineWeb val shard that instruments/yardstick/fineweb_val scores: after
        # a prefill, move chunk_0.npy (125M tokens) out of the bucket, or a run trains
        # on its own yardstick. What the #489 pair read is chunk_1 and chunk_2.
        "path": "HuggingFaceFW/fineweb",
        "config": "sample-10BT",
        "target_tokens": 500_000_000,
        "folder": "pretrain",
        "alias": "fineweb"
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


def load_status(save_path):
    """The folder's status.json, or None when it has none."""
    status_file = os.path.join(save_path, "status.json")
    if not os.path.exists(status_file):
        return None
    with open(status_file, 'r') as f:
        return json.load(f)


def load_progress(save_path, name):
    """Returns (file_idx, total_tokens, items_processed, stream_offset) for a dataset,
    reading status.json or — recovery mode — scanning existing chunk files.
    `items_processed` counts the documents written; `stream_offset` the raw records
    consumed from the stream, which is where a resume continues. A status written before
    the two were told apart has only the first, and it is used for both."""
    file_idx = 0
    total_tokens = 0
    items_processed = 0

    status = load_status(save_path)
    if status is not None:
        file_idx = status.get("file_idx", 0)
        total_tokens = status.get("total_tokens", 0)
        items_processed = status.get("items_processed", 0)
        stream_offset = status.get("stream_offset", items_processed)
        print(f"🔄 Resuming {name} from status.json: record {stream_offset:,} (Tokens: {total_tokens/1e6:.1f}M, Chunk: {file_idx})")
        return file_idx, total_tokens, items_processed, stream_offset

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

    return file_idx, total_tokens, items_processed, items_processed


def save_progress(save_path, file_idx, total_tokens, items_processed, stream_offset, dedup=None):
    """status.json: where a resume continues, and the source's manifest. With `dedup`,
    its index is written first, so a status never points past the index it resumes with."""
    if dedup is not None:
        dedup.save(os.path.join(save_path, DEDUP_INDEX))
    tmp = os.path.join(save_path, "status.json.tmp")
    with open(tmp, 'w') as f:
        json.dump({
            "file_idx": file_idx,
            "total_tokens": total_tokens,
            "items_processed": items_processed,
            "stream_offset": stream_offset,
            "dedup": None if dedup is None else dedup.record(),
        }, f, indent=1)
    os.replace(tmp, os.path.join(save_path, "status.json"))


# The dedup index beside a source's chunks, for a resume. Not `.npy`: the loaders read
# every `.npy` in a bucket as tokens.
DEDUP_INDEX = "dedup_index.npz"


def open_dedup(save_path, name, params):
    """The near-dedup a source's folder continues with: a fresh one, the one it saved,
    or a refusal. A folder is only resumed under the dedup settings that started it, so
    `--dedup` can never add to a corpus written without it (runs/data/), and a deduped
    corpus is never continued without it."""
    status = load_status(save_path)
    started = status is not None or bool(glob.glob(os.path.join(save_path, "chunk_*.npy")))
    if not started:
        return None if params is None else NearDedup(params)
    saved = (status or {}).get("dedup")
    wanted = None if params is None else params.record()
    built = None if saved is None else {k: saved.get(k) for k in (wanted or saved)}
    if built != wanted:
        raise SystemExit(
            f"{save_path} was written with dedup={built} and this prefill asks for "
            f"dedup={wanted}. A folder is only continued the way it was started: point "
            f"DATA_ROOT at a new folder for a deduplicated corpus.")
    if params is None:
        return None
    index = os.path.join(save_path, DEDUP_INDEX)
    if not os.path.exists(index):
        raise SystemExit(f"{save_path} is a deduplicated corpus without its {DEDUP_INDEX}: "
                         f"it cannot be continued without re-admitting what it dropped.")
    print(f"🔄 Resuming {name}'s dedup index ({os.path.getsize(index)/1e6:.0f} MB)")
    return NearDedup.load(index, params)


def stream_with_retries(ds_cfg, name, start_offset, out_queue, stop_event):
    """Producer: streams (stream offset, raw text) into out_queue, reconnecting on
    network failures and resuming from the last consumed record. The offset counts every
    raw record read, the ones extract_text filters out included, so it is where a resume
    must skip to. Ends with a None sentinel."""
    split_name = ds_cfg.get('split', 'train')

    def open_stream(offset):
        ds = load_dataset(ds_cfg['path'], name=ds_cfg.get('config'), split=split_name, streaming=True)
        return ds.skip(offset) if offset > 0 else ds

    current_offset = start_offset
    active_ds = open_stream(current_offset)

    while not stop_event.is_set():
        try:
            for item in active_ds:
                if stop_event.is_set():
                    break
                current_offset += 1
                txt = extract_text(item, name)
                if txt:
                    out_queue.put((current_offset, txt))
            # Natural exit means dataset is completed
            break
        except Exception as e:
            # Catch connection timeouts, name resolution failures, or closed HTTPX client errors
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


def write_chunk(save_path, file_idx, token_acc, stride):
    """Saves accumulated tokens as a chunk aligned to `stride` (a row: two MAX_SEQ_LEN
    windows and the target after them). Returns the next chunk index and the
    unaligned remainder (carried into the next chunk)."""
    chunk_data = np.concatenate(token_acc)
    valid_len = (len(chunk_data) // stride) * stride
    if valid_len == 0:
        return file_idx, token_acc

    t_save = time.strftime('%H:%M:%S', time.localtime())
    np.save(os.path.join(save_path, f"chunk_{file_idx}.npy"), chunk_data[:valid_len])
    print(f"\n[{t_save}] 💾 Saved chunk_{file_idx}.npy ({valid_len/1e6:.1f}M tokens)")

    remainder = chunk_data[valid_len:]
    return file_idx + 1, [remainder] if len(remainder) > 0 else []


def source_name(ds_cfg):
    return ds_cfg.get('alias') or ds_cfg['path'].split('/')[-1]


def start_stream(ds_cfg, name, offset):
    """The producer thread, reading from raw record `offset`, and the queue it fills."""
    prefetch_queue = queue.Queue(maxsize=PREFETCH_BUFFER)
    stop_event = threading.Event()
    threading.Thread(
        target=stream_with_retries,
        args=(ds_cfg, name, offset, prefetch_queue, stop_event),
        daemon=True,
    ).start()
    return prefetch_queue, stop_event


def stop_stream(prefetch_queue, stop_event):
    stop_event.set()
    # Empty the queue to unblock the producer thread
    while True:
        try:
            prefetch_queue.get_nowait()
        except queue.Empty:
            break


def next_batch(prefetch_queue):
    """Up to TOKENIZE_BATCH_ITEMS texts, the stream offset after the last of them (None
    if there were none), and whether the stream has ended."""
    texts, offset = [], None
    while len(texts) < TOKENIZE_BATCH_ITEMS:
        item = prefetch_queue.get()
        if item is None:  # Sentinel: stream completed
            return texts, offset, True
        offset, txt = item
        texts.append(txt)
    return texts, offset, False


def process_dataset(pool, ds_cfg, stride, dedup_params=None):
    """Stream one source into `stride`-aligned chunks until its token target, resuming
    where its status.json says. With `dedup_params`, near-duplicate documents are
    dropped before they are tokenized (see the note at the top)."""
    name = source_name(ds_cfg)
    target = ds_cfg['target_tokens']
    save_path = os.path.join(OUTPUT_DIR, ds_cfg['folder'], name)
    os.makedirs(save_path, exist_ok=True)

    print(f"\n🚀 Processing {name} | Target: {target/1e9:.2f}B tokens"
          + (f" | near-dedup {dedup_params.record()}" if dedup_params else ""))

    dedup = open_dedup(save_path, name, dedup_params)
    file_idx, total_tokens_ds, items_processed, stream_offset = load_progress(save_path, name)
    if total_tokens_ds >= target:
        print(f"⏩ {name} already completed. Skipping.")
        return

    prefetch_queue, stop_event = start_stream(ds_cfg, name, stream_offset)
    token_acc = []
    t_start = time.time()
    initial_tokens = total_tokens_ds

    try:
        done = False
        while not done and total_tokens_ds < target:
            texts, offset, done = next_batch(prefetch_queue)
            if offset is not None:
                stream_offset = offset
            if dedup is not None:
                texts = dedup.keep(texts, pool.map)
            if texts:
                flat_batch = tokenize_buffer(pool, texts)
                token_acc.append(flat_batch)
                total_tokens_ds += len(flat_batch)
                items_processed += len(texts)

            elapsed = time.time() - t_start
            tokens_sec = (total_tokens_ds - initial_tokens) / max(1e-3, elapsed)
            progress = (total_tokens_ds / target) * 100
            dropped = f" | Near-dups dropped: {dedup.record()['doc_removal_rate']:.1%}" if dedup else ""
            sys.stdout.write(
                f"\rProgress: {total_tokens_ds/1e6:.1f}M/{target/1e6:.0f}M tokens ({progress:.1f}%) | "
                f"Speed: {tokens_sec/1e3:.1f}k tok/s | Elapsed: {elapsed/60:.1f}m{dropped}"
            )
            sys.stdout.flush()

            if sum(len(x) for x in token_acc) >= TOKENS_PER_FILE:
                file_idx, token_acc = write_chunk(save_path, file_idx, token_acc, stride)
                save_progress(save_path, file_idx, total_tokens_ds, items_processed, stream_offset, dedup)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Cleaning up background threads...")
        raise
    finally:
        stop_stream(prefetch_queue, stop_event)

    # Final flush: what the last full chunk left, whether the stream ended or the target
    # was reached, so the status (and its dedup counts) describe exactly what was written.
    if token_acc:
        file_idx, _ = write_chunk(save_path, file_idx, token_acc, stride)
    save_progress(save_path, file_idx, total_tokens_ds, items_processed, stream_offset, dedup)
    print(f"\n🏁 Finished {name}. Total: {total_tokens_ds/1e9:.2f}B tokens"
          + (f" | {json.dumps(dedup.record())}" if dedup else ""))


def measure_duplicates(pool, ds_cfg, docs, dedup_params):
    """Judge a source's first `docs` documents and write nothing: its near-duplicate rate
    at no disk cost, for deciding whether a source needs the pass. A prefix's rate is a
    floor on the whole source's, since every document has fewer earlier ones to match."""
    name = source_name(ds_cfg)
    print(f"\n🔎 Measuring {name}'s near-duplicates over its first {docs:,} documents")
    dedup = NearDedup(dedup_params)
    prefetch_queue, stop_event = start_stream(ds_cfg, name, 0)
    t_start = time.time()
    try:
        done = False
        while not done and dedup.counts["docs_seen"] < docs:
            texts, _, done = next_batch(prefetch_queue)
            dedup.keep(texts[:docs - dedup.counts["docs_seen"]], pool.map)
            sys.stdout.write(f"\r{dedup.counts['docs_seen']:,} documents | "
                             f"dropped {dedup.record()['doc_removal_rate']:.1%} | "
                             f"{(time.time() - t_start)/60:.1f}m")
            sys.stdout.flush()
    finally:
        stop_stream(prefetch_queue, stop_event)
    record = {"source": name, **dedup.record()}
    print("\nDEDUP_RATE " + json.dumps(record), flush=True)
    return record


def run_prefill(only=(), dedup=False, measure=None):
    """Every MIXTURE source, or the ones named in `only`, in MIXTURE order."""
    known = [source_name(c) for c in MIXTURE]
    unknown = set(only) - set(known)
    if unknown:
        raise SystemExit(f"unknown sources {sorted(unknown)}; the MIXTURE has {known}")
    sources = [c for c in MIXTURE if not only or source_name(c) in only]
    params = DedupParams.of(CONFIG) if dedup or measure else None
    # Leave one core free to keep the system responsive
    num_workers = max(1, int(cpu_count() - 1))

    with Pool(num_workers) as pool:
        for ds_cfg in sources:
            if measure:
                measure_duplicates(pool, ds_cfg, measure, params)
            else:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                process_dataset(pool, ds_cfg, 2 * CONFIG.MAX_SEQ_LEN + 1, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize the MIXTURE's sources into DATA_ROOT.")
    parser.add_argument("--only", nargs="+", default=(), metavar="SOURCE",
                        help="just these sources (their aliases), e.g. --only codeparrot")
    parser.add_argument("--dedup", action="store_true",
                        help="drop near-duplicate documents before tokenizing (#486); needs a "
                             "DATA_ROOT whose folders were not written without it")
    parser.add_argument("--measure", type=int, default=None, metavar="DOCS",
                        help="write nothing: print each source's near-duplicate rate over its "
                             "first DOCS documents")
    args = parser.parse_args()
    run_prefill(args.only, args.dedup, args.measure)
