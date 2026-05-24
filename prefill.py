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
from layers import MAX_SEQ_LEN
from dotenv import load_dotenv

# Load environment variables (such as HF_TOKEN) before datasets loads
load_dotenv()

# Config
ENC_NAME = "cl100k_base"
OUTPUT_DIR = "./tpu_data"
TOKENS_PER_FILE = 125_000_000  # ~500MB per chunk

# Targets: 7.8B total for $10 run + ~15% buffer
MIXTURE = [
    {
        "path": "HuggingFaceFW/fineweb-edu",
        "target_tokens": 4_500_000_000,
        "folder": "pretrain",
        "alias": "fineweb-edu"
    },
    {
        "path": "TokenBender/code_instructions_122k_alpaca_style",
        "target_tokens": 2_000_000_000,
        "folder": "pretrain",
        "alias": "code_instructions"
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
        "target_tokens": 1_500_000_000,
        "folder": "chat",
        "alias": "ultrachat"
    }
]

def tokenize_batch_parallel(text):
    """Worker function for parallel tokenization."""
    enc = tiktoken.get_encoding(ENC_NAME)
    return enc.encode(text) + [enc.eot_token]

def run_prefill():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Use 75% of cores to keep system responsive
    num_workers = max(1, int(cpu_count() - 1))
    
    with Pool(num_workers) as pool:
        for ds_cfg in MIXTURE:
            name = ds_cfg.get('alias') or ds_cfg['path'].split('/')[-1]
            target = ds_cfg['target_tokens']
            save_path = os.path.join(OUTPUT_DIR, ds_cfg['folder'], name)
            os.makedirs(save_path, exist_ok=True)
            
            print(f"\n🚀 Processing {name} | Target: {target/1e9:.2f}B tokens")

            # Resume logic: Check if we already have progress
            status_file = os.path.join(save_path, "status.json")
            file_idx = 0
            total_tokens_ds = 0
            items_processed = 0

            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    status = json.load(f)
                    file_idx = status.get("file_idx", 0)
                    total_tokens_ds = status.get("total_tokens", 0)
                    items_processed = status.get("items_processed", 0)
                print(f"🔄 Resuming {name} from status.json: item {items_processed:,} (Tokens: {total_tokens_ds/1e6:.1f}M, Chunk: {file_idx})")
            else:
                # Recovery Mode: Scan for existing .npy files if status.json is missing
                existing_chunks = glob.glob(os.path.join(save_path, "chunk_*.npy"))
                if existing_chunks:
                    try:
                        # Extract indices and find max
                        indices = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing_chunks]
                        file_idx = max(indices) + 1
                        
                        total_tokens_ds = 0
                        for f in existing_chunks:
                            data = np.load(f, mmap_mode='r')
                            total_tokens_ds += len(data)
                        
                        print(f"🔎 Auto-discovered progress for {name}: {total_tokens_ds/1e6:.1f}M tokens in {len(existing_chunks)} chunks.")
                        print(f"⚠️ Note: Starting stream from beginning as row count is unknown (Next chunk: {file_idx})")
                    except Exception as e:
                        print(f"⚠️ Could not recover progress for {name}: {e}")
                        file_idx = 0
                        total_tokens_ds = 0
                
            if total_tokens_ds >= target:
                print(f"⏩ {name} already completed. Skipping.")
                continue

            split_name = ds_cfg.get('split', 'train')
            ds = load_dataset(ds_cfg['path'], name=ds_cfg.get('config'), split=split_name, streaming=True)
            if items_processed > 0:
                ds = ds.skip(items_processed)
            
            # Setup prefetch pipeline: Producer thread to buffer raw text records
            prefetch_queue = queue.Queue(maxsize=15000)
            stop_event = threading.Event()
            
            def producer():
                current_offset = items_processed
                active_ds = ds
                
                while not stop_event.is_set():
                    producer_consumed = 0
                    try:
                        for item in active_ds:
                            if stop_event.is_set():
                                break
                            
                            producer_consumed += 1
                            
                            if name == "fineweb-edu":
                                score = item.get("score", item.get("educational_score", 0.0))
                                if score < 3.0:
                                    continue
                            
                            if name == "ultrachat" and "messages" in item:
                                msg_list = item["messages"]
                                if isinstance(msg_list, list):
                                    txt = "\n\n".join(f"{msg.get('role', 'unknown').capitalize()}: {msg.get('content', '')}" for msg in msg_list)
                                else:
                                    txt = str(msg_list)
                            else:
                                txt = item.get("text") or item.get("content") or item.get("prompt")
                            
                            if not txt and "data" in item:
                                if isinstance(item["data"], list):
                                    txt = "\n".join(str(x) for x in item["data"])
                                else:
                                    txt = str(item["data"])
                            
                            if txt:
                                prefetch_queue.put(txt)
                        # Natural exit means dataset is completed
                        break
                    except Exception as e:
                        # Catch connection timeouts, name resolution failures, or closed HTTPX client errors
                        current_offset += producer_consumed
                        t_now = time.strftime('%H:%M:%S', time.localtime())
                        print(f"\n[{t_now}] 🔌 Prefetcher connection dropped: {e}")
                        print(f"[{t_now}] 🔄 Re-connecting and resuming dataset from item {current_offset:,} in 5 seconds...")
                        time.sleep(5)
                        
                        try:
                            active_ds = load_dataset(ds_cfg['path'], name=ds_cfg.get('config'), split=split_name, streaming=True)
                            if current_offset > 0:
                                active_ds = active_ds.skip(current_offset)
                        except Exception as conn_err:
                            print(f"\n⚠️ Re-connection failed: {conn_err}. Will retry shortly...")
                
                # Signal end of stream
                prefetch_queue.put(None)

            prefetch_thread = threading.Thread(target=producer, daemon=True)
            prefetch_thread.start()

            buffer = []
            token_acc = []
            
            t_start = time.time()
            initial_tokens = total_tokens_ds
            items_since_start = 0

            try:
                while True:
                    # Pull next item from the background download queue
                    txt = prefetch_queue.get()
                    if txt is None: # Sentinel received
                        break
                    
                    buffer.append(txt)
                    
                    # Batch size for parallel processing
                    if len(buffer) >= 4000:
                        token_lists = pool.map(tokenize_batch_parallel, buffer)
                        flat_batch = np.array([t for sub in token_lists for t in sub], dtype=np.int32)
                        
                        token_acc.append(flat_batch)
                        total_tokens_ds += len(flat_batch)
                        items_processed += len(buffer)
                        items_since_start += len(buffer)
                        buffer = []
                        
                        # Calculate Metrics
                        elapsed = time.time() - t_start
                        tokens_sec = (total_tokens_ds - initial_tokens) / max(1e-3, elapsed)
                        progress = (total_tokens_ds / target) * 100
                        
                        sys.stdout.write(
                            f"\rProgress: {total_tokens_ds/1e6:.1f}M/{target/1e6:.0f}M tokens ({progress:.1f}%) | "
                            f"Speed: {tokens_sec/1e3:.1f}k tok/s | Elapsed: {elapsed/60:.1f}m"
                        )
                        sys.stdout.flush()

                        current_chunk_size = sum(len(x) for x in token_acc)
                        if current_chunk_size >= TOKENS_PER_FILE:
                            chunk_data = np.concatenate(token_acc)
                            stride = 2 * MAX_SEQ_LEN + 1
                            valid_len = (len(chunk_data) // stride) * stride
                            
                            t_save = time.strftime('%H:%M:%S', time.localtime())
                            np.save(os.path.join(save_path, f"chunk_{file_idx}.npy"), chunk_data[:valid_len])
                            print(f"\n[{t_save}] 💾 Saved chunk_{file_idx}.npy ({valid_len/1e6:.1f}M tokens)")
                            
                            file_idx += 1
                            remainder = chunk_data[valid_len:]
                            token_acc = [remainder] if len(remainder) > 0 else []
                            
                            # Save status after writing chunk
                            with open(status_file, 'w') as f:
                                json.dump({
                                    "file_idx": file_idx,
                                    "total_tokens": total_tokens_ds,
                                    "items_processed": items_processed
                                }, f)

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

            # Final Flush for this dataset
            if (token_acc or buffer) and total_tokens_ds < target:
                if buffer:
                    token_lists = pool.map(tokenize_batch_parallel, buffer)
                    flat_batch = np.array([t for sub in token_lists for t in sub], dtype=np.int32)
                    token_acc.append(flat_batch)
                    total_tokens_ds += len(flat_batch)
                    items_processed += len(buffer)
                if token_acc:
                    chunk_data = np.concatenate(token_acc)
                    stride = 2 * MAX_SEQ_LEN + 1
                    valid_len = (len(chunk_data) // stride) * stride
                    
                    if valid_len > 0:
                        t_save = time.strftime('%H:%M:%S', time.localtime())
                        np.save(os.path.join(save_path, f"chunk_{file_idx}.npy"), chunk_data[:valid_len])
                        print(f"\n[{t_save}] 💾 Saved final flush chunk_{file_idx}.npy ({valid_len/1e6:.1f}M tokens)")
                    
                    # Final status update
                    with open(status_file, 'w') as f:
                        json.dump({
                            "file_idx": file_idx + 1,
                            "total_tokens": total_tokens_ds,
                            "items_processed": items_processed
                        }, f)
                    
                    print(f"\n🏁 Finished {name}. Total: {total_tokens_ds/1e9:.2f}B tokens")

if __name__ == "__main__":
    run_prefill()