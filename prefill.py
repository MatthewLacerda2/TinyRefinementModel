import os
import numpy as np
import tiktoken
from datasets import load_dataset
from multiprocessing import Pool, cpu_count

ENC = tiktoken.get_encoding("cl100k_base")
OUTPUT_DIR = "./tpu_data"
TOKENS_PER_FILE = 125_000_000 # ~500MB per chunk

MIXTURE = [
    {"path": "HuggingFaceFW/fineweb-edu", "pct": 0.60, "folder": "pretrain", "alias": "fineweb-edu"},
    {"path": "TokenBender/code_instructions_122k_alpaca_style", "pct": 0.25, "folder": "pretrain", "alias": "code_instructions"},
    {"path": "HuggingFaceTB/finemath", "config": "finemath-4plus", "pct": 0.15, "folder": "pretrain", "alias": "finemath"},
    {"path": "HuggingFaceH4/ultrachat_200k", "pct": 1.0, "folder": "chat", "alias": "ultrachat"}
]

# This function runs in the worker processes
def tokenize_single_text(text):
    # Re-getting encoding is fast/cached and safer for multiprocess
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.encode(text) + [enc.eot_token]

def run_prefill():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Use most available cores to leave room for the OS and downloader
    num_workers = max(1, int(cpu_count() * 0.75))
    print(f"🔥 Starting parallel prefill with {num_workers} workers...")

    with Pool(num_workers) as pool:
        for ds_cfg in MIXTURE:
            name = ds_cfg.get('alias') or ds_cfg['path'].split('/')[-1]
            dataset_folder = os.path.join(OUTPUT_DIR, ds_cfg['folder'], name)
            os.makedirs(dataset_folder, exist_ok=True)
            
            print(f"🚀 Processing {ds_cfg['path']} into {dataset_folder}...")

            ds = load_dataset(ds_cfg['path'], name=ds_cfg.get('config'), split="train", streaming=True)
            
            buffer = []
            file_idx = 0
            token_acc = []
            current_count = 0
            
            for i, item in enumerate(ds):
                txt = item.get("text") or item.get("content") or item.get("prompt")
                if txt: buffer.append(txt)
                
                # Process in large batches to keep cores busy
                if len(buffer) >= 10000:
                    results = pool.map(tokenize_single_text, buffer)
                    
                    # Flatten results
                    flat_batch = [t for tokens in results for t in tokens]
                    chunk = np.array(flat_batch, dtype=np.int32)
                    
                    token_acc.append(chunk)
                    current_count += len(chunk)
                    buffer = []
                    
                    if current_count >= TOKENS_PER_FILE:
                        full_flat = np.concatenate(token_acc)
                        # Save exactly TOKENS_PER_FILE to help skip_count math
                        save_chunk = full_flat[:TOKENS_PER_FILE]
                        leftover = full_flat[TOKENS_PER_FILE:]
                        
                        chunk_path = os.path.join(dataset_folder, f"chunk_{file_idx}.npy")
                        np.save(chunk_path, save_chunk)
                        print(f"✅ Saved {name} chunk {file_idx} ({len(save_chunk)} tokens)")
                        
                        file_idx += 1
                        token_acc = [leftover] if len(leftover) > 0 else []
                        current_count = len(leftover)

            # Final Flush for this dataset
            if buffer:
                results = pool.map(tokenize_single_text, buffer)
                flat_batch = [t for tokens in results for t in tokens]
                token_acc.append(np.array(flat_batch, dtype=np.int32))
            
            if token_acc:
                final_flat = np.concatenate(token_acc)
                chunk_path = os.path.join(dataset_folder, f"chunk_{file_idx}.npy")
                np.save(chunk_path, final_flat)
                print(f"🏁 Saved final {name} chunk {file_idx} ({len(final_flat)} tokens)")

if __name__ == "__main__":
    run_prefill()