import os
import numpy as np
import tiktoken
import sys
from datasets import load_dataset
from multiprocessing import Pool, cpu_count

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

            ds = load_dataset(ds_cfg['path'], name=ds_cfg.get('config'), split="train", streaming=True)
            
            buffer = []
            file_idx = 0
            token_acc = []
            total_tokens_ds = 0
            
            for item in ds:
                txt = item.get("text") or item.get("content") or item.get("prompt")
                if txt: buffer.append(txt)
                
                # Batch size for parallel processing
                if len(buffer) >= 4000:
                    # Parallel Tokenization
                    token_lists = pool.map(tokenize_batch_parallel, buffer)
                    flat_batch = np.array([t for sub in token_lists for t in sub], dtype=np.int32)
                    
                    token_acc.append(flat_batch)
                    total_tokens_ds += len(flat_batch)
                    buffer = []
                    
                    # Live Progress Report
                    progress = (total_tokens_ds / target) * 100
                    sys.stdout.write(f"\rProgress: {total_tokens_ds/1e6:.1f}M / {target/1e6:.0f}M tokens ({progress:.1f}%)")
                    sys.stdout.flush()

                    current_chunk_size = sum(len(x) for x in token_acc)
                    if current_chunk_size >= TOKENS_PER_FILE:
                        chunk_data = np.concatenate(token_acc)
                        np.save(os.path.join(save_path, f"chunk_{file_idx}.npy"), chunk_data)
                        file_idx += 1
                        token_acc = []

                    if total_tokens_ds >= target:
                        print(f"\n✅ {name} target reached.")
                        break
            
            # Final Flush for this dataset
            if (token_acc or buffer) and total_tokens_ds < target:
                if buffer:
                    token_lists = pool.map(tokenize_batch_parallel, buffer)
                    flat_batch = np.array([t for sub in token_lists for t in sub], dtype=np.int32)
                    token_acc.append(flat_batch)
                    total_tokens_ds += len(flat_batch)
                if token_acc:
                    chunk_data = np.concatenate(token_acc)
                    np.save(os.path.join(save_path, f"chunk_{file_idx}.npy"), chunk_data)
                    print(f"\n🏁 Finished {name}. Total: {total_tokens_ds/1e9:.2f}B tokens")

if __name__ == "__main__":
    run_prefill()