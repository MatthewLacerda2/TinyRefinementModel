import os
import numpy as np
import tiktoken
import sys
from datasets import load_dataset
from multiprocessing import Pool, cpu_count
import json
import glob

#train on fineweb and densefinelist and truthfulqa and python-edu
#than openmath and proofwriter and believemenot
#than ultrachat

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
                        
                        # Calculate total tokens from file sizes (assuming int32 = 4 bytes)
                        # More robust to just add them up
                        total_tokens_ds = 0
                        for f in existing_chunks:
                            # We check shape without loading full data for speed
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
            
            buffer = []
            token_acc = []
            current_batch_count = 0
            
            for item in ds:
                txt = item.get("text") or item.get("content") or item.get("prompt")
                
                # Handle lists of strings (like UltraChat's 'data' field)
                if not txt and "data" in item:
                    if isinstance(item["data"], list):
                        txt = "\n".join(str(x) for x in item["data"])
                    else:
                        txt = str(item["data"])

                if txt: 
                    buffer.append(txt)
                
                current_batch_count += 1
                
                # Batch size for parallel processing
                if len(buffer) >= 4000:
                    # Parallel Tokenization
                    token_lists = pool.map(tokenize_batch_parallel, buffer)
                    flat_batch = np.array([t for sub in token_lists for t in sub], dtype=np.int32)
                    
                    token_acc.append(flat_batch)
                    total_tokens_ds += len(flat_batch)
                    items_processed += len(buffer)
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
                    np.save(os.path.join(save_path, f"chunk_{file_idx}.npy"), chunk_data)
                    
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