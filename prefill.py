import os
import numpy as np
import tiktoken
import json
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

def tokenize_single_text(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.encode(text) + [enc.eot_token]

def run_prefill():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    num_workers = max(1, int(cpu_count() * 0.75))
    print(f"🔥 Starting parallel prefill with {num_workers} workers...")

    with Pool(num_workers) as pool:
        for ds_cfg in MIXTURE:
            name = ds_cfg.get('alias') or ds_cfg['path'].split('/')[-1]
            dataset_folder = os.path.join(OUTPUT_DIR, ds_cfg['folder'], name)
            os.makedirs(dataset_folder, exist_ok=True)
            
            state_path = os.path.join(dataset_folder, "resume_state.json")
            start_doc_idx = 0
            file_idx = 0
            
            if os.path.exists(state_path):
                try:
                    with open(state_path, 'r') as f:
                        state = json.load(f)
                        start_doc_idx = state.get("last_doc_idx", 0)
                        file_idx = state.get("next_file_idx", 0)
                        print(f"⏩ Resuming {name} from Doc {start_doc_idx}, Chunk {file_idx}...")
                except Exception as e:
                    print(f"⚠️ Could not load resume state for {name}: {e}. Starting from scratch.")

            print(f"🚀 Processing {ds_cfg['path']}...")
            ds = load_dataset(ds_cfg['path'], name=ds_cfg.get('config'), split="train", streaming=True)
            
            if start_doc_idx > 0:
                ds = ds.skip(start_doc_idx)
            
            buffer = []
            token_acc = []
            current_count = 0
            docs_processed_in_this_run = 0
            
            for i, item in enumerate(ds):
                txt = item.get("text") or item.get("content") or item.get("prompt")
                if txt: buffer.append(txt)
                docs_processed_in_this_run += 1
                
                if len(buffer) >= 10000:
                    results = pool.map(tokenize_single_text, buffer)
                    flat_batch = [t for tokens in results for t in tokens]
                    chunk = np.array(flat_batch, dtype=np.int32)
                    
                    token_acc.append(chunk)
                    current_count += len(chunk)
                    buffer = []
                    
                    if current_count >= TOKENS_PER_FILE:
                        full_flat = np.concatenate(token_acc)
                        # Exact chunking ensures skip_count math works in start_training.py
                        save_chunk = full_flat[:TOKENS_PER_FILE]
                        leftover = full_flat[TOKENS_PER_FILE:]
                        
                        chunk_path = os.path.join(dataset_folder, f"chunk_{file_idx}.npy")
                        np.save(chunk_path, save_chunk)
                        
                        # Save state for resume
                        with open(state_path, 'w') as f:
                            json.dump({
                                "last_doc_idx": start_doc_idx + docs_processed_in_this_run,
                                "next_file_idx": file_idx + 1
                            }, f)
                            
                        print(f"✅ Saved {name} chunk {file_idx} (Total Docs: {start_doc_idx + docs_processed_in_this_run})")
                        
                        file_idx += 1
                        token_acc = [leftover] if len(leftover) > 0 else []
                        current_count = len(leftover)

            # Final Flush
            if buffer:
                results = pool.map(tokenize_single_text, buffer)
                flat_batch = [t for tokens in results for t in tokens]
                token_acc.append(np.array(flat_batch, dtype=np.int32))
            
            if token_acc:
                final_flat = np.concatenate(token_acc)
                chunk_path = os.path.join(dataset_folder, f"chunk_{file_idx}.npy")
                np.save(chunk_path, final_flat)
                print(f"🏁 Saved final {name} chunk {file_idx}")

if __name__ == "__main__":
    run_prefill()