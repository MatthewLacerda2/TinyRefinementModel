import os
import numpy as np
import tiktoken
from datasets import load_dataset

ENC = tiktoken.get_encoding("cl100k_base")
OUTPUT_DIR = "./tpu_data" # Change this as needed for your local drive
TOKENS_PER_FILE = 125_000_000 # ~500MB per chunk

MIXTURE = [
    # We use 'alias' to ensure folder names match start_training.py exactly
    {"path": "HuggingFaceFW/fineweb-edu", "pct": 0.60, "folder": "pretrain", "alias": "fineweb-edu"},
    {"path": "TokenBender/code_instructions_122k_alpaca_style", "pct": 0.25, "folder": "pretrain", "alias": "code_instructions"},
    {"path": "HuggingFaceTB/finemath", "config": "finemath-4plus", "pct": 0.15, "folder": "pretrain", "alias": "finemath"},
    {"path": "HuggingFaceH4/ultrachat_200k", "pct": 1.0, "folder": "chat", "alias": "ultrachat"}
]

def tokenize_batch(batch):
    tokens = []
    for text in batch:
        tokens.extend(ENC.encode(text))
        tokens.append(ENC.eot_token)
    return np.array(tokens, dtype=np.int32)

def run_prefill():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for ds_cfg in MIXTURE:
        name = ds_cfg.get('alias') or ds_cfg['path'].split('/')[-1]
        # Creates tpu_data/pretrain/fineweb-edu/ etc.
        dataset_folder = os.path.join(OUTPUT_DIR, ds_cfg['folder'], name)
        os.makedirs(dataset_folder, exist_ok=True)
        
        print(f"🚀 Processing {ds_cfg['path']} into {dataset_folder}...")

        ds = load_dataset(ds_cfg['path'], name=ds_cfg.get('config'), split="train", streaming=True)
        
        buffer = []
        file_idx = 0
        token_acc = []
        
        for i, item in enumerate(ds):
            txt = item.get("text") or item.get("content") or item.get("prompt")
            if txt: buffer.append(txt)
            
            if len(buffer) >= 2000:
                tokens = tokenize_batch(buffer)
                token_acc.append(tokens)
                buffer = []
                
                current_count = sum(len(x) for x in token_acc)
                if current_count >= TOKENS_PER_FILE:
                    flat = np.concatenate(token_acc)
                    chunk_path = os.path.join(dataset_folder, f"chunk_{file_idx}.npy")
                    np.save(chunk_path, flat)
                    print(f"✅ Saved {name} chunk {file_idx} ({current_count} tokens)")
                    file_idx += 1
                    token_acc = []
        
        # FINAL FLUSH: Save the remaining tokens
        if token_acc or buffer:
            if buffer:
                token_acc.append(tokenize_batch(buffer))
            if token_acc:
                flat = np.concatenate(token_acc)
                chunk_path = os.path.join(dataset_folder, f"chunk_{file_idx}.npy")
                np.save(chunk_path, flat)
                print(f"🏁 Saved final {name} chunk {file_idx} ({len(flat)} tokens)")

if __name__ == "__main__":
    run_prefill()