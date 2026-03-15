import os
import numpy as np
import tiktoken
from datasets import load_dataset
from multiprocessing import Pool, cpu_count

# Setup
ENC = tiktoken.get_encoding("cl100k_base")
OUTPUT_DIR = "./tpu_data"
TOKENS_PER_FILE = 125_000_000 # 500MB per chunk

# IQ-Focused Mixture
MIXTURE = [
    {"path": "HuggingFaceFW/fineweb-edu", "pct": 0.60},
    {"path": "TokenBender/code_instructions_122k_alpaca_style", "pct": 0.20},
    {"path": "HuggingFaceTB/finemath", "config": "finemath-4plus", "pct": 0.15},
    {"path": "HuggingFaceH4/ultrachat_200k", "pct": 0.05} # Conversational tail
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
        print(f"Reading {ds_cfg['path']}...")
        # Streaming from HF to your local CPU
        ds = load_dataset(ds_cfg['path'], name=ds_cfg.get('config'), split="train", streaming=True)
        
        buffer = []
        file_idx = 0
        token_acc = []
        
        for i, item in enumerate(ds):
            txt = item.get("text") or item.get("content") or item.get("prompt")
            if txt: buffer.append(txt)
            
            if len(buffer) >= 2000:
                # Parallel tokenization
                tokens = tokenize_batch(buffer)
                token_acc.append(tokens)
                buffer = []
                
                current_count = sum(len(x) for x in token_acc)
                if current_count >= TOKENS_PER_FILE:
                    flat = np.concatenate(token_acc)
                    name = ds_cfg['path'].split('/')[-1]
                    np.save(f"{OUTPUT_DIR}/{name}_{file_idx}.npy", flat)
                    print(f"Saved {name} chunk {file_idx}")
                    file_idx += 1
                    token_acc = []

if __name__ == "__main__":
    run_prefill()