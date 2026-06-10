import os
import sys
from datasets import load_dataset

class BaseDatasetCurator:
    """
    Abstract Base Class for modular Hugging Face dataset curation,
    featuring dynamic schema auto-discovery, statistical batch analytics,
    and automatic prefill integration config exporting.
    """
    REGISTRY = {}

    def __init__(self, dataset_path, config_name=None, split="train", text_key=None, score_key=None, alias=None):
        self.dataset_path = dataset_path
        self.config_name = config_name
        self.split = split
        self.text_key = text_key
        self.score_key = score_key
        self.alias = alias or dataset_path.split("/")[-1]
        
        # Register in global registry for prefill extraction
        BaseDatasetCurator.REGISTRY[self.alias] = self

    def init_stream(self):
        """Safely initializes the streaming dataset generator from Hugging Face."""
        try:
            print(f"📡 Initializing stream: {self.dataset_path} (subset: {self.config_name}, split: {self.split})...")
            ds = load_dataset(
                self.dataset_path, 
                name=self.config_name, 
                split=self.split, 
                streaming=True
            )
            return ds
        except Exception as e:
            print(f"❌ Failed to stream dataset {self.dataset_path}: {e}")
            return None

    def auto_discover(self, ds_stream=None):
        """
        Streams a single sample, automatically discovers text and score columns,
        and prints the schema analysis report.
        """
        if ds_stream is None:
            ds_stream = self.init_stream()
        if ds_stream is None:
            return False

        try:
            sample = next(iter(ds_stream))
            keys = list(sample.keys())
            
            # Auto-detect text field
            if not self.text_key:
                for candidate in ["text", "content", "prompt", "instruction", "messages"]:
                    if candidate in keys:
                        self.text_key = candidate
                        break
            
            # Auto-detect educational/difficulty score field
            if not self.score_key:
                for candidate in ["score", "int_score", "educational_score"]:
                    if candidate in keys:
                        self.score_key = candidate
                        break
            
            print(f"🔍 Auto-Discovery for Alias '{self.alias}':")
            print(f"   - Identified Text Key:  '{self.text_key}'")
            print(f"   - Identified Score Key: '{self.score_key}' (Available keys: {keys})")
            return True
        except Exception as e:
            print(f"❌ Schema auto-discovery failed for {self.alias}: {e}")
            return False

    def fetch_samples(self, n=5):
        """Fetches n samples from the streaming dataset."""
        ds = self.init_stream()
        if ds is None:
            return []
        
        samples = []
        try:
            iterator = iter(ds)
            for _ in range(n):
                samples.append(next(iterator))
        except StopIteration:
            pass
        return samples

    def extract_text(self, item):
        """Extracts text content cleanly based on custom schema layout."""
        if not self.text_key:
            return ""
        
        raw_val = item.get(self.text_key)
        if not raw_val:
            return ""
            
        # Handle ultrachat list of messages dicts
        if self.text_key == "messages" and isinstance(raw_val, list):
            turns = []
            for msg in raw_val:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                turns.append(f"{role.capitalize()}: {content}")
            return "\n\n".join(turns)
            
        return str(raw_val)

    def analyze_batch(self, n=10):
        """Computes statistical metrics over a streamed batch."""
        samples = self.fetch_samples(n)
        if not samples:
            print(f"⚠️ No samples fetched for {self.alias}.")
            return {}

        self.auto_discover()

        char_lengths = []
        scores = []
        null_count = 0

        for item in samples:
            txt = self.extract_text(item)
            if not txt:
                null_count += 1
                continue
                
            char_lengths.append(len(txt))
            
            if self.score_key:
                score_val = item.get(self.score_key)
                if score_val is not None:
                    scores.append(float(score_val))

        avg_len = sum(char_lengths) / len(char_lengths) if char_lengths else 0
        avg_score = sum(scores) / len(scores) if scores else None

        analysis = {
            "alias": self.alias,
            "samples_analyzed": len(samples),
            "text_key_used": self.text_key,
            "score_key_used": self.score_key,
            "avg_char_length": avg_len,
            "avg_quality_score": avg_score,
            "null_text_percentage": (null_count / len(samples)) * 100
        }
        return analysis

    def get_export_config(self):
        """Generates dynamic configuration compatible with prefill.py."""
        return {
            "path": self.dataset_path,
            "config": self.config_name,
            "split": self.split,
            "alias": self.alias,
            "text_key": self.text_key,
            "score_key": self.score_key
        }
