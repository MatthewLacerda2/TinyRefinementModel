import os
import sys
import json
import time
import datetime
import subprocess

from config import (
    LATENT_DIM,
    NUM_BLOCKS,
    SHARED_SLOTS,
    MAX_SEQ_LEN,
    VOCAB_SIZE,
    MAX_STEPS_LIMIT,
    BATCH_SIZE,
    ACCUMULATION_STEPS,
    PAD_TOKEN_ID,
    NUM_HEADS,
    NUM_GROUPS,
)

class RunTracker:
    def __init__(self, runs_root="runs"):
        self.runs_root = runs_root
        self.run_id = None
        self.run_dir = None
        self.start_time = None
        self.session_index = None

    @staticmethod
    def get_git_metadata():
        metadata = {
            "commit": "unknown",
            "branch": "unknown",
            "dirty": False
        }
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            metadata["commit"] = commit
        except (OSError, subprocess.SubprocessError) as e:
            print(f"⚠️ Could not read git commit ({e}); recording 'unknown'.")

        try:
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            metadata["branch"] = branch
        except (OSError, subprocess.SubprocessError) as e:
            print(f"⚠️ Could not read git branch ({e}); recording 'unknown'.")

        try:
            status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
            metadata["dirty"] = len(status) > 0
        except (OSError, subprocess.SubprocessError) as e:
            print(f"⚠️ Could not read git status ({e}); recording dirty=False.")

        return metadata

    @staticmethod
    def get_hyperparameters():
        return {
            "LATENT_DIM": LATENT_DIM,
            "NUM_BLOCKS": NUM_BLOCKS,
            "SHARED_SLOTS": SHARED_SLOTS,
            "MAX_SEQ_LEN": MAX_SEQ_LEN,
            "VOCAB_SIZE": VOCAB_SIZE,
            "MAX_STEPS_LIMIT": MAX_STEPS_LIMIT,
            "BATCH_SIZE": BATCH_SIZE,
            "ACCUMULATION_STEPS": ACCUMULATION_STEPS,
            "PAD_TOKEN_ID": PAD_TOKEN_ID,
            "NUM_HEADS": NUM_HEADS,
            "NUM_GROUPS": NUM_GROUPS,
        }

    def _check_compatibility(self, metadata_path):
        if not os.path.exists(metadata_path):
            return
        
        try:
            with open(metadata_path, "r") as f:
                old_meta = json.load(f)
            
            old_params = old_meta.get("parameters", {})
            current_params = self.get_hyperparameters()
            
            critical_keys = [
                "LATENT_DIM", "NUM_BLOCKS", "SHARED_SLOTS", "MAX_SEQ_LEN", 
                "VOCAB_SIZE", "NUM_HEADS", "NUM_GROUPS"
            ]
            mismatches = [
                f"  - {k}: run used {old_params[k]}, current code uses {current_params[k]}"
                for k in critical_keys
                if k in old_params and old_params[k] != current_params[k]
            ]
            
            if mismatches:
                print("\n" + "🛑"*20)
                print("🛑 ERROR: Parameter Mismatch Detected! Cannot resume this training run:")
                print("\n".join(mismatches))
                print("\n💡 Options:")
                print("  1. Revert your code parameters back to match the run's parameters.")
                print("  2. Start a brand new training run with: python start_training.py --new-run")
                print("  3. Point to a different checkpoint folder with: python start_training.py --checkpoint-path <path>")
                print("🛑"*20 + "\n")
                sys.exit(1)
        except SystemExit:
            raise
        except (OSError, json.JSONDecodeError, KeyError) as e:
            # A malformed metadata file must not kill training, but the disabled
            # compatibility check must be visible.
            print(f"⚠️ Could not verify run compatibility from {metadata_path}: {e}")

    def start_session(self, run_id=None):
        os.makedirs(self.runs_root, exist_ok=True)
        self.start_time = time.time()
        start_timestamp = datetime.datetime.now().astimezone().isoformat()

        if run_id is None:
            # Generate a new unique run ID
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"run_{timestamp_str}"
            self.run_dir = os.path.join(self.runs_root, self.run_id)
            os.makedirs(self.run_dir, exist_ok=True)

            git_meta = self.get_git_metadata()
            params = self.get_hyperparameters()

            metadata = {
                "run_id": self.run_id,
                "git_commit": git_meta["commit"],
                "git_branch": git_meta["branch"],
                "git_dirty": git_meta["dirty"],
                "parameters": params,
                "sections": [
                    {
                        "start_time": start_timestamp,
                        "end_time": None,
                        "duration_seconds": None
                    }
                ]
            }
            self.session_index = 0
            self.save_metadata(metadata)
            print(f"📁 Created new training run folder: {self.run_dir}")
        else:
            # Resume existing run
            self.run_id = run_id
            self.run_dir = os.path.join(self.runs_root, self.run_id)
            os.makedirs(self.run_dir, exist_ok=True)

            metadata_path = os.path.join(self.run_dir, "run_metadata.json")
            self._check_compatibility(metadata_path)

            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                except (OSError, json.JSONDecodeError) as e:
                    print(f"⚠️ Could not read {metadata_path} ({e}); regenerating run metadata.")
                    metadata = None
            else:
                metadata = None

            if metadata is None:
                git_meta = self.get_git_metadata()
                params = self.get_hyperparameters()
                metadata = {
                    "run_id": self.run_id,
                    "git_commit": git_meta["commit"],
                    "git_branch": git_meta["branch"],
                    "git_dirty": git_meta["dirty"],
                    "parameters": params,
                    "sections": []
                }

            metadata["sections"].append({
                "start_time": start_timestamp,
                "end_time": None,
                "duration_seconds": None
            })
            self.session_index = len(metadata["sections"]) - 1
            self.save_metadata(metadata)
            print(f"🔄 Resumed training run folder: {self.run_dir}")

        return self.run_id

    def update_session_duration(self):
        if self.run_dir is None or self.session_index is None:
            return
        metadata_path = os.path.join(self.run_dir, "run_metadata.json")
        if not os.path.exists(metadata_path):
            return
        
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            end_timestamp = datetime.datetime.now().astimezone().isoformat()
            duration = time.time() - self.start_time

            metadata["sections"][self.session_index]["end_time"] = end_timestamp
            metadata["sections"][self.session_index]["duration_seconds"] = round(duration, 2)

            self.save_metadata(metadata)
        except (OSError, json.JSONDecodeError, KeyError, IndexError) as e:
            # Metadata bookkeeping must never kill training, but failures stay visible.
            print(f"⚠️ Could not update session duration in {metadata_path}: {e}")

    def save_metadata(self, metadata):
        metadata_path = os.path.join(self.run_dir, "run_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
