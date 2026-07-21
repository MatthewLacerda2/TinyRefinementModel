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
    DATA_SEED,
    MODEL_SEED,
    TRAIN_TOKEN_BUDGET,
    MODEL_ARCH,
)
from schedules import DECAY_STEPS

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
    def capture_environment_snapshot(run_dir):
        """Freeze everything a revival (tools/timemachine.py) needs beyond the commit
        SHA: the dirty working tree, the pinned Python libs, and the host it assumed.

        These make a run self-describing. Until now they were written by hand (only
        one run ever had them), so reproducibility was accidental; capturing them
        here makes every future run revivable by construction. Best-effort: a failure
        to snapshot must never take down a training launch.
        """
        # Every shell-out below carries a timeout: a raised error is caught, but a
        # *hang* (wedged D-state nvidia-smi, a stuck pip) is not — without the timeout
        # it would stall every launch. TimeoutExpired is a SubprocessError, so the
        # existing excepts already handle it once it fires.

        # 1. Pinned libs — the second half of the compat surface (code is the first).
        try:
            freeze = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"],
                stderr=subprocess.DEVNULL, timeout=120)
            with open(os.path.join(run_dir, "env_freeze.txt"), "wb") as f:
                f.write(freeze)
        except (OSError, subprocess.SubprocessError) as e:
            print(f"⚠️ Could not capture pip freeze ({e}); libs not pinned.")

        # 2. The host the venv assumed — driver/GPU/python. Not part of the compat
        #    surface (driver is a shared passthrough) but invaluable for debugging a
        #    failed revival. MODEL_ARCH also rides in run_metadata.json, machine-readable.
        lines = [f"python {sys.version.split()[0]}", f"platform {sys.platform}",
                 f"MODEL_ARCH {MODEL_ARCH}"]
        try:
            smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version,name",
                 "--format=csv,noheader"], stderr=subprocess.DEVNULL,
                timeout=30).decode().strip()
            lines.append(f"gpu {smi}")
        except (OSError, subprocess.SubprocessError):
            lines.append("gpu (nvidia-smi unavailable)")
        try:
            with open(os.path.join(run_dir, "system_snapshot.txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
        except OSError as e:  # e.g. disk full at run creation on the near-full root fs
            print(f"⚠️ Could not write system_snapshot.txt ({e}).")

        RunTracker.capture_worktree_snapshot(run_dir)

    @staticmethod
    def capture_worktree_snapshot(run_dir):
        """Freeze the working tree so a dirty launch is reproducible.

        `git_dirty` records *that* the tree diverged from HEAD; this records *how*.
        Without it, reviving a weight (tools/timemachine.py) can only reconstruct the
        commit, not the uncommitted edits that were actually live at launch. We save
        the tracked-file diff as a patch the time machine re-applies onto the worktree,
        plus the list of untracked non-ignored files (whose contents we deliberately
        do NOT copy — bloat/surprise risk — but warn about on reconstruction).
        """
        try:
            patch = subprocess.check_output(
                ["git", "diff", "HEAD"], stderr=subprocess.DEVNULL, timeout=60)
            with open(os.path.join(run_dir, "worktree.patch"), "wb") as f:
                f.write(patch)
        except (OSError, subprocess.SubprocessError) as e:
            print(f"⚠️ Could not capture worktree patch ({e}); dirty state not saved.")

        try:
            untracked = subprocess.check_output(
                ["git", "ls-files", "--others", "--exclude-standard"],
                stderr=subprocess.DEVNULL, timeout=60).decode()
            with open(os.path.join(run_dir, "worktree.untracked.txt"), "w") as f:
                f.write(untracked)
        except (OSError, subprocess.SubprocessError) as e:
            print(f"⚠️ Could not list untracked files ({e}).")

    @staticmethod
    def get_hyperparameters():
        return {
            # Which arch built the param tree — the two arches are not
            # checkpoint-compatible, so a faithful revival (tools/timemachine.py)
            # must rebuild the same skeleton. Recorded machine-readably here.
            "MODEL_ARCH": MODEL_ARCH,
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
            "DATA_SEED": DATA_SEED,
            "MODEL_SEED": MODEL_SEED,
            # The run's recipe horizon (#83): budget in, resolved anneal out.
            "TRAIN_TOKEN_BUDGET": TRAIN_TOKEN_BUDGET,
            "DECAY_STEPS": DECAY_STEPS,
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
            self.capture_environment_snapshot(self.run_dir)
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
