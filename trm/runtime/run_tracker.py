import datetime
import json
import os
import subprocess
import sys
import time

from trm.config import VOCAB_SIZE
from trm.runtime.run_metadata import METADATA_FILENAME, RunMetadata, Section
from trm.train.schedules import Schedules

# What the model's param tree is built from (#317). A resume that changes one of these
# cannot load its checkpoint, or loads it into a different network whose tree happens
# to match.
TREE_KEYS = ("LATENT_DIM", "VOCAB_SIZE", "NUM_HEADS", "MAX_SEQ_LEN", "PLAIN_LAYERS", "POST_NORM")
# Recorded values computed from knobs, not knobs themselves: one may differ on a resume
# only because the knob it comes from was set on purpose (#535).
DERIVED_FROM = {"ACCUMULATION_STEPS": ("BATCH_SIZE",), "TOKENS_PER_OPT_STEP": ("BATCH_SIZE",),
                "DECAY_STEPS": ("TRAIN_TOKEN_BUDGET", "LR_SCHEDULE", "WSD_DECAY_START"),
                "CURRICULUM_STEPS": ("TRAIN_TOKEN_BUDGET",)}

class RunTracker:
    """A run's folder and its run_metadata.json. `config` is the run's Config, the one
    the trainer is handed (#475): what the metadata records and what a resume is
    checked against."""

    def __init__(self, config, runs_root="runs"):
        self.config = config
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
        """Freeze everything a revival (instruments.timemachine) needs beyond the commit
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
        #    failed revival.
        lines = [f"python {sys.version.split()[0]}", f"platform {sys.platform}"]
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
        Without it, reviving a weight (instruments.timemachine) can only reconstruct the
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
    def get_hyperparameters(config):
        """What run_metadata.json records as the run's parameters: every knob of its
        Config, whole (so none can be left out: #358, #475), plus what a reader cannot
        recover from the knobs alone."""
        schedules = Schedules.of(config)
        return {
            **config.model_dump(),
            # A constant, not a knob, that shapes the param tree (TREE_KEYS).
            "VOCAB_SIZE": VOCAB_SIZE,
            # The horizons resolved from the budget: the LR anneal's (#83) and the
            # mixture ramp's (#362). A resume checks the first against the run's (#197).
            "DECAY_STEPS": schedules.decay_steps,
            "CURRICULUM_STEPS": schedules.curriculum_steps,
        }

    def _check_compatibility(self, old):
        """Refuse a resume whose recorded parameters the current code would change
        silently. `old` is the run's RunMetadata; None (no readable file) checks nothing."""
        if old is None:
            return
        old_params = old.parameters
        current_params = self.get_hyperparameters(self.config)

        # Compared as recorded: through JSON, so a tuple and its list are equal.
        current_params = json.loads(json.dumps(current_params))
        # A key the run's metadata predates (or that no longer exists) is skipped.
        changed = [k for k in old_params.keys() & current_params.keys() if old_params[k] != current_params[k]]
        # The param tree can never change. A recipe knob may, but only on purpose:
        # set for this launch (Config.model_fields_set: the knobs the environment
        # gave). A default that moved in the checked-out code since the run started
        # would otherwise change the run silently (#535).
        set_now = self.config.model_fields_set
        asked = {k for k in changed if k in set_now or set_now.intersection(DERIVED_FROM.get(k, ()))}
        mismatches = [
            f"  - {k}: run used {old_params[k]}, current code uses {current_params[k]}"
            + ("" if k in TREE_KEYS else " (a recipe default moved; set it in the environment to keep or change it on purpose)")
            for k in sorted(changed) if k in TREE_KEYS or k not in asked
        ]
        if asked and not mismatches:
            print("⚠️ Resuming with knobs changed on purpose (set in the environment): "
                  + ", ".join(f"{k} {old_params[k]} -> {current_params[k]}" for k in sorted(asked)))

        if mismatches:
            # Raised, not sys.exit'd: a caller (or a test) can catch it, and an
            # uncaught one still ends the process with exit code 1 and this text.
            raise SystemExit("\n".join([
                "\n" + "🛑" * 20,
                "🛑 ERROR: Parameter Mismatch Detected! Cannot resume this training run:",
                *mismatches,
                "\n💡 Options:",
                "  1. Revert your code parameters back to match the run's parameters.",
                "  2. Start a brand new training run with: python -m trm.train.start --new-run",
                "  3. Point to a different checkpoint folder with: "
                "python -m trm.train.start --checkpoint-path <path>",
                "🛑" * 20 + "\n",
            ]))

    def _fresh_metadata(self):
        """run_metadata.json for a run that has none yet: its code, its parameters,
        and no sessions."""
        git_meta = self.get_git_metadata()
        return RunMetadata(run_id=self.run_id, git_commit=git_meta["commit"], git_branch=git_meta["branch"],
                           git_dirty=git_meta["dirty"],
                           parameters=json.loads(json.dumps(self.get_hyperparameters(self.config))))

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

            metadata = self._fresh_metadata()
            metadata.sections.append(Section(start_time=start_timestamp))
            self.session_index = 0
            metadata.write(self.run_dir)
            self.capture_environment_snapshot(self.run_dir)
            print(f"📁 Created new training run folder: {self.run_dir}")
        else:
            # Resume existing run
            self.run_id = run_id
            self.run_dir = os.path.join(self.runs_root, self.run_id)
            os.makedirs(self.run_dir, exist_ok=True)

            metadata = RunMetadata.read(self.run_dir)
            self._check_compatibility(metadata)
            if metadata is None:
                if os.path.exists(os.path.join(self.run_dir, METADATA_FILENAME)):
                    print(f"⚠️ Could not read {self.run_dir}/{METADATA_FILENAME}; regenerating run metadata.")
                metadata = self._fresh_metadata()

            metadata.sections.append(Section(start_time=start_timestamp))
            self.session_index = len(metadata.sections) - 1
            metadata.write(self.run_dir)
            # Snapshot on resume too (#173): each session describes its own commit
            # and edits. Why, and the guard: tests/core/test_run_tracker_snapshot.py.
            self.capture_environment_snapshot(self.run_dir)
            print(f"🔄 Resumed training run folder: {self.run_dir}")

        return self.run_id

    def update_session_duration(self):
        if self.run_dir is None or self.session_index is None or self.start_time is None:
            return
        try:
            metadata = RunMetadata.read(self.run_dir)
            if metadata is None:
                return
            section = metadata.sections[self.session_index]
            section.end_time = datetime.datetime.now().astimezone().isoformat()
            section.duration_seconds = round(time.time() - self.start_time, 2)
            metadata.write(self.run_dir)
        except (OSError, ValueError, IndexError) as e:
            # Metadata bookkeeping must never kill training, but failures stay visible.
            print(f"⚠️ Could not update session duration in {self.run_dir}/{METADATA_FILENAME}: {e}")
