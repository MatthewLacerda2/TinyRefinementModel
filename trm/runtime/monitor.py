from trm.config import PLATEAU_MIN_DELTA, PLATEAU_PATIENCE


class LossMonitor:
    """Tracks the held-out best for checkpointing and detects CE plateaus. A plateau is
    reported, never acted on: the in-run SFT flip it used to trigger was removed (#323)."""

    def __init__(self, patience=PLATEAU_PATIENCE, window=4, min_delta=PLATEAU_MIN_DELTA):
        # The plateau bar is config's (#318): these defaults used to be literals,
        # and drifted to 0.005 while config raised PLATEAU_MIN_DELTA to 0.01.
        # `window` counts VALIDATION readings (one per VAL_EVERY_OPT_STEPS), not
        # logging rows: 4 readings at the 64-step cadence is a 256-step smoothing.
        self.patience = patience
        self.window = window
        self.min_delta = min_delta
        self.ce_history = []
        self.best_ce = float("inf")
        self.best_loss = float("inf")
        self.best_avg_ce = float("inf")
        # Best held-out CE, the one thing `best_val_ce/` is selected on (#222).
        self.best_val_ce = float("inf")
        self.last_improvement_step = 0
        self._plateaued = False
        # Samples the data pipeline has served (#24). Restored from the
        # checkpoint rather than re-derived from the step count, so a resume
        # seeks correctly even if BATCH_SIZE changed between runs.
        self.samples_seen = 0
        # The data loader's exact state after the last batch consumed (#424), saved
        # with every checkpoint so a resume reads the rows the run would have.
        self.data_state = None

    def push(self, step, ce_loss, total_loss):
        """Record one logging-window observation of TRAIN CE: the raw bests only.

        The plateau signal used to be computed here, on train CE, which the
        curriculum moves underneath it (#184); it now lives in push_val.
        """
        self.best_ce = min(self.best_ce, ce_loss)
        self.best_loss = min(self.best_loss, total_loss)

    @property
    def plateaued(self):
        """Whether the windowed held-out CE has failed to improve by `min_delta`
        for more than `patience` opt steps, as of the last push_val."""
        return self._plateaued

    def push_val(self, val_ce, step=None):
        """Record one held-out CE. True when it is a new best — the trigger for
        saving `best_val_ce/`. With `step`, also advances the plateau detector:
        `ce_history` (checkpointed) is the window of recent val readings.

        Train CE used to be the trigger (#222). It is per-batch noisy, so the
        tracker latched onto a lucky window early and never beat it again while
        val CE kept improving: on the 4B run `best/` ended 6,000 steps stale, and
        each lucky window also cost a 1.7GB write (#174). Val CE is scored on the
        same held-out text every time, so an improvement is an improvement.
        """
        is_best = val_ce < self.best_val_ce
        self.best_val_ce = min(self.best_val_ce, val_ce)
        if step is not None:
            self.ce_history.append(val_ce)
            if len(self.ce_history) > self.window:
                self.ce_history.pop(0)
            avg_ce = sum(self.ce_history) / len(self.ce_history)
            if avg_ce < (self.best_avg_ce - self.min_delta):
                self.best_avg_ce = avg_ce
                self.last_improvement_step = step
            self._plateaued = (step - self.last_improvement_step) > self.patience
        return is_best
