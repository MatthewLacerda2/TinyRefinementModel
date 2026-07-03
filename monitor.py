class LossMonitor:
    """Tracks CE/loss bests for checkpointing and detects plateaus for phase changes."""

    def __init__(self, patience=400, window=20, min_delta=0.005):
        self.patience = patience
        self.window = window
        self.min_delta = min_delta
        self.ce_history = []
        self.best_ce = float("inf")
        self.best_loss = float("inf")
        self.best_avg_ce = float("inf")
        self.last_improvement_step = 0
        self.is_new_best = False
        # Step at which the SFT phase began; None while still pretraining.
        self.sft_start_step = None

    def reset_for_new_phase(self, step):
        """Forget the previous phase's plateau state so the new phase gets a
        fresh patience window and fresh bests. The attribute set must stay
        stable — checkpoint_utils serializes these fields by name."""
        self.ce_history = []
        self.best_ce = float("inf")
        self.best_loss = float("inf")
        self.best_avg_ce = float("inf")
        self.last_improvement_step = step

    def push(self, step, ce_loss, total_loss):
        """Record one logging-window observation.

        Returns True when the windowed CE average has not improved for
        `patience` steps — the plateau signal the trainer acts on (switch to
        SFT, or halt if already there). Side effect: sets `is_new_best`, which
        the trainer reads for best-CE checkpointing.
        """
        self.ce_history.append(ce_loss)
        if len(self.ce_history) > self.window:
            self.ce_history.pop(0)

        # Record-breaking logic (raw metrics)
        self.is_new_best = False
        if ce_loss < self.best_ce:
            self.best_ce = ce_loss
            self.is_new_best = True

        if total_loss < self.best_loss:
            self.best_loss = total_loss
            self.is_new_best = True

        # Early stopping logic (windowed average)
        avg_ce = sum(self.ce_history) / len(self.ce_history)
        # Using a small epsilon for early stopping stability as before
        # But we decouple this from the "is_new_best" flag used for checkpointing
        if avg_ce < (self.best_avg_ce - self.min_delta):
            self.best_avg_ce = avg_ce
            self.last_improvement_step = step
            return False

        return (step - self.last_improvement_step) > self.patience
