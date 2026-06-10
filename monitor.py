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

    def push(self, step, ce_loss, total_loss):
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
