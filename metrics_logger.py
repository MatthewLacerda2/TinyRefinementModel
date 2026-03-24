import csv
import fsspec
import os

class LossMonitor:
    def __init__(self, patience=100000, window=5000): # Effectively immortal for 140k steps
        self.patience = patience
        self.window = window
        self.ce_history = []
        self.best_ce = float("inf")
        self.last_improvement_step = 0

    def push(self, step, ce_loss):
        self.ce_history.append(float(ce_loss))
        if len(self.ce_history) > self.window: self.ce_history.pop(0)
        avg_ce = sum(self.ce_history) / len(self.ce_history)
        if avg_ce < (self.best_ce - 0.01):
            self.best_ce = avg_ce
            self.last_improvement_step = step
            return False
        return (step - self.last_improvement_step) > self.patience


class MetricsLogger:
    def __init__(self, filename):
        self.filename = filename
        self.diag_keys = [
            'logits_mean', 'logits_std', 'logits_min', 'logits_max', 
            'prob_mean', 'prob_std', 'saturation', 'temporal_drift', 
            'forget_density', 'logit_spread', 'diversity_loss'
        ]
        self.fields = ["step", "loss", "ce", "avg_ponder", "avg_forget_cost", 
                       "t_total", "data_wait", "compute_time"] + self.diag_keys

        # Only write header if file doesn't exist or is empty
        if not os.path.exists(self.filename) or os.path.getsize(self.filename) == 0:
            with fsspec.open(self.filename, 'w', newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def extract_diags(self, halt_diag, jnp_mean_fn):
        """Extracts and formats diagnostics from the model step using a provided mean function."""
        return {k: float(jnp_mean_fn(halt_diag.get(k, 0))) for k in self.diag_keys}

    def log(self, step, loss, ce, p_cost, f_cost, t_total, d_wait, c_time, diags):
        print(
            f"Step {step:04d} | CE: {ce:.4f} | Agg Loss: {loss:.4f} | "
            f"Avg Steps: {p_cost:.2f} | Forget: {f_cost:.4f} | Time: {t_total:.2f}s\n"
            f"      Wait: {d_wait:.3f}s | Compute: {c_time:.3f}s\n"
            f"      Logits [μ:{diags.get('logits_mean',0):.2f}, σ:{diags.get('logits_std',0):.2f}, min:{diags.get('logits_min',0):.2f}, max:{diags.get('logits_max',0):.2f}] | Spread: {diags.get('logit_spread',0):.2f}\n"
            f"      Prob [μ:{diags.get('prob_mean',0):.3f}, σ:{diags.get('prob_std',0):.3f}] | Sat: {diags.get('saturation',0):.3f} | Drift: {diags.get('temporal_drift',0):.3f} | Forget: {diags.get('forget_density',0):.3f}"
        )
        row = {
            "step": step, "loss": loss, "ce": ce, 
            "avg_ponder": p_cost, "avg_forget_cost": f_cost,
            "t_total": t_total, "data_wait": d_wait, "compute_time": c_time,
            **diags
        }
        with fsspec.open(self.filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(row)
