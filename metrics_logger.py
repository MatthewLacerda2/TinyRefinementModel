import csv
import fsspec
import os # Added import for os module

class LossMonitor:
    def __init__(self, patience=10000, window=1000): # Changed patience from 2000 to 10000
        self.patience = patience
        self.window = window
        self.ce_history = []
        self.best_ce = float("inf")
        self.last_improvement_step = 0

    def push(self, step, ce_loss):
        self.ce_history.append(ce_loss)
        if len(self.ce_history) > self.window: self.ce_history.pop(0)
        avg_ce = sum(self.ce_history) / len(self.ce_history)
        if avg_ce < (self.best_ce - 0.01):
            self.best_ce = avg_ce
            self.last_improvement_step = step
            return False
        return (step - self.last_improvement_step) > self.patience


class MetricsLogger:
    def __init__(self, filename): # Changed history_file to filename
        self.filename = filename # Changed history_file to filename
        self.diag_keys = [
            'logits_mean', 'logits_std', 'logits_min', 'logits_max', 
            'prob_mean', 'prob_std', 'saturation', 'temporal_drift', 
            'forget_density', 'logit_spread', 'diversity_loss'
        ]
        self.fields = ["step", "loss", "ce", "avg_ponder", "avg_forget_cost", 
                       "t_total", "data_wait", "compute_time"] + self.diag_keys

        # Only write header if file doesn't exist or is empty
        if not os.path.exists(self.filename) or os.path.getsize(self.filename) == 0:
            with fsspec.open(self.filename, 'w', newline="") as f: # Use fsspec.open for consistency
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def extract_diags(self, halt_diag, jnp_mean_fn):
        """Extracts and formats diagnostics from the model step using a provided mean function."""
        return {k: float(jnp_mean_fn(halt_diag.get(k, 0))) for k in self.diag_keys}

    def log(self, step, loss, ce, p, forget_cost, t_total, wait, compute, diag_dict):
        print(
            f"Step {step:04d} | CE: {ce:.4f} | Agg Loss: {loss:.4f} | "
            f"Avg Steps: {p:.2f} | Forget: {forget_cost:.4f} | Time: {t_total:.2f}s\n"
            f"      Wait: {wait:.3f}s | Compute: {compute:.3f}s\n"
            f"      Logits [μ:{diag_dict.get('logits_mean',0):.2f}, σ:{diag_dict.get('logits_std',0):.2f}, min:{diag_dict.get('logits_min',0):.2f}, max:{diag_dict.get('logits_max',0):.2f}] | Spread: {diag_dict.get('logit_spread',0):.2f}\n"
            f"      Prob [μ:{diag_dict.get('prob_mean',0):.3f}, σ:{diag_dict.get('prob_std',0):.3f}] | Sat: {diag_dict.get('saturation',0):.3f} | Drift: {diag_dict.get('temporal_drift',0):.3f} | Forget: {diag_dict.get('forget_density',0):.3f}"
        )

        with fsspec.open(self.history_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            if f.tell() == 0: 
                writer.writeheader()
            
            row = {
                "step": int(step), "loss": f"{loss:.4f}", "ce": f"{ce:.4f}",
                "avg_ponder": f"{p:.2f}", "avg_forget_cost": f"{forget_cost:.4f}", 
                "t_total": f"{t_total:.2f}", "data_wait": f"{wait:.4f}", "compute_time": f"{compute:.4f}"
            }
            row.update({k: f"{v:.4f}" for k, v in diag_dict.items() if k in self.fields})
            writer.writerow(row)
