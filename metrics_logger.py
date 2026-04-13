import csv
import fsspec
import jax.numpy as jnp

class LossMonitor:
    def __init__(self, patience=2000, window=1000):
        self.patience = patience
        self.window = window
        self.ce_history = []
        self.best_ce = float("inf")
        self.best_loss = float("inf")
        self.best_avg_ce = float("inf")
        self.last_improvement_step = 0
        self.is_new_best = False

    def push(self, step, ce_loss, total_loss):
        self.ce_history.append(ce_loss)
        if len(self.ce_history) > self.window: self.ce_history.pop(0)

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
        if avg_ce < (getattr(self, 'best_avg_ce', float('inf')) - 0.01):
            self.best_avg_ce = avg_ce
            self.last_improvement_step = step
            return False
        
        return (step - self.last_improvement_step) > self.patience


class MetricsLogger:
    def __init__(self, history_file):
        self.history_file = history_file
        self.diag_keys = [
            'logits_mean', 'logits_std', 'logits_min', 'logits_max', 
            'prob_mean', 'prob_std', 'saturation', 'temporal_drift', 
            'forget_density', 'logit_spread', 'diversity_loss'
        ]
        self.fields = ["step", "loss", "ce", "avg_ponder", "avg_forget_cost", "avg_storage_cost",
                       "t_total", "compute_time",
                       "grad_norm_avg",
                       "logit_drift_intra", "first_ce"] + self.diag_keys

    def extract_diags(self, halt_diag, jnp_mean_fn):
        """Extracts and formats diagnostics from the model step using a provided mean function."""
        return {k: float(jnp_mean_fn(halt_diag.get(k, 0))) for k in self.diag_keys}

    def log(self, step, loss, out, t_total, compute,
            grad_norm_avg=None, logit_drift=None, first_ce=None):
        """Logs training metrics to console and CSV."""
        diag_dict = self.extract_diags(out.halt_diag, jnp.mean)
        
        grad_line = ""
        if grad_norm_avg is not None:
            grad_line = (
                f"\n      GradNorm [avg:{grad_norm_avg:.4f}]"
                f" | IntraStep Logit Δ: {logit_drift:.5f}"
                f" | CE μ→micro[0]:{first_ce:.4f}"
            )

        # Assuming 'ce' is part of the loss components if we want to log it specifically
        # For now, we'll assume 'loss' passed in is the CE (token loss) as per trainer logic
        ce = out.halt_diag.get('token_loss', 0.0) # Or we pass it explicitly
        
        print(
            f"Step {step:04d} | CE: {loss:.4f} | " # In this context loss is usually the token_loss
            f"Ponder(KL): {out.ponder_cost:.4f} | Forget: {out.forget_cost:.4f} | Storage: {out.storage_cost:.4f} | Time: {t_total:.2f}s\n"
            f"      Compute: {compute:.3f}s\n"
            f"      Logits [μ:{diag_dict.get('logits_mean',0):.2f}, σ:{diag_dict.get('logits_std',0):.2f}] | Spread: {diag_dict.get('logit_spread',0):.2f}\n"
            f"      Prob [μ:{diag_dict.get('prob_mean',0):.3f}, σ:{diag_dict.get('prob_std',0):.3f}] | Sat:{diag_dict.get('saturation',0):.3f}| Drift:{diag_dict.get('temporal_drift',0):.3f}| Density:{diag_dict.get('forget_density',0):.3f}\n"
            f"      Diversity: {diag_dict.get('diversity_loss',0):.3f}"
            + grad_line
        )

        # Check if file exists and has content to avoid duplicate headers
        file_is_empty = True
        try:
            fs, path = fsspec.core.url_to_fs(self.history_file)
            if fs.exists(path) and fs.size(path) > 0:
                file_is_empty = False
        except:
            # Fallback if filesystem check fails
            pass

        with fsspec.open(self.history_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            if file_is_empty: 
                writer.writeheader()
            
            row = {
                "step": int(step), "loss": f"{loss:.4f}", "ce": f"{loss:.4f}",
                "avg_ponder": f"{out.ponder_cost:.4f}", "avg_forget_cost": f"{out.forget_cost:.4f}", 
                "avg_storage_cost": f"{out.storage_cost:.4f}",
                "t_total": f"{t_total:.2f}", "compute_time": f"{compute:.4f}",
                "grad_norm_avg": f"{grad_norm_avg:.4f}" if grad_norm_avg is not None else "",
                "logit_drift_intra": f"{logit_drift:.5f}" if logit_drift is not None else "",
                "first_ce": f"{first_ce:.4f}" if first_ce is not None else "",
            }
            row.update({k: f"{v:.4f}" for k, v in diag_dict.items() if k in self.fields})
            writer.writerow(row)
