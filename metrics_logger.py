import csv
import fsspec
import jax.numpy as jnp

class LossMonitor:
    def __init__(self, patience=1500, window=50):
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
        # Keys to extract from halt_diag
        self.diag_keys = [
            'expected_steps', 'temporal_drift', 'forget_density', 
            'diversity_loss', 'saturation', 'ponder_kl', 'tau'
        ]
        # Full set of fields for CSV
        self.fields = [
            "step", "ce", "loss", "first_ce", "avg_ponder", "expected_steps",
            "grad_norm_avg", "avg_forget_cost", "avg_storage_cost",
            "diversity_loss", "temporal_drift", "forget_density", "saturation", "tau"
        ]

    def extract_diags(self, halt_diag, jnp_mean_fn):
        """Extracts and formats diagnostics from the model step using a provided mean function."""
        return {k: float(jnp_mean_fn(halt_diag.get(k, 0))) for k in self.diag_keys}

    def log(self, step, ce, loss, out, compute_time, 
            grad_norm_avg=None, first_ce=None):
        """Logs training metrics to console and CSV based on the routing specification."""
        diag_dict = self.extract_diags(out.halt_diag, jnp.mean)
        
        # Log to BOTH and TERMINAL ONLY
        print(
            f"Step {step:04d} | CE: {ce:.4f} (first: {first_ce:.4f}) | "
            f"Ponder: {out.ponder_cost:.4f} | Tau: {diag_dict.get('tau', 0):.4f}\n"
            f"      Loss: {loss:.4f} | Saturation: {diag_dict.get('saturation', 0):.1f}% | "
            f"Steps: {diag_dict.get('expected_steps', 0):.2f} | Compute: {compute_time:.3f}s"
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
            writer = csv.DictWriter(f, fieldnames=self.fields, extrasaction='ignore')
            if file_is_empty: 
                writer.writeheader()
            
            row = {
                "step": int(step), 
                "ce": f"{ce:.4f}",
                "loss": f"{loss:.4f}",
                "first_ce": f"{first_ce:.4f}" if first_ce is not None else "",
                "avg_ponder": f"{out.ponder_cost:.4f}",
                "expected_steps": f"{diag_dict.get('expected_steps', 0):.4f}",
                "grad_norm_avg": f"{grad_norm_avg:.4f}" if grad_norm_avg is not None else "",
                "avg_forget_cost": f"{out.forget_cost:.4f}", 
                "avg_storage_cost": f"{out.storage_cost:.4f}",
                "diversity_loss": f"{out.diversity_loss:.6f}",
                "temporal_drift": f"{diag_dict.get('temporal_drift', 0):.6f}",
                "forget_density": f"{diag_dict.get('forget_density', 0):.6f}",
                "saturation": f"{diag_dict.get('saturation', 0):.4f}",
                "tau": f"{diag_dict.get('tau', 0):.6f}",
            }
            writer.writerow(row)
