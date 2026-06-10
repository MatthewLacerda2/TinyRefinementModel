import csv
import fsspec
import jax.numpy as jnp


class MetricsLogger:
    def __init__(self, history_file):
        self.history_file = history_file
        self.diag_keys = [
            'temporal_drift', 'forget_density', 
            'diversity_loss', 'tau', 'mean_halt_step',
        ]
        # Full set of fields for CSV
        self.fields = [
            "step", "ce", "loss", "first_ce",
            "grad_norm_avg", "avg_forget_cost",
            "diversity_loss", "temporal_drift", "forget_density", "tau",
            "mean_halt_step", "ponder_cost",
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
            f"Tau: {diag_dict.get('tau', 0):.4f} | Halt@: {diag_dict.get('mean_halt_step', 0):.2f}\n"
            f"      Loss: {loss:.4f} | Drift: {diag_dict.get('temporal_drift', 0):.6f} | "
            f"Compute: {compute_time:.3f}s"
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
                "grad_norm_avg": f"{grad_norm_avg:.4f}" if grad_norm_avg is not None else "",
                "avg_forget_cost": f"{out.forget_cost:.4f}", 
                "diversity_loss": f"{out.diversity_loss:.6f}",
                "temporal_drift": f"{diag_dict.get('temporal_drift', 0):.6f}",
                "forget_density": f"{diag_dict.get('forget_density', 0):.6f}",
                "tau": f"{diag_dict.get('tau', 0):.6f}",
                "mean_halt_step": f"{diag_dict.get('mean_halt_step', 0):.4f}",
                "ponder_cost": f"{out.ponder_cost:.4f}",
            }
            writer.writerow(row)
