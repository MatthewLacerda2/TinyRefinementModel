import csv
import math
import fsspec
import jax.numpy as jnp


class MetricsLogger:
    def __init__(self, history_file, start_opt_step=None):
        self.history_file = history_file
        self.diag_keys = [
            'temporal_drift', 'forget_density',
            'diversity_loss', 'tau',
        ]
        # Full set of fields for CSV
        self.fields = [
            "step", "ce", "loss", "seg1_ce",
            "grad_norm_avg", "avg_forget_cost",
            "diversity_loss", "temporal_drift", "forget_density", "tau",
            "depth_avg", "val_ce",
        ]
        # Warn once per metric name when a non-finite value shows up, so a broken
        # diagnostic can't silently fill the CSV with NaN.
        self._warned_nonfinite = set()
        if start_opt_step is not None:
            self._truncate_replayed_rows(start_opt_step)

    def _truncate_replayed_rows(self, start_opt_step):
        """On resume, drop rows at/after the restored step. Checkpoints restore to
        the last *best* step, which can be earlier than the last logged row — without
        trimming, every resume appends an overlapping step range to the CSV."""
        try:
            fs, path = fsspec.core.url_to_fs(self.history_file)
            if not fs.exists(path) or fs.size(path) == 0:
                return
            with fsspec.open(self.history_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                old_fields = reader.fieldnames
            kept = [r for r in rows if r.get("step") and int(r["step"]) < start_opt_step]
            # Rewrite also when the schema gained columns, otherwise appended
            # rows would be wider than the existing header.
            if len(kept) == len(rows) and list(old_fields or []) == self.fields:
                return
            print(f"✂️ Trimming {len(rows) - len(kept)} replayed metric rows (step >= {start_opt_step}) from {self.history_file}")
            with fsspec.open(self.history_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(kept)
        except (OSError, ValueError, KeyError) as e:
            print(f"⚠️ Could not trim replayed rows from {self.history_file}: {e}")

    def extract_diags(self, diag, jnp_mean_fn):
        """Extracts and formats diagnostics from the model step using a provided mean function."""
        return {k: float(jnp_mean_fn(diag.get(k, 0))) for k in self.diag_keys}

    def log(self, step, ce, loss, out, compute_time,
            grad_norm_avg=None, seg1_ce=None, depth_avg=None, val_ce=None):
        """Logs training metrics to console and CSV based on the routing specification."""
        diag_dict = self.extract_diags(out.diag, jnp.mean)

        for name, value in {**diag_dict, "ce": ce, "loss": loss}.items():
            if not math.isfinite(value) and name not in self._warned_nonfinite:
                self._warned_nonfinite.add(name)
                print(f"⚠️ Non-finite metric '{name}' ({value}) at step {step} — check the diagnostics pipeline.")

        # Log to BOTH and TERMINAL ONLY
        print(
            f"Step {step:04d} | CE: {ce:.4f} (seg1: {seg1_ce:.4f}) | "
            f"Tau: {diag_dict.get('tau', 0):.4f} | Depth: {depth_avg:.2f}\n"
            f"      Loss: {loss:.4f} | Drift: {diag_dict.get('temporal_drift', 0):.6f} | "
            f"Compute: {compute_time:.3f}s"
        )

        # Check if file exists and has content to avoid duplicate headers
        file_is_empty = True
        try:
            fs, path = fsspec.core.url_to_fs(self.history_file)
            if fs.exists(path) and fs.size(path) > 0:
                file_is_empty = False
        except OSError as e:
            print(f"⚠️ Could not stat {self.history_file} ({e}); assuming empty.")

        with fsspec.open(self.history_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields, extrasaction='ignore')
            if file_is_empty: 
                writer.writeheader()
            
            row = {
                "step": int(step),
                "ce": f"{ce:.4f}",
                "loss": f"{loss:.4f}",
                "seg1_ce": f"{seg1_ce:.4f}" if seg1_ce is not None else "",
                "grad_norm_avg": f"{grad_norm_avg:.4f}" if grad_norm_avg is not None else "",
                "avg_forget_cost": f"{out.forget_cost:.4f}",
                "diversity_loss": f"{out.diversity_loss:.6f}",
                "temporal_drift": f"{diag_dict.get('temporal_drift', 0):.6f}",
                "forget_density": f"{diag_dict.get('forget_density', 0):.6f}",
                "tau": f"{diag_dict.get('tau', 0):.6f}",
                "depth_avg": f"{depth_avg:.4f}" if depth_avg is not None else "",
                "val_ce": f"{val_ce:.4f}" if val_ce is not None else "",
            }
            writer.writerow(row)
