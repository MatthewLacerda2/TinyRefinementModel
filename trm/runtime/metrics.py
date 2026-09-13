import csv
import datetime
import math
import fsspec
import jax.numpy as jnp


def _fmt(diags, key, places):
    """A reported metric, or an empty cell if this architecture doesn't measure it."""
    return f"{diags[key]:.{places}f}" if key in diags else ""


def _arena_peak_mib():
    """peak_bytes_in_use in MiB, or empty where the allocator keeps no statistics
    (CPU, the platform allocator)."""
    import jax
    try:
        stats = jax.local_devices()[0].memory_stats() or {}
    except (AttributeError, RuntimeError):
        stats = {}
    peak = stats.get("peak_bytes_in_use")
    return f"{peak / 2**20:.0f}" if peak else ""


class MetricsLogger:
    def __init__(self, history_file, start_opt_step=None):
        self.history_file = history_file
        # Telemetry this logger knows how to write. A model reports the subset it
        # actually measures (#105) — an architecture without a forget gate simply
        # omits those keys, and their columns stay empty instead of being filled
        # with zeros that look like measurements.
        self.diag_keys = [
            'temporal_drift', 'forget_density',
            'forget_cost', 'diversity_loss', 'tau',
            'out_entropy', 'logz_mean', 'max_abs_logit',
            # Peak |activation| through the stack. The number that decides whether
            # this model can be served in f16 at all, and the one nothing watched
            # during the 4B run: it finished at 65,120 against a 65,504 ceiling and
            # that was found two weeks later, by hand (#235).
            'act_max',
        ]
        # Full set of fields for CSV
        self.fields = [
            "step", "ce", "loss", "seg1_ce",
            "grad_norm_avg", "zero_frac_dense_max", "applied_zero_frac_dense_max", "applied_grad_norm", "avg_forget_cost",
            "diversity_loss", "temporal_drift", "forget_density", "tau",
            "out_entropy", "logz_mean", "max_abs_logit", "act_max",
            "depth_avg", "val_ce",
            # Context a row cannot be read without, and that cannot be backfilled
            # (#186): when it was written — the only clock the run keeps against its
            # progress — and the data mixture its CE was measured on, which the
            # curriculum moves every step.
            "wall_clock", "mix",
            # The allocator's own high-water mark so far (#168): exact, not a poll.
            # Every run records how close it came to its limit, and the fit gate
            # reads it from the probe run's first row.
            "arena_peak_mib",
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
        """Reduces the diagnostics this model reported to plain floats. Keys the
        model did not report are absent, not zero."""
        return {k: float(jnp_mean_fn(diag[k])) for k in self.diag_keys if k in diag}

    def log(self, step, ce, loss, out, compute_time,
            grad_norm_avg=None, seg1_ce=None, depth_avg=None, val_ce=None,
            zero_frac_dense_max=None, applied_zero_frac_dense_max=None, applied_grad_norm=None, mix=None):
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
            f"H: {diag_dict.get('out_entropy', 0):.3f} | "
            f"logZ: {diag_dict.get('logz_mean', 0):.2f} | "
            f"max|logit|: {diag_dict.get('max_abs_logit', 0):.1f} | "
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
                # One micro-step's grads (#82's original reading), and the window mean
                # the optimizer actually applies (#191). Only the second can say f16
                # underflow reached the weights.
                "zero_frac_dense_max": f"{zero_frac_dense_max:.6f}" if zero_frac_dense_max is not None else "",
                "applied_zero_frac_dense_max": (f"{applied_zero_frac_dense_max:.6f}"
                                                if applied_zero_frac_dense_max is not None else ""),
                # Norm of the window mean the optimizer clips (#180), comparable to CLIP_NORM.
                "applied_grad_norm": f"{applied_grad_norm:.4f}" if applied_grad_norm is not None else "",
                "avg_forget_cost": _fmt(diag_dict, "forget_cost", 4),
                "diversity_loss": _fmt(diag_dict, "diversity_loss", 6),
                "temporal_drift": _fmt(diag_dict, "temporal_drift", 6),
                "forget_density": _fmt(diag_dict, "forget_density", 6),
                "tau": _fmt(diag_dict, "tau", 6),
                "out_entropy": _fmt(diag_dict, "out_entropy", 4),
                "logz_mean": _fmt(diag_dict, "logz_mean", 4),
                "max_abs_logit": _fmt(diag_dict, "max_abs_logit", 2),
                "depth_avg": f"{depth_avg:.4f}" if depth_avg is not None else "",
                "val_ce": f"{val_ce:.4f}" if val_ce is not None else "",
                # Listed in fields and diag_keys since #252 but never written: the
                # column existed and was always empty, so the f16-margin invariant
                # reading it could not fire.
                "act_max": _fmt(diag_dict, "act_max", 1),
                "wall_clock": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "mix": mix or "",
                "arena_peak_mib": _arena_peak_mib(),
            }
            writer.writerow(row)
