import csv
import datetime
import math
from typing import NamedTuple
import fsspec
# jax at module level: the module already needs jax.numpy, so a lazy import in
# _arena_peak_mib bought nothing.
import jax
import jax.numpy as jnp


class Column(NamedTuple):
    name: str
    places: int | None       # decimals written; None writes the value as it is
    diag: str | None = None  # the model diagnostic it holds; None: the log() argument `name`


# The CSV schema, declared once: every column in order, what it holds, and how it is
# written. Old runs and every reader in instruments/ depend on the column set, the
# order and the formatting. An absent value — an optional argument not passed, a
# diagnostic this architecture does not measure (#105) — is an empty cell, never a
# zero that looks like a measurement.
COLUMNS = (
    Column("step", None),
    Column("ce", 4),
    Column("loss", 4),
    Column("seg1_ce", 4),
    Column("grad_norm_avg", 4),
    # One micro-step's grads (#82's original reading), and the window mean the
    # optimizer actually applies (#191). Only the second can say f16 underflow
    # reached the weights.
    Column("zero_frac_dense_max", 6),
    Column("applied_zero_frac_dense_max", 6),
    # Norm of the window mean the optimizer clips (#180), comparable to CLIP_NORM.
    Column("applied_grad_norm", 4),
    Column("avg_forget_cost", 4, diag="forget_cost"),
    Column("diversity_loss", 6, diag="diversity_loss"),
    Column("temporal_drift", 6, diag="temporal_drift"),
    Column("forget_density", 6, diag="forget_density"),
    Column("tau", 6, diag="tau"),
    Column("out_entropy", 4, diag="out_entropy"),
    Column("logz_mean", 4, diag="logz_mean"),
    Column("max_abs_logit", 2, diag="max_abs_logit"),
    # Peak |activation| through the stack. The number that decides whether this
    # model can be served in f16 at all, and the one nothing watched during the 4B
    # run: it finished at 65,120 against a 65,504 ceiling and that was found two
    # weeks later, by hand (#235). It then sat in the header from #252 on and was
    # never written, because the header and the row were two separate lists.
    Column("act_max", 1, diag="act_max"),
    Column("depth_avg", 4),
    Column("val_ce", 4),
    # Context a row cannot be read without, and that cannot be backfilled (#186):
    # when it was written — the only clock the run keeps against its progress —
    # and the data mixture its CE was measured on, which the curriculum moves
    # every step.
    Column("wall_clock", None),
    Column("mix", None),
    # The allocator's own high-water mark so far (#168): exact, not a poll. Every
    # run records how close it came to its limit, and the fit gate reads it from
    # the probe run's first row.
    Column("arena_peak_mib", None),
)


# The diagnostics the console line shows when the model reports them:
# (diagnostic, label, decimals).
CONSOLE_DIAGNOSTICS = (
    ("tau", "Tau", 4),
    ("temporal_drift", "Drift", 6),
    ("out_entropy", "H", 3),
    ("logz_mean", "logZ", 2),
    ("max_abs_logit", "max|logit|", 1),
)


def _cell(value, places):
    if value is None:
        return ""
    return value if places is None else f"{value:.{places}f}"


def _arena_peak_mib():
    """peak_bytes_in_use in MiB, or empty where the allocator keeps no statistics
    (CPU, the platform allocator)."""
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
        self.diag_keys = [c.diag for c in COLUMNS if c.diag]
        self.fields = [c.name for c in COLUMNS]
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

        # Console: only the diagnostics this model reported. A 0 for one it has none
        # of (Tau, Drift on the plain stack) reads as a measurement (#317).
        reported = "".join(f" | {label}: {diag_dict[key]:.{places}f}"
                           for key, label, places in CONSOLE_DIAGNOSTICS if key in diag_dict)
        print(
            f"Step {step:04d} | CE: {ce:.4f} (seg1: {seg1_ce:.4f}) | Depth: {depth_avg:.2f}\n"
            f"      Loss: {loss:.4f}{reported} | Compute: {compute_time:.3f}s"
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

            args = {
                "step": int(step), "ce": ce, "loss": loss, "seg1_ce": seg1_ce,
                "grad_norm_avg": grad_norm_avg, "zero_frac_dense_max": zero_frac_dense_max,
                "applied_zero_frac_dense_max": applied_zero_frac_dense_max,
                "applied_grad_norm": applied_grad_norm, "depth_avg": depth_avg, "val_ce": val_ce,
                "wall_clock": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "mix": mix or "",
                "arena_peak_mib": _arena_peak_mib(),
            }
            row = {c.name: _cell(diag_dict.get(c.diag) if c.diag else args[c.name], c.places)
                   for c in COLUMNS}
            writer.writerow(row)
