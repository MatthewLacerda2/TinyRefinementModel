import csv
import datetime
import math
import posixpath
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
# written. Every reader in instruments/ reads columns by name, never by position. An
# absent value (an optional argument not passed) is an empty cell, never a zero that
# looks like a measurement. A file that already has a header is appended to under
# that header, and a resume first widens it with any current column it lacks
# (`_truncate_replayed_rows`), so a run resumed across a schema change stays aligned
# and keeps its retired columns (#292).
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
    # 1 when that norm exceeded CLIP_NORM, so the clip, not the LR, sized the step (#358).
    Column("clip_active", None),
    # The f16 loss scale at the row, and the micro-steps skipped as non-finite since
    # launch (#368): the scaler's state as a column instead of a grep of train.log.
    Column("loss_scale", None),
    Column("skipped_micro_steps", None),
    Column("out_entropy", 4, diag="out_entropy"),
    Column("logz_mean", 4, diag="logz_mean"),
    Column("max_abs_logit", 2, diag="max_abs_logit"),
    # Peak |activation| through the stack. The number that decides whether this
    # model can be served in f16 at all, and the one nothing watched during the 4B
    # run: it finished at 65,120 against a 65,504 ceiling and that was found two
    # weeks later, by hand (#235). It then sat in the header from #252 on and was
    # never written, because the header and the row were two separate lists.
    Column("act_max", 1, diag="act_max"),
    # Peak |output| of any branch (attention or MLP) in the compute dtype, before the
    # add (#536). With the stream f32 (#357), this is the f16 margin the alarm reads.
    Column("branch_max", 1, diag="branch_max"),
    Column("val_ce", 4),
    # The opt step the probe measured val_ce at (#351). The value is written on the
    # next logged row, up to LOG_REAL_STEPS - 1 steps later, so the row's own `step`
    # is not when it was measured.
    Column("val_step", None),
    # Held-out CE on the other corpora (#363), `source=ce;...`, measured at val_step
    # every VAL_BY_SOURCE_EVERY_OPT_STEPS. val_ce stays fineweb's.
    Column("val_by_source", None),
    # Context a row cannot be read without, and that cannot be backfilled (#186):
    # when it was written — the only clock the run keeps against its progress —
    # and the data mixture its CE was measured on, which the curriculum moves
    # every step.
    Column("wall_clock", None),
    Column("mix", None),
    # Micro-step gradient norm per data source over the window, as
    # `source=mean/max/guard-clipped/micro-steps` (#364): which data makes the tail.
    Column("grad_by_source", None),
    # The allocator's own high-water mark so far (#168): exact, not a poll. Every
    # run records how close it came to its limit, and the fit gate reads it from
    # the probe run's first row.
    Column("arena_peak_mib", None),
    # The allocator's ceiling, the other half of headroom (#346). It moves with
    # XLA_PYTHON_CLIENT_MEM_FRACTION and the allocator, so a run records its own
    # rather than every reader assuming the RTX 2060's 4,883 MiB.
    Column("arena_limit_mib", None),
)


# The diagnostics the console line shows: (diagnostic, label, decimals).
CONSOLE_DIAGNOSTICS = (
    ("out_entropy", "H", 3),
    ("logz_mean", "logZ", 2),
    ("max_abs_logit", "max|logit|", 1),
)


def _cell(value, places):
    if value is None:
        return ""
    return value if places is None else f"{value:.{places}f}"


def _allocator_mib(key):
    """One allocator statistic in MiB, or empty where the allocator keeps none
    (CPU, the platform allocator)."""
    try:
        stats = jax.local_devices()[0].memory_stats() or {}
    except (AttributeError, RuntimeError):
        stats = {}
    value = stats.get(key)
    return f"{value / 2**20:.0f}" if value else ""


def _arena_peak_mib():
    return _allocator_mib("peak_bytes_in_use")


def _arena_limit_mib():
    return _allocator_mib("bytes_limit")


# The per-block activation readings (#392), beside metrics.csv: long format, one row
# per (step, state), because the number of states follows PLAIN_LAYERS and a fixed
# metrics schema cannot. State 0 is the embedding, state k the stream after block k.
BLOCKS_FILENAME = "blocks.csv"
# attn_out_max / mlp_out_max (#536): block k's two branch outputs before the add;
# empty on row 0, the embedding, which has no branches.
BLOCKS_FIELDS = ("step", "block", "act_max", "act_rms", "attn_out_max", "mlp_out_max")


def blocks_file_for(history_file):
    """blocks.csv beside metrics.csv, for a local path or an fsspec URL alike."""
    return posixpath.join(posixpath.dirname(history_file), BLOCKS_FILENAME)


class MetricsLogger:
    def __init__(self, history_file, start_opt_step=None):
        self.history_file = history_file
        # Telemetry this logger knows how to write; a key the model did not report
        # leaves its column empty rather than a zero that looks like a measurement.
        self.diag_keys = [c.diag for c in COLUMNS if c.diag]
        self.blocks_file = blocks_file_for(history_file)
        self.fields = [c.name for c in COLUMNS]
        # Warn once per metric name when a non-finite value shows up, so a broken
        # diagnostic can't silently fill the CSV with NaN.
        self._warned_nonfinite = set()
        if start_opt_step is not None:
            self._truncate_replayed_rows(start_opt_step)
            self._truncate_replayed_blocks(start_opt_step)

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
            # The file keeps every column it has (a retired one included: its old rows
            # are history) and gains the current ones it lacks, at the end; `log`
            # then appends under this header.
            old_fields = list(old_fields or [])
            fields = old_fields + [c for c in self.fields if c not in old_fields]
            if len(kept) == len(rows) and fields == old_fields:
                return
            print(f"✂️ Trimming {len(rows) - len(kept)} replayed metric rows (step >= {start_opt_step}) from {self.history_file}")
            with fsspec.open(self.history_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', restval="")
                writer.writeheader()
                writer.writerows(kept)
        except (OSError, ValueError, KeyError) as e:
            print(f"⚠️ Could not trim replayed rows from {self.history_file}: {e}")

    def _truncate_replayed_blocks(self, start_opt_step):
        """The same resume trim for blocks.csv: rows at or after the restored step go."""
        try:
            fs, path = fsspec.core.url_to_fs(self.blocks_file)
            if not fs.exists(path) or fs.size(path) == 0:
                return
            with fsspec.open(self.blocks_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                old_fields = list(reader.fieldnames or [])
            kept = [r for r in rows if r.get("step") and int(r["step"]) < start_opt_step]
            # Widened like metrics.csv: a file from before a column existed gains it.
            fields = old_fields + [c for c in BLOCKS_FIELDS if c not in old_fields]
            if len(kept) == len(rows) and fields == old_fields:
                return
            with fsspec.open(self.blocks_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
                writer.writeheader()
                writer.writerows(kept)
        except (OSError, ValueError, KeyError) as e:
            print(f"⚠️ Could not trim replayed rows from {self.blocks_file}: {e}")

    def _log_blocks(self, step, diag):
        """One blocks.csv row per state, when the model reports per-block readings
        (a stub output in a test may not): absent, never zero."""
        if "act_max_blocks" not in diag:
            return
        maxes = [float(v) for v in jnp.ravel(diag["act_max_blocks"])]
        rmses = [float(v) for v in jnp.ravel(diag["act_rms_blocks"])]
        # Row k > 0 is the stream after block k, so it carries block k's two branches;
        # row 0 (the embedding) has none.
        branches = [("", "")] * len(maxes)
        if "branch_max_blocks" in diag:
            branches[1:] = [(f"{attn:.2f}", f"{mlp:.2f}")
                            for attn, mlp in jnp.reshape(diag["branch_max_blocks"], (-1, 2)).tolist()]
        fs, path = fsspec.core.url_to_fs(self.blocks_file)
        fresh = not fs.exists(path) or fs.size(path) == 0
        with fsspec.open(self.blocks_file, "a", newline="") as f:
            writer = csv.writer(f)
            if fresh:
                writer.writerow(BLOCKS_FIELDS)
            for block, (peak, rms, (attn, mlp)) in enumerate(zip(maxes, rmses, branches)):
                writer.writerow([int(step), block, f"{peak:.2f}", f"{rms:.4f}", attn, mlp])

    def extract_diags(self, diag, jnp_mean_fn):
        """Reduces the diagnostics this model reported to plain floats. Keys the
        model did not report are absent, not zero."""
        return {k: float(jnp_mean_fn(diag[k])) for k in self.diag_keys if k in diag}

    def log(self, step, ce, loss, out, compute_time,
            grad_norm_avg=None, seg1_ce=None, val_ce=None,
            zero_frac_dense_max=None, applied_zero_frac_dense_max=None, applied_grad_norm=None,
            clip_active=None, val_step=None, val_by_source=None, mix=None, grad_by_source=None,
            loss_scale=None, skipped_micro_steps=None):
        """Logs training metrics to console and CSV based on the routing specification."""
        diag_dict = self.extract_diags(out.diag, jnp.mean)

        for name, value in {**diag_dict, "ce": ce, "loss": loss}.items():
            if not math.isfinite(value) and name not in self._warned_nonfinite:
                self._warned_nonfinite.add(name)
                print(f"⚠️ Non-finite metric '{name}' ({value}) at step {step} — check the diagnostics pipeline.")

        # Console: only the diagnostics the model reported.
        reported = "".join(f" | {label}: {diag_dict[key]:.{places}f}"
                           for key, label, places in CONSOLE_DIAGNOSTICS if key in diag_dict)
        print(
            f"Step {step:04d} | CE: {ce:.4f} (seg1: {seg1_ce:.4f})\n"
            f"      Loss: {loss:.4f}{reported} | Compute: {compute_time:.3f}s"
        )

        # A file with content keeps its own header: rows go under the columns it
        # already has, and a column it lacks is dropped rather than shifting the rest.
        header = None
        try:
            fs, path = fsspec.core.url_to_fs(self.history_file)
            if fs.exists(path) and fs.size(path) > 0:
                with fsspec.open(self.history_file, "r", newline="") as f:
                    header = next(csv.reader(f), None)
        except OSError as e:
            print(f"⚠️ Could not read {self.history_file} ({e}); assuming empty.")

        with fsspec.open(self.history_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header or self.fields,
                                    extrasaction='ignore', restval="")
            if header is None:
                writer.writeheader()

            args = {
                "step": int(step), "ce": ce, "loss": loss, "seg1_ce": seg1_ce,
                "grad_norm_avg": grad_norm_avg, "zero_frac_dense_max": zero_frac_dense_max,
                "applied_zero_frac_dense_max": applied_zero_frac_dense_max,
                "applied_grad_norm": applied_grad_norm, "clip_active": clip_active,
                "loss_scale": loss_scale, "skipped_micro_steps": skipped_micro_steps,
                "val_ce": val_ce, "val_step": val_step,
                "wall_clock": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "mix": mix or "",
                "grad_by_source": grad_by_source or "",
                "val_by_source": val_by_source or "",
                "arena_peak_mib": _arena_peak_mib(),
                "arena_limit_mib": _arena_limit_mib(),
            }
            row = {c.name: _cell(diag_dict.get(c.diag) if c.diag else args[c.name], c.places)
                   for c in COLUMNS}
            writer.writerow(row)
        self._log_blocks(step, out.diag)
