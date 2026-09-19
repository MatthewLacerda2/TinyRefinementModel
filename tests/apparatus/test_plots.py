"""The plotter draws what was measured, and nothing else.

Two properties are worth a test here, and they are the two the previous
plotter got wrong (#177):

  * **A panel with no data is omitted.** Both flavours of "no data": a blank
    column (the arch never measured it, #105) and a column of literal zeros
    (what runs written before the blank convention put there). Either one drawn
    flat reads as a measurement of zero.
  * **The model is never instantiated.** Building a whole network to count its
    parameters is slow, wrong-architecture-prone, and a hazard while the card
    is training. The subprocess test is the one that can actually prove it —
    it checks that no model module was ever imported.

Everything else here is survival: a run three steps old, a header-only CSV, a
torn final row. A live run gets plotted while it is being written to, so the
instrument meets all three routinely.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest

HEADER = ("step,ce,loss,seg1_ce,grad_norm_avg,zero_frac_dense_max,avg_forget_cost,"
          "diversity_loss,temporal_drift,forget_density,tau,out_entropy,logz_mean,"
          "max_abs_logit,depth_avg,val_ce,arena_peak_mib")


def row(step, **overrides):
    """One refiner-shaped CSV row: the reasoner-only columns blank, as the live
    trainer writes them."""
    cells = {
        "ce": 11.0 - step / 500, "loss": 22.0, "seg1_ce": 10.8,
        "grad_norm_avg": 12.0 + step / 1000, "zero_frac_dense_max": 0.004,
        "avg_forget_cost": "", "diversity_loss": "", "temporal_drift": "",
        "forget_density": "", "tau": "", "out_entropy": 9.0 - step / 800,
        "logz_mean": 11.3, "max_abs_logit": 4.3 + step / 200,
        "depth_avg": 4.5, "val_ce": "",
        # The allocator's high-water mark, as every run has logged it since #168.
        "arena_peak_mib": 4400 + step // 10,
    }
    cells.update(overrides)
    return ",".join([str(step)] + [str(cells[name]) for name in HEADER.split(",")[1:]])


def write_csv(tmp_path, rows, name="run_20260101_000000"):
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.csv").write_text("\n".join([HEADER, *rows]) + "\n")
    return run_dir / "metrics.csv"


def a_run(tmp_path, steps=400, **overrides):
    rows = [row(step, **overrides) for step in range(5, steps + 1, 5)]
    return write_csv(tmp_path, rows)


def build(csv_path, out):
    from instruments import plots

    return {figure["path"].split("/")[-1]: figure for figure in plots.build(str(csv_path), out)}


def panels_of(figures, filename):
    return figures[filename]["panels"] if filename in figures else []


# ── the rule: absent data is omitted, and the reason is said out loud ────────

def test_reasoner_only_columns_never_get_a_panel(tmp_path):
    """The refiner has no forget gate, no slots and no drift (#105). Those
    columns are blank in every refiner run and must not reach a figure."""
    figures = build(a_run(tmp_path), tmp_path)

    drawn = [name for figure in figures.values() for name in figure["panels"]]
    for absent in ("temporal_drift", "avg_forget_cost", "diversity_loss", "forget_density",
                   "tau", "seg1_ce"):
        assert absent not in drawn


def test_blank_column_omits_its_chart_and_says_why(tmp_path, capsys):
    figures = build(a_run(tmp_path, grad_norm_avg=""), tmp_path)

    assert "grad_norm.png" not in figures
    assert "grad_norm: omitted — not measured by this architecture" in capsys.readouterr().out
    assert "logits.png" in figures, "the other diagnostics still have data"


def test_literal_zero_column_is_omitted_too(tmp_path, capsys):
    """The case `has()` alone cannot catch: a pre-#105 run wrote 0.0000 into the
    columns it did not measure, so the column is present and still not data."""
    figures = build(a_run(tmp_path, depth_avg="0.0000"), tmp_path)

    assert "depth.png" not in figures
    # Told, not silently dropped: absence and a flat zero must stay tellable apart.
    assert "depth: omitted — constant 0 throughout" in capsys.readouterr().out


def test_a_real_flat_signal_is_still_drawn(tmp_path):
    """Constant-at-zero is the tombstone; constant at some other value is a
    measurement (a sampler pinned to depth 4, say), and it gets its panel."""
    figures = build(a_run(tmp_path, depth_avg="4.0000"), tmp_path)

    assert "depth.png" in figures


def test_every_chart_is_its_own_image_for_a_full_log(tmp_path):
    """One chart per image (the board hangs each as its own widget), and only
    the charts worth a glance: no VRAM, zero-gradient, LR or progress panels."""
    figures = build(a_run(tmp_path), tmp_path)

    assert sorted(figures) == ["depth.png", "grad_norm.png", "logits.png", "training_curve.png"]
    assert all(len(figure["panels"]) <= 2 for figure in figures.values())


def test_val_ce_is_drawn_only_where_it_exists(tmp_path):
    """val_ce is written every 64 opt steps, so most rows are blank — it is a
    series of its own, never interpolated onto every row."""
    from instruments import plots
    from instruments.runlog import load

    rows = [row(step, val_ce=(6.0 if step % 320 == 0 else "")) for step in range(5, 1001, 5)]
    csv_path = write_csv(tmp_path, rows)
    figures = build(csv_path, tmp_path)
    assert "val_ce" in figures["training_curve.png"]["panels"]

    tokens, values = plots.series(load(str(csv_path)), "val_ce")
    assert len(values) == 3, "one point per row that actually carried a value"

    no_val = build(a_run(tmp_path / "b"), tmp_path / "b")
    assert "val_ce" not in no_val["training_curve.png"]["panels"]


# ── survival: the shapes a live CSV actually takes ───────────────────────────

def test_a_run_three_steps_old(tmp_path):
    figures = build(a_run(tmp_path, steps=15), tmp_path)

    assert "training_curve.png" in figures
    assert pathlib.Path(figures["training_curve.png"]["path"]).stat().st_size > 0


def test_header_only_csv_writes_nothing(tmp_path):
    figures = build(write_csv(tmp_path, []), tmp_path)

    assert figures == {}
    assert not list(tmp_path.glob("*.png"))


def test_torn_final_row(tmp_path):
    """The trainer can be mid-write when the plotter reads. A half-line is
    dropped; every complete row before it still gets drawn."""
    rows = [row(step) for step in range(5, 401, 5)]
    csv_path = write_csv(tmp_path, rows)
    with open(csv_path, "a") as handle:
        handle.write("405,3.9,7.8,4.1,11.\n")

    figures = build(csv_path, tmp_path)
    assert panels_of(figures, "training_curve.png")


def test_non_finite_values_do_not_break_the_curve(tmp_path):
    """A diverged step logs nan/inf. The loader keeps them (they mean something
    went wrong); the plotter must not hand them to a log axis."""
    rows = [row(step) for step in range(5, 401, 5)] + [row(405, ce="nan"), row(410, ce="inf")]
    figures = build(write_csv(tmp_path, rows), tmp_path)

    assert "training_curve.png" in figures


def test_throughput_needs_the_supervisor_log(tmp_path):
    """metrics.csv has no timestamps at all, so with no heartbeats to sample
    there is no honest throughput figure — and none is written."""
    figures = build(a_run(tmp_path), tmp_path)
    assert "throughput.png" not in figures

    csv_path = a_run(tmp_path / "with_log", steps=600)
    (csv_path.parent.parent / f"{csv_path.parent.name}.supervisor.log").write_text(
        "2026-01-01 00:00:00 RUNNING: step 200/30518 (ce=7.0)\n"
        "2026-01-01 01:00:00 RUNNING: step 400/30518 (ce=6.5)\n"
        "2026-01-01 02:00:00 RUNNING: step 600/30518 (ce=6.1)\n")

    figures = build(csv_path, tmp_path / "with_log")
    assert panels_of(figures, "throughput.png") == ["throughput"]


def test_missing_run_is_an_error_not_an_empty_figure(tmp_path):
    from instruments import plots

    with pytest.raises(FileNotFoundError):
        plots.build(str(tmp_path / "nope" / "metrics.csv"), tmp_path)


# ── the constraint: no model, ever ───────────────────────────────────────────

def test_the_plotter_never_imports_a_model(tmp_path, repo_root):
    """The defect this rewrite removes: the old plotter built a whole
    UniversalReasoner to count parameters — the wrong architecture, and a
    hazard on a card that is training. Run it for real in a clean interpreter
    and check no model module was ever imported."""
    csv_path = a_run(tmp_path)
    probe = (
        "import sys, runpy;"
        f"sys.argv = ['plots', '--log', {str(csv_path)!r}, '--out', {str(tmp_path)!r}];"
        "runpy.run_module('instruments.plots', run_name='__main__');"
        "print('MODELS:', [m for m in sys.modules if m.startswith('trm.model')])"
    )
    env = dict(os.environ, PYTHONPATH=str(repo_root), JAX_PLATFORMS="cpu", FORCE_F32_COMPUTE="1")
    proc = subprocess.run([sys.executable, "-c", probe], cwd=repo_root, env=env,
                          capture_output=True, text=True, timeout=300)

    assert proc.returncode == 0, proc.stderr
    assert "MODELS: []" in proc.stdout, "the plotter pulled in a model module"
    assert (tmp_path / "training_curve.png").exists()


def test_the_grad_norm_panel_reads_against_the_clip_when_the_run_logged_it(tmp_path, capsys):
    """#180: with applied_grad_norm the panel draws the clip line and says how often it bit."""
    from instruments import plots, runlog
    header = HEADER + ",applied_grad_norm"
    rows = [row(step) + f",{0.5 if step % 10 else 1.5}" for step in range(5, 401, 5)]
    run_dir = tmp_path / "run_20990101_000000"
    run_dir.mkdir()
    (run_dir / "metrics.csv").write_text("\n".join([header, *rows]) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    fig, ax = matplotlib.pyplot.subplots()
    log = runlog.load(str(run_dir / "metrics.csv"))
    plots._panel_grad_norm(ax, log, plots.RunConfig.of(log))
    title = ax.get_title(loc="left")
    assert "clip sees" in title and "50%" in title
    matplotlib.pyplot.close(fig)


# ── the run's own config decides the figure, not this process's (#305) ───────

def with_metadata(csv_path, **parameters):
    """Give a run the `parameters` block its tracker would have written. A None
    value means the run predates that key and never recorded it."""
    recorded = {key: value for key, value in parameters.items() if value is not None}
    (csv_path.parent / "run_metadata.json").write_text(json.dumps(
        {"run_id": csv_path.parent.name, "parameters": recorded}))
    return csv_path


def a_short_arm(tmp_path, name="run_026_muon_m100_s1", **parameters):
    """A 512-step ablation arm: the shape that crashed the plotter. Its LR
    schedule completes inside the run, so its warmup is 100, not the 1000 this
    process defaults to."""
    rows = [row(step, val_ce=(6.0 if step % 80 == 0 else "")) for step in range(5, 513, 5)]
    recorded = {"MODEL_ARCH": "plain", "LATENT_DIM": 960, "PLAIN_LAYERS": 9,
                "MAX_SEQ_LEN": 512, "BATCH_SIZE": 1, "ACCUMULATION_STEPS": 128,
                "TRAIN_TOKEN_BUDGET": 512 * 131072, "DECAY_STEPS": 512,
                "WARMUP_STEPS": 100, "VAL_EVERY_OPT_STEPS": 16,
                "TRM_OPTIMIZER": "muon", "MUON_LR_MULT": 100.0}
    recorded.update(parameters)
    return with_metadata(write_csv(tmp_path, rows, name=name), **recorded)


def test_a_short_run_renders_and_names_its_own_schedule(tmp_path):
    """The #305 crash was the plotter rebuilding the LR schedule with its own
    1000-step warmup against a 512-step horizon. The schedule is a sentence in
    the note now, read from the run, so the short arm renders and says 100."""
    from instruments import plots
    from instruments.runlog import load

    figures = build(a_short_arm(tmp_path), tmp_path)
    assert figures["training_curve.png"]["panels"] == ["ce", "val_ce"]

    cfg = plots.RunConfig.of(load(str(a_short_arm(tmp_path / "again"))))
    assert plots.schedule_line(cfg) == "LR: 100-step warmup to 0.0006, cosine to 512 optimizer steps."


def test_a_plain_run_omits_the_depth_panel_and_says_why(tmp_path, capsys):
    """PlainTransformer ignores the depth argument, so depth_avg is the sampler's
    dice roll — logged, and not a measurement of this model."""
    figures = build(a_short_arm(tmp_path), tmp_path)

    assert "depth.png" not in figures
    assert "ignores the depth argument" in capsys.readouterr().out


def test_a_refiner_run_still_draws_the_depth_panel(tmp_path):
    """The looping arches do have a depth, and the realised mean of the draw is
    still worth a check there (the 4B champion is one of these)."""
    figures = build(a_short_arm(tmp_path, MODEL_ARCH="refiner", MAX_STEPS_LIMIT=8), tmp_path)

    assert "depth.png" in figures


def test_the_subtitle_identifies_a_plain_run_by_arch_layers_and_optimizer(tmp_path):
    """`depth ≤8` said nothing about a plain run. What separates two runs of this
    stack is the layer count and the optimizer — the #26 pair differs in nothing
    else."""
    from instruments import plots
    from instruments.runlog import load

    described = plots.describe(plots.RunConfig.of(load(str(a_short_arm(tmp_path)))))
    assert described == "plain · dim 960 · 9 layers · muon (LR ×100 on matrices)"
    assert "depth" not in described

    refiner = plots.RunConfig({"parameters": {"MODEL_ARCH": "refiner", "LATENT_DIM": 960,
                                              "MAX_STEPS_LIMIT": 8, "TRM_OPTIMIZER": "adamw"}})
    assert plots.describe(refiner) == "refiner · dim 960 · depth ≤8 · adamw"


def test_a_fact_the_run_did_not_record_is_named_as_missing(tmp_path):
    """The failure this rule prevents: every run on disk predates #305, and
    labelling the muon arm `adamw` from this process's default is exactly the
    confident-wrong picture the issue is about. PLAIN_LAYERS is the same story —
    it was 8 before 2026-09-13."""
    from instruments import plots

    older = plots.RunConfig({"parameters": {"MODEL_ARCH": "plain", "LATENT_DIM": 960}})
    assert plots.describe(older) == "plain · dim 960 · layer count not recorded · optimizer not recorded"


def test_the_val_cadence_comes_from_the_run(tmp_path):
    """The note used to read "every 64 optimizer steps" whatever the run did;
    every arm of the #26 pair probes every 16."""
    from instruments import plots
    from instruments.runlog import load

    recorded = load(str(a_short_arm(tmp_path)))
    assert plots.val_cadence(recorded, plots.RunConfig.of(recorded)) == (16, "recorded")

    # A run that never recorded the knob: read the spacing off its val rows. The
    # mean gap, not the median — a probe every 16 steps logged every 5 writes its
    # value on the next logged row, so the gaps alternate and only the mean is 16.
    older = load(str(a_short_arm(tmp_path / "older", VAL_EVERY_OPT_STEPS=None)))
    assert plots.val_cadence(older, plots.RunConfig.of(older)) == (80, "observed")

    uneven = load(str(write_csv(tmp_path / "uneven",
                                [row(step, val_ce=(6.0 if step in (20, 35, 50, 65, 80, 100) else ""))
                                 for step in range(5, 201, 5)], name="run_20990101_000000")))
    assert plots.val_cadence(uneven, plots.RunConfig.of(uneven)) == (16, "observed")


def test_a_run_written_before_these_parameters_still_renders(tmp_path):
    """Every finished run on disk predates #305. They keep their figures: the
    plotter falls back to this process's config for what they never recorded,
    and only drops what that fallback cannot honestly reconstruct."""
    rows = [row(step) for step in range(5, 1001, 5)]
    csv_path = with_metadata(write_csv(tmp_path, rows, name="run_20260719_020802"),
                             MODEL_ARCH="refiner", LATENT_DIM=960, MAX_STEPS_LIMIT=8,
                             TRAIN_TOKEN_BUDGET=4_000_000_000, DECAY_STEPS=30518)
    figures = build(csv_path, tmp_path)

    assert figures["training_curve.png"]["panels"] == ["ce"]
    assert "depth.png" in figures


def test_a_run_with_no_metadata_at_all_still_renders(tmp_path):
    """A hand-assembled run directory, or one from before the tracker."""
    figures = build(a_run(tmp_path), tmp_path)

    assert "training_curve.png" in figures and "grad_norm.png" in figures


def test_the_margins_are_reported_in_words_not_drawn(tmp_path, capsys):
    """Arena peak and the f16 zero-gradient fraction are margins, not curves: on
    a healthy run one is flat and the other is a 1e-4 blip that an autoscaled
    axis draws as a crisis (#379). Each is one line, and neither warns here."""
    from instruments import plots
    from instruments.runlog import load

    warnings = plots.margin_report(load(str(a_short_arm(tmp_path, name="run_026_adamw_s2"))))
    out = capsys.readouterr().out
    assert warnings == []
    assert "VRAM: arena peak" in out and "f16 zero-gradient fraction" in out
    assert plots.ARENA_LIMIT_MIB == 4883.0, "the measured cuda_async bytes_limit on this card"


def test_a_margin_that_crosses_its_line_is_flagged(tmp_path):
    """The two failures these margins exist for: a run near the arena ceiling, and
    gradients underflowing in f16 (the dead base runs sat at 0.50-0.75)."""
    from instruments import plots
    from instruments.runlog import load

    near = load(str(a_run(tmp_path / "near", arena_peak_mib=4800)))
    assert any("headroom" in w for w in plots.margin_report(near))
    underflow = load(str(a_run(tmp_path / "under", zero_frac_dense_max=0.6)))
    assert any("underflowing" in w for w in plots.margin_report(underflow))


def test_no_vram_line_where_the_allocator_kept_no_statistics(tmp_path, capsys):
    """A CPU run logs an empty cell — the margin is not reported, not reported as zero."""
    from instruments import plots
    from instruments.runlog import load

    plots.margin_report(load(str(a_run(tmp_path, arena_peak_mib=""))))
    assert "VRAM" not in capsys.readouterr().out


def test_the_arena_limit_is_the_runs_own_when_logged_and_flagged_when_assumed():
    """#319: the limit was one frozen card's number for every run. A run that logs its
    own is drawn against it; one that does not gets the RTX 2060 number, marked assumed."""
    from instruments import plots
    from instruments.runlog import RunLog

    logged = RunLog("r", [{"step": 5, "arena_limit_mib": 5100.0}, {"step": 10, "arena_limit_mib": 5120.0}], {})
    assert plots.arena_limit_mib(logged) == (5120.0, True)
    assert plots.arena_limit_mib(RunLog("r", [{"step": 5}], {})) == (plots.ARENA_LIMIT_MIB, False)


def test_a_blank_column_the_recorded_arch_does_log_is_not_blamed_on_the_arch(tmp_path):
    """#332 review: on a plain run, a missing column plain does log was reported as
    'not measured by this architecture'. The reason comes from runlog, shared with report."""
    from instruments import plots
    from instruments.runlog import RunLog

    plain = RunLog("r", [{"step": 5, "grad_norm_avg": None}], {"parameters": {"MODEL_ARCH": "plain"}})
    assert plots.why_omitted(plain, ["grad_norm_avg"]) == "not logged by this run"
    reasoner_only = plots.why_omitted(plain, ["tau"])
    assert reasoner_only == "not measured by this architecture"


def test_a_run_with_blocks_csv_gets_the_two_heatmaps(tmp_path):
    """#392: state on y, tokens on x, max and RMS as two images; parsed by runlog."""
    from instruments.runlog import load

    csv_path = a_run(tmp_path)
    lines = ["step,block,act_max,act_rms"] + [
        f"{step},{block},{10.0 * (block + 1) + step / 100:.2f},{0.1 * (block + 1):.4f}"
        for step in range(5, 401, 5) for block in range(3)]
    (csv_path.parent / "blocks.csv").write_text("\n".join(lines) + "\n")

    steps, maxes, rmses = load(str(csv_path)).blocks()
    assert maxes.shape == rmses.shape == (80, 3) and steps[0] == 5
    figures = build(csv_path, tmp_path)
    assert "blocks_act_max.png" in figures and "blocks_act_rms.png" in figures


def test_the_other_corpora_are_drawn_on_the_ce_chart(tmp_path, monkeypatch):
    """#363: the per-corpus held-out CE joins the CE chart as its own lines, and a
    run that logged none draws none."""
    import matplotlib.axes
    from instruments import plots
    from instruments.runlog import load

    run_dir = tmp_path / "run_20260101_000000"
    run_dir.mkdir()
    lines = ["step,ce,val_ce,val_step,val_by_source"] + [
        f"{s},{6 - s / 1000},{6.1 - s / 1000},{s},"
        + ("codeparrot=2.5;finemath=3.0" if s % 100 == 0 else "") for s in range(5, 401, 5)]
    (run_dir / "metrics.csv").write_text("\n".join(lines) + "\n")

    labels = []
    original = matplotlib.axes.Axes.legend
    def spy(ax, *args, **kwargs):
        labels.extend(ax.get_legend_handles_labels()[1])
        return original(ax, *args, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, "legend", spy)

    (tmp_path / "out").mkdir()
    plots.training_curve(load(str(run_dir)), str(tmp_path / "out"))
    assert "held-out codeparrot" in labels and "held-out finemath" in labels
