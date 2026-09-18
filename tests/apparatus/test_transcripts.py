"""The transcript logbook has to stay comparable across checkpoints (#203).

Its whole value is that an entry written today can be read against one written
three weeks and two billion tokens later. That only holds if the things which
define a comparison — the prompt set, the depth ladder, the seed, the device, the
step arithmetic — are pinned rather than drifting quietly.

So these guard the properties that make an old transcript *interpretable*, plus
the one that stops the tool from killing the run it exists to observe.

Importable without JAX, and everything here is pure Python EXCEPT `TestARealRunOfMain`,
which drives `dump_transcripts.main()` and imports jax through trm.config and trm.infer.
The rest can run beside a live trainer without competing for memory; deselect that
class there (`-k "not TestARealRunOfMain"`). CI's pytest job runs all of it.
"""

import pytest

from instruments.dump_transcripts import (
    DEFAULT_DEPTHS,
    depths_for,
    PROMPTS,
    PROMPT_SET_VERSION,
    nearest_metric,
    opt_step_from_checkpoint,
    parse_depths,
    render_frontmatter,
    repetition_score,
    select_device,
    transcript_filename,
)


class TestRepetitionScore:
    """The number that turns 'it loops' into something you can plot."""

    def test_no_repetition_scores_zero(self):
        assert repetition_score(list(range(50))) == 0.0

    def test_a_hard_loop_approaches_one(self):
        """The Aug 15 failure mode: 'the ratio of the ratio of the ratio'."""
        assert repetition_score([1, 2, 3, 4] * 25) > 0.9

    def test_a_completion_shorter_than_the_window_is_not_an_error(self):
        assert repetition_score([1, 2]) == 0.0

    def test_partial_repetition_lands_between(self):
        fresh, looped = list(range(40)), [7, 7, 7, 7] * 10
        assert 0.0 < repetition_score(fresh + looped) < 1.0

    def test_it_is_deterministic(self):
        """A metric that moved between runs would be worse than no metric."""
        tokens = [3, 1, 4, 1, 5, 9, 2, 6, 3, 1, 4, 1, 5]
        assert repetition_score(tokens) == repetition_score(tokens)


class TestDepthLadder:
    def test_the_default_ladder_spans_the_trained_range(self):
        """1 is the no-refinement baseline, 8 is MAX_STEPS_LIMIT. Both ends matter:
        without 1 there is no floor to compare against, without 8 the 'does it
        overthink' question cannot be asked at all."""
        assert DEFAULT_DEPTHS == (1, 2, 4, 8)

    def test_a_looped_arch_runs_the_whole_ladder(self):
        assert depths_for("refiner") == DEFAULT_DEPTHS
        assert depths_for("reasoner", (4, 8)) == (4, 8)

    def test_an_arch_without_a_depth_dial_runs_once(self):
        """Plain ignores depth: four rungs were four identical completions at 4x the
        cost, and a repetition table of four identical rows (#317)."""
        assert depths_for("plain") == (1,)
        assert depths_for("plain", (4, 8)) == (4,)

    def test_depths_parse_in_order(self):
        assert parse_depths("1,2,4,8") == (1, 2, 4, 8)

    def test_whitespace_and_duplicates_are_tolerated(self):
        assert parse_depths(" 4 , 8 , 4 ") == (4, 8)

    @pytest.mark.parametrize("bad", ["0", "-1", "", " , "])
    def test_a_meaningless_ladder_is_refused(self, bad):
        """Depth 0 would run the block zero times and silently produce garbage
        rather than failing, which is the worst of both."""
        with pytest.raises(ValueError):
            parse_depths(bad)


class TestPromptSet:
    def test_the_set_is_frozen_at_eight(self):
        """Not a style rule. Every stored transcript was generated against this
        list, so an edit that is not accompanied by a version bump silently breaks
        comparability with every entry already on disk."""
        assert len(PROMPTS) == 8
        assert PROMPT_SET_VERSION == 1

    def test_the_loop_detector_is_present(self):
        """Enumeration is what provokes the failure mode the logbook watches; drop
        this prompt and the repetition column loses its most sensitive probe."""
        assert "the american flag is red, white and" in PROMPTS

    def test_code_and_math_are_probed(self):
        """~65% of the current curriculum is code and math. A prose-only prompt set
        would log a third of the model."""
        assert any(p.startswith("def ") for p in PROMPTS)
        assert any("+" in p for p in PROMPTS)


class TestStepArithmetic:
    def test_the_real_checkpoint_maps_to_its_real_opt_step(self):
        """Checkpoint 1458175 of the #157 run is opt step 11,392 — the trainer
        logged a validation there. Orbax names checkpoints by samples consumed and
        the trainer resumes at n+1, so the naive floor division is off by one and
        would mislabel every entry by a checkpoint interval."""
        assert opt_step_from_checkpoint(1458175, 128) == 11392

    def test_the_boundary_does_not_drift(self):
        # The last sample of opt step 11,008 (11,008 x 128 - 1 = 1,409,023): resuming at
        # n+1 lands exactly on the boundary, so it must not round down to 11,007.
        assert opt_step_from_checkpoint(11_008 * 128 - 1, 128) == 11_008


class TestNearestMetric:
    def test_it_takes_the_last_reading_at_or_before_the_step(self):
        """val_ce is written every 64 opt steps, so a checkpoint almost never lands
        on a row that has one."""
        steps, values = [64, 128, 192], [4.2, 4.0, 3.9]
        assert nearest_metric(steps, values, 150) == 4.0

    def test_an_exact_hit_is_used(self):
        assert nearest_metric([64, 128], [4.2, 4.0], 128) == 4.0

    def test_nothing_earlier_means_nothing(self):
        """Absent, never zero — a 0.0 CE in a model card would be read as a result."""
        assert nearest_metric([128], [4.0], 64) is None

    def test_an_empty_column_is_not_an_error(self):
        assert nearest_metric([], [], 11392) is None


class TestFrontmatter:
    def test_absent_fields_are_omitted_rather_than_zeroed(self):
        """A finished or foreign model has no metrics.csv beside it. Writing
        `val_ce: 0` there would be a fabricated result."""
        rendered = render_frontmatter({"step": 11392, "val_ce": None})
        assert "step: 11392" in rendered
        assert "val_ce" not in rendered

    def test_the_depth_ladder_renders_as_a_list(self):
        assert "depths: [1, 2, 4, 8]" in render_frontmatter({"depths": [1, 2, 4, 8]})

    def test_booleans_are_yaml_not_python(self):
        """`True` would not parse as a boolean in any YAML reader."""
        assert "model_commit_dirty: true" in render_frontmatter({"model_commit_dirty": True})

    def test_it_is_delimited_so_a_parser_can_find_it(self):
        rendered = render_frontmatter({"step": 1})
        assert rendered.startswith("---\n") and rendered.endswith("\n---")

    def test_the_two_depths_are_recorded_separately(self):
        """val_ce is measured at VAL_FIXED_DEPTH while completions span the ladder.
        A single `depth:` key would claim they describe the same configuration."""
        rendered = render_frontmatter({"depths": [1, 2, 4, 8], "val_ce_depth": 4})
        assert "val_ce_depth: 4" in rendered and "depths: [1, 2, 4, 8]" in rendered


class TestFilename:
    def test_the_device_is_in_the_name(self):
        """CPU runs f32 and GPU runs f16, so identical weights and seed still
        diverge. Two series that must never be silently mixed on one plot."""
        assert transcript_filename(11392, "cpu") != transcript_filename(11392, "gpu")
        assert "cpu" in transcript_filename(11392, "cpu")

    def test_steps_sort_lexically(self):
        """Zero-padded so `ls` orders the logbook chronologically."""
        assert transcript_filename(9000, "gpu") < transcript_filename(11392, "gpu")


class TestDeviceSafety:
    def test_cpu_pins_the_backend_before_jax_loads(self, monkeypatch):
        monkeypatch.delenv("JAX_PLATFORMS", raising=False)
        select_device("cpu")
        import os
        assert os.environ["JAX_PLATFORMS"] == "cpu"

    def test_the_gpu_is_refused_while_a_run_owns_it(self, monkeypatch):
        """The defect this replaces: the module used to take ~3GB of a 6GB card at
        import time, unconditionally. Running it beside the #157 base run — which
        holds 5.2GB — would have OOM'd twelve days of training."""
        monkeypatch.setattr("instruments.dump_transcripts.gpu_memory_used_mib", lambda: 5193)
        with pytest.raises(SystemExit, match="training run"):
            select_device("gpu")

    def test_force_overrides_a_busy_card(self, monkeypatch):
        monkeypatch.setattr("instruments.dump_transcripts.gpu_memory_used_mib", lambda: 5193)
        select_device("gpu", force=True)

    def test_an_idle_card_is_allowed(self, monkeypatch):
        monkeypatch.setattr("instruments.dump_transcripts.gpu_memory_used_mib", lambda: 4)
        select_device("gpu")

    def test_an_unreadable_card_does_not_block_the_tool(self, monkeypatch):
        """No nvidia-smi is a dev box, not a busy card."""
        monkeypatch.setattr("instruments.dump_transcripts.gpu_memory_used_mib", lambda: None)
        select_device("gpu")


class TestARealRunOfMain:
    """main() end to end, with only the model, the tokenizer and the generator stubbed.

    These replace source-string guards (#338 review): a check that `main()`'s text
    contains `PROMPTS[:args.prompts]` passes for code that never runs it. Here the
    order of events, the prompts generated, the frontmatter written and the contract
    line printed are all observed. Imports jax through trm.config and trm.infer, so
    unlike the rest of this file it runs in CI's pytest job, not beside a trainer.

    `tool_commit` must describe the code that generated the text (#214): it was once
    read after every completion, and a ~1h48m CPU entry stamped itself with commits
    that merged mid-run.
    """

    @staticmethod
    def _run(tmp_path, monkeypatch, capsys, *argv, run_dir=None):
        """Run main(). With `run_dir`, the checkpoint is discovered as that run's (so main
        reads its metrics.csv and run_metadata.json); otherwise --checkpoint-path is passed."""
        import tiktoken
        import trm.infer
        import trm.runtime.checkpoints
        import trm.runtime.restore
        from instruments import dump_transcripts as tool
        from instruments import runlog

        events = []

        class Enc:
            def encode(self, text):
                return [len(word) for word in text.split()]

            def decode(self, ids):
                return " ".join("w" * i for i in ids)

        def generate(model, enc, prompt, **kwargs):
            events.append(("generate", prompt))
            return enc.encode(prompt) + [3, 4, 5]

        monkeypatch.setattr(tiktoken, "get_encoding", lambda name: Enc())
        monkeypatch.setattr(trm.infer, "generate_text", generate)
        monkeypatch.setattr(trm.runtime.restore, "restore_model",
                            lambda path: events.append(("restore", path)) or ("model", 1279))
        monkeypatch.setattr(tool, "git_head", lambda: events.append(("commit",)) or "abc1234")
        real_load = runlog.load
        monkeypatch.setattr(runlog, "load", lambda path: events.append(("run log", path)) or real_load(path))
        if run_dir is None:
            located = ["--checkpoint-path", str(tmp_path / "ck")]
        else:
            located = []
            monkeypatch.chdir(run_dir.parents[1])  # main names the run as runs/<run_id>, relative
            monkeypatch.setattr(trm.runtime.checkpoints, "discover_latest_checkpoint_run",
                                lambda: (str(run_dir / "checkpoints"), run_dir.name))
        # select_device writes these; setting them first lets monkeypatch put them back.
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")
        monkeypatch.setenv("FORCE_F32_COMPUTE", "1")
        capsys.readouterr()
        tool.main([*located, "--out", str(tmp_path / "out"), "--depths", "1", *argv])
        path = tool.written_transcript(capsys.readouterr().out)
        return events, path, open(path).read()

    def test_the_commit_and_the_weights_are_read_before_any_generation(self, tmp_path, monkeypatch, capsys):
        events, _, document = self._run(tmp_path, monkeypatch, capsys)
        first_generation = next(i for i, e in enumerate(events) if e[0] == "generate")
        assert events.index(("commit",)) < first_generation
        assert [i for i, e in enumerate(events) if e[0] == "restore"][0] < first_generation
        assert document.count("tool_commit: abc1234") == 1 and events.count(("commit",)) == 1, \
            "the frontmatter uses the value captured before generation, not a second read"

    def test_the_run_metadata_is_read_before_any_generation(self, tmp_path, monkeypatch, capsys):
        """`model_commit` and the CE fields come from the run's own files, read once before
        generation, like the tool commit. Only a discovered run has a run dir to read."""
        import json

        run_dir = tmp_path / "runs" / "run_t"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.csv").write_text("step,ce,val_ce\n5,6.1,\n10,5.9,6.2\n")
        (run_dir / "run_metadata.json").write_text(json.dumps({"git_commit": "fedcba9876543", "git_dirty": False}))
        events, _, document = self._run(tmp_path, monkeypatch, capsys, run_dir=run_dir)

        first_generation = next(i for i, e in enumerate(events) if e[0] == "generate")
        (read,) = [i for i, e in enumerate(events) if e[0] == "run log"]
        assert read < first_generation
        assert "model_commit: fedcba9" in document

    def test_a_subset_runs_a_prefix_and_the_frontmatter_says_how_many(self, tmp_path, monkeypatch, capsys):
        events, path, document = self._run(tmp_path, monkeypatch, capsys,
                                           "--prompts", "2", "--prompt", "an extra prompt")
        generated = [e[1] for e in events if e[0] == "generate"]
        from instruments.dump_transcripts import PROMPTS
        assert generated == [*PROMPTS[:2], "an extra prompt"], "a prefix of the standard set, then extras"
        assert "standard_prompts: 2" in document
        assert "non-standard" in document, "the extra prompt is marked as such"
        assert path.startswith(str(tmp_path)), "the contract line names the file this run wrote"


class TestPromptSubset:
    """`milestone_report --quick` asked for `--prompts 2`, a flag this tool did not
    define, so argparse exited and every quick report's transcripts section FAILED (#335)."""

    def test_milestone_reports_quick_argv_parses_with_the_real_parser(self, monkeypatch):
        """The argv milestone_report actually builds, fed through this tool's own parser:
        a flag drifting on either side fails here instead of in a report."""
        from instruments import milestone_report
        from instruments.dump_transcripts import build_arg_parser

        sent = []
        monkeypatch.setattr(milestone_report, "run_tool",
                            lambda module, argv, timeout: sent.append((module, argv)) or "")
        fwd_args = ["--checkpoint-path", "runs/run_x/checkpoints"]
        milestone_report.section_transcripts(fwd_args, list(milestone_report.QUICK_TRANSCRIPT_ARGS), None)

        ((module, argv),) = sent
        assert module == "instruments.dump_transcripts"
        args = build_arg_parser().parse_args(argv)
        assert (args.checkpoint_path, args.prompts, args.max_new_tokens) == ("runs/run_x/checkpoints", 2, 32)

    def test_the_default_is_the_whole_standard_set(self):
        from instruments.dump_transcripts import build_arg_parser
        assert build_arg_parser().parse_args([]).prompts == len(PROMPTS)

    @pytest.mark.parametrize("bad", ["0", str(len(PROMPTS) + 1), "two"])
    def test_a_count_outside_the_set_is_refused(self, bad):
        from instruments.dump_transcripts import build_arg_parser
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(["--prompts", bad])


def test_the_written_line_round_trips_and_ignores_human_output(tmp_path, capsys):
    """milestone_report finds the transcript through this line, not through the human
    `✨ <path>` print it used to scan for with the wrong word (#338)."""
    from instruments.dump_transcripts import announce_written, written_transcript

    announce_written(tmp_path / "step_000184_cpu.md")
    stdout = "▶ 8 prompts x 1 depths\n\n✨ runs/x/transcripts/step_000184_cpu.md\n" + capsys.readouterr().out
    assert written_transcript(stdout) == str(tmp_path / "step_000184_cpu.md")
    assert written_transcript("✨ runs/x/transcripts/step_000184_cpu.md\n") is None
