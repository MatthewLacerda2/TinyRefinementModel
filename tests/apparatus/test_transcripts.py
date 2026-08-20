"""The transcript logbook has to stay comparable across checkpoints (#203).

Its whole value is that an entry written today can be read against one written
three weeks and two billion tokens later. That only holds if the things which
define a comparison — the prompt set, the depth ladder, the seed, the device, the
step arithmetic — are pinned rather than drifting quietly.

So these guard the properties that make an old transcript *interpretable*, plus
the one that stops the tool from killing the run it exists to observe.

Deliberately importable without JAX: everything here is pure Python, so the suite
stays fast and can run beside a live trainer without competing for memory.
"""

import pytest

from instruments.dump_transcripts import (
    DEFAULT_DEPTHS,
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
        assert opt_step_from_checkpoint(1409023, 128) == 11008


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


class TestProvenanceTiming:
    """`tool_commit` must describe the code that generated the text (#214).

    It was read while assembling the frontmatter, after every completion. On a
    GPU pass that gap is seconds; on the CPU fallback a full entry takes ~1h48m,
    and the first real entry recorded `5893bd7` for text generated on `e25be42`
    because #207 and #206 both merged mid-run — #206 changing the inference path.

    The field exists so a future session can tell which code produced a
    transcript, so being wrong on exactly the long runs is the whole failure.

    Source-order rather than behavioural: exercising the real timing would mean
    restoring a checkpoint and generating, which is far too heavy for this tier.
    The repo already inspects source this way in `tests/core/test_infer_arch.py`.
    """

    def _main_source(self):
        from pathlib import Path

        import instruments.dump_transcripts as tool

        source = Path(tool.__file__).read_text()
        return source.split("def main(")[1].split("\ndef ")[0]

    def test_the_commit_is_read_before_any_generation(self):
        body = self._main_source()
        captured = body.index("tool_commit = git_head()")
        generated = body.index("for depth in depths:")

        assert captured < generated, (
            "tool_commit is read after generation starts — a long run will stamp "
            "itself with whatever merged while it was working")

    def test_the_frontmatter_uses_the_captured_value(self):
        """A second `git_head()` call at write time would reintroduce the bug
        even with the early capture sitting right above it."""
        body = self._main_source()
        fields = body.split('"tool_commit":')[1].split(",")[0]

        assert "git_head()" not in fields, (
            "the frontmatter calls git_head() again instead of using the value "
            "captured before generation")

    def test_the_checkpoint_derived_fields_were_never_affected(self):
        """`model_commit` comes from run_metadata.json and `checkpoint_step` from
        the restored checkpoint — both read before generation, both already
        correct. Pinned so a future refactor does not move them the wrong way."""
        body = self._main_source()

        assert body.index("restore_model(") < body.index("for depth in depths:")
