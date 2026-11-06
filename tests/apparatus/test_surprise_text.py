"""The #465 surprise picture has to show what it claims: every target drawn exactly once,
where it sits, byte-level tokens included, and one colour scale across milestones."""

import numpy as np
import tiktoken

from instruments import surprise_text as st

EOT = 50256


def test_every_target_gets_exactly_one_cell_in_order():
    texts = ["Hello", ",", " world", "\n", "\n\n", "def", " f", "():", "\t", "x", " " * 3, ""]
    cells = st.layout(texts, width=12)
    assert len(cells) == len(texts)
    assert [c[1] for c in cells[:3]] == [0, 5, 6], "laid end to end on a line"
    assert all(text for _, _, text in cells), "an empty or whitespace token is still drawn"
    positions = [(line, col) for line, col, _ in cells]
    assert positions == sorted(positions), "reading order is the passage's order"


def test_a_long_token_wraps_and_never_runs_past_the_edge():
    cells = st.layout(["a" * 8, "b" * 8, "c" * 3], width=10)
    assert [c[0] for c in cells] == [0, 1, 2], "8 + 3 would pass 10: the third token starts a line"
    assert all(col + len(text) <= 10 for _, col, text in cells)


def test_a_token_that_is_part_of_a_character_is_still_drawn():
    enc = tiktoken.get_encoding("r50k_base")
    ids = enc.encode("naïve 東京")
    pieces = [st.token_text(enc, t) for t in ids]
    assert len(st.layout(pieces)) == len(ids), "one cell per token, split characters included"
    assert any(p.startswith("\\x") for p in pieces), "this text has a token that isn't valid UTF-8 alone"


def test_one_colour_scale_covers_every_milestone():
    early, late = np.array([9.0, 8.0, 7.5]), np.array([1.0, 0.5, 2.0])
    vmin, vmax = st.colour_scale([early, late], quantile=1.0)
    assert (vmin, vmax) == (0.0, 9.0), "the late panel is drawn on the early one's scale"


def test_the_passage_is_the_first_long_enough_document_and_never_crosses_a_separator():
    row = np.array([5, 6, EOT, 10, 11, 12, EOT, 20, 21])
    assert st.passage_from_row(row, 10, EOT, shortest=3).tolist() == [10, 11, 12]
    assert st.passage_from_row(row, 2, EOT, shortest=3).tolist() == [10, 11]
    assert st.passage_from_row(np.array([1, 2, 3]), 2, EOT).tolist() == [1, 2], "no document starts: the head"
    short_first = np.array([1, EOT, 7, EOT, 8, 9, 10, 11])
    assert st.passage_from_row(short_first, 10, EOT, shortest=3).tolist() == [8, 9, 10, 11], \
        "a one-token document is skipped, and no passage spans two documents"


def test_a_newline_and_indent_token_keeps_the_indentation():
    cells = st.layout(["if", " x", ":", "\n    ", "return", "\n", "y"], width=40)
    assert len(cells) == 7
    assert cells[4] == (1, 4, "return"), "the indent the newline token carried moves the cursor"
    assert cells[6] == (2, 0, "y")


def test_a_cold_run_and_a_live_run_each_get_their_own_folder():
    from trm.settings import REPO_ROOT
    assert st.default_out("/mnt/d_drive/TRM_cold/run_A/milestones") == REPO_ROOT / "runs/run_A/surprise_text"
    assert st.default_out("runs/run_B/checkpoints/milestones") == REPO_ROOT / "runs/run_B/surprise_text"
    assert st.default_out("/tmp/elsewhere/milestones") is None


def test_largest_drops_name_what_was_just_learned():
    drops = st.largest_drops(np.array([5.0, 1.0, 4.0]), np.array([1.0, 1.5, 3.5]), ["a", "b", "c"])
    assert drops == [("a", 5.0, 1.0), ("c", 4.0, 3.5)], "biggest fall first; a rise is not a drop"


def test_a_pinned_passage_is_refused_when_its_tokens_change(tmp_path):
    path = tmp_path / "passages.json"
    path.write_text('{"fineweb-edu": {"tokens": [1, 2, 3], "sha256": "%s"}}' % st.sha([1, 2, 4]))
    import pytest
    with pytest.raises(SystemExit, match="sha256"):
        st.load_passages(path, None, None, 10)
