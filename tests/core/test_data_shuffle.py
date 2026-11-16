"""DATA_SHUFFLE: a bucket's rows in a seeded order, read off the corpus as it is (#378).

What has to hold for the knob to be safe to turn on: every row before the held-out
tail is served once, the tail never, a resume continues the same order, and a
checkpoint never crosses from one order to the other. And off stays the stream every
run on record read.
"""

import json

import numpy as np
import pytest

from trm.data.loaders import DataMixer, ShuffledTextDataGenerator, TextDataGenerator, tokens_before_heldout
from trm.data.stream_state import data_refusal
from trm.settings import CONFIG, Config

MAX_SEQ_LEN = 4
STRIDE = 2 * MAX_SEQ_LEN + 1
HELDOUT = 7


def _corpus(root, name, tokens_per_file):
    """Chunk files whose every token is its own position in the corpus, so a row says
    exactly which tokens it is."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    start = 0
    for i, n in enumerate(tokens_per_file):
        np.save(d / f"chunk_{i}.npy", np.arange(start, start + n, dtype=np.uint16))
        start += n
    return str(d)


def _ragged(rows_per_file):
    """File lengths with a tail that fills no row, like a real shard."""
    return [rows * STRIDE + 5 for rows in rows_per_file]


def _shuffled(directory, seed=0, heldout=HELDOUT):
    return ShuffledTextDataGenerator(directory, max_seq_len=MAX_SEQ_LEN, rng=np.random.default_rng(seed),
                                     heldout_rows=heldout)


def _drain(reader, batch_size=1):
    rows = []
    while (batch := reader.get_batch(batch_size)) is not None:
        rows.extend(batch)
    return rows


def _heldout_tokens(directory, monkeypatch, data_seed, heldout=HELDOUT):
    """The tokens the trainer's probe may score: the tail, read the way it reads it."""
    from trm.train import validation
    monkeypatch.setattr(validation, "VAL_TAIL_ROWS", heldout)
    rows = validation.read_heldout_rows(directory, heldout, validation.tail_skip(directory, MAX_SEQ_LEN),
                                        max_seq_len=MAX_SEQ_LEN, data_seed=data_seed)
    return {int(t) for row in rows for t in row.ravel()}


def test_every_row_before_the_tail_is_served_exactly_once(tmp_path):
    lengths = _ragged([12, 9, 15, 11])
    rows = _drain(_shuffled(_corpus(tmp_path, "web", lengths)))
    served = [int(t) for row in rows for t in row]
    assert len(served) == len(set(served)), "a token was served twice"
    assert all(np.array_equal(row, np.arange(row[0], row[0] + STRIDE)) for row in rows), \
        "a row is one stride of consecutive tokens"
    # Each file gives up less than a stride to its start offset and less than one to its
    # ragged end; everything else before the tail is read.
    trainable = sum(tokens_before_heldout(lengths, STRIDE, HELDOUT))
    assert trainable - len(served) < 2 * STRIDE * len(lengths)
    assert len(rows) >= sum(n // STRIDE for n in lengths) - HELDOUT - len(lengths)


def test_the_rows_come_shuffled_and_the_seed_chooses_the_order(tmp_path):
    directory = _corpus(tmp_path, "web", _ragged([30, 30, 30, 30, 30, 30]))
    first = [int(row[0]) for row in _drain(_shuffled(directory, seed=0))]
    again = [int(row[0]) for row in _drain(_shuffled(directory, seed=0))]
    other = [int(row[0]) for row in _drain(_shuffled(directory, seed=1))]
    assert first == again, "one seed is one order"
    assert first != other, "two seeds read the rows in different orders"
    assert first != sorted(first), "the order is not the corpus's own"
    ascending = sum(b > a for a, b in zip(first, first[1:], strict=False))
    assert ascending < 0.75 * len(first), "rows inside a file are permuted, not read front to back"


def test_a_batch_spans_a_file_boundary_without_dropping_a_row(tmp_path):
    directory = _corpus(tmp_path, "web", _ragged([11, 13, 9]))
    one_by_one = _drain(_shuffled(directory))
    in_threes = _drain(_shuffled(directory), batch_size=3)
    whole = len(one_by_one) - len(one_by_one) % 3
    np.testing.assert_array_equal(np.array(in_threes), np.array(one_by_one[:whole]))


def test_off_is_the_stream_every_run_on_record_read(tmp_path, monkeypatch):
    """The in-order reader, rebuilt here from its definition: files in sorted order,
    each from an offset the reader's rng draws, front to back."""
    from trm.train import trainer
    lengths = _ragged([6, 4, 5])
    directory = _corpus(tmp_path, "web", lengths)
    config = Config.from_env({"DATA_SEED": "3", "MAX_SEQ_LEN": str(MAX_SEQ_LEN)})
    assert config.DATA_SHUFFLE is False
    reader = trainer.bucket_reader(config, directory)
    assert type(reader) is TextDataGenerator

    rng, expected, start = np.random.default_rng(3), [], 0
    for n in lengths:
        offset = int(rng.integers(0, STRIDE))
        expected += [start + offset + i * STRIDE for i in range((n - offset) // STRIDE)]
        start += n
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    assert [int(row[0]) for row in _drain(reader)] == expected


def _mixer(tmp_path, seed=0):
    dirs = [_corpus(tmp_path, "web", _ragged([10, 8, 12])), _corpus(tmp_path, "code", _ragged([9, 11]))]
    return DataMixer([_shuffled(d, seed) for d in dirs], [0.6, 0.4], rng=np.random.default_rng(11),
                     names=["web", "code"])


@pytest.mark.parametrize("cut", [1, 6, 14, 23, 31])  # mid-file, across file ends, past a source's end
def test_a_restored_mixer_serves_the_rows_an_uninterrupted_one_would(tmp_path, cut):
    reference = _mixer(tmp_path)
    expected = [reference.get_batch(1) for _ in range(40)]
    assert any(row is None for row in expected), "the reference runs past the end of the corpus"

    first = _mixer(tmp_path)
    for _ in range(cut):
        first.get_batch(1)
    snapshot = json.loads(json.dumps(first.state()))  # the snapshot crosses a checkpoint as JSON

    # Built with another seed: the order has to come from the snapshot, not the constructor.
    resumed = _mixer(tmp_path, seed=5)
    resumed.load_state(snapshot, legacy_names=())
    for want in expected[cut:]:
        got = resumed.get_batch(1)
        if want is None:
            assert got is None
            break
        np.testing.assert_array_equal(got, want)


def test_the_state_is_what_regenerates_the_order_not_the_order(tmp_path):
    reader = _shuffled(_corpus(tmp_path, "web", _ragged([40, 40])))
    reader.get_batch(3)
    assert set(reader.state()) == {"order_seed", "file_pos", "row"}
    assert all(isinstance(v, int) for v in reader.state().values())


@pytest.mark.parametrize("skip", [0, 1, 8, 9, 15, 19, 25])  # inside a file, to its end, across two, past the corpus
def test_advance_skips_the_rows_reading_them_would(tmp_path, skip):
    """The branch offset of #489, in the shuffled order."""
    directory = _corpus(tmp_path, "web", _ragged([10, 8, 12]))
    reference = _drain(_shuffled(directory))
    reader = _shuffled(directory)
    reader.get_batch(2)
    reader.advance(skip)
    assert [int(row[0]) for row in _drain(reader)] == [int(row[0]) for row in reference[2 + skip:]]


# The shapes the held-out tail can take: inside the last file, across the last two, and
# starting exactly where a file's rows end (with a ragged end the probe's skip lands in
# that file's leftover, and the next file is then read from a drawn offset).
TAIL_SHAPES = {
    "inside the last file": [15 * STRIDE + 5, 12 * STRIDE + 5],
    "across two files": [15 * STRIDE + 5, 4 * STRIDE + 5, 3 * STRIDE + 5],
    "from a file's first row": [15 * STRIDE, HELDOUT * STRIDE],
    "from a ragged file end": [15 * STRIDE + 5, HELDOUT * STRIDE + 5],
    "a small last file": [15 * STRIDE + 5, 13 * STRIDE + 5, 2],
}


@pytest.mark.parametrize("shape", TAIL_SHAPES)
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_no_shuffled_row_holds_a_token_the_probe_scores(tmp_path, monkeypatch, shape, seed):
    directory = _corpus(tmp_path, "web", TAIL_SHAPES[shape])
    heldout = _heldout_tokens(directory, monkeypatch, data_seed=seed)
    # Not always the whole tail: past a file boundary the probe's reader draws an offset.
    assert len(heldout) >= (HELDOUT - 1) * STRIDE, "the probe read its tail"
    for train_seed in range(4):
        rows = _drain(_shuffled(directory, train_seed))
        assert rows, "there is something to train on"
        assert not heldout & {int(t) for row in rows for t in row}


def test_the_trainer_keeps_the_shuffled_reader_off_the_probes_tail(tmp_path, monkeypatch):
    """The tail the trainer hands the reader is the one the probes read, by name."""
    from trm.train import trainer, validation
    assert trainer.VAL_TAIL_ROWS is validation.VAL_TAIL_ROWS
    directory = _corpus(tmp_path, "web", TAIL_SHAPES["across two files"])
    monkeypatch.setattr(trainer, "VAL_TAIL_ROWS", HELDOUT)
    reader = trainer.bucket_reader(Config.from_env({"DATA_SHUFFLE": "1", "MAX_SEQ_LEN": str(MAX_SEQ_LEN)}), directory)
    assert type(reader) is ShuffledTextDataGenerator
    heldout = _heldout_tokens(directory, monkeypatch, data_seed=CONFIG.DATA_SEED)
    assert not heldout & {int(t) for row in _drain(reader) for t in row}


def test_a_bucket_no_larger_than_its_tail_is_refused(tmp_path):
    """Its every row is the probe's, so a shuffle has nothing to serve: said at the
    reader's construction, not as a bucket the mixer quietly drops."""
    directory = _corpus(tmp_path, "web", _ragged([3, 3]))
    with pytest.raises(ValueError, match="held-out tail"):
        _shuffled(directory, heldout=6)


@pytest.mark.parametrize("branch", [False, True])
@pytest.mark.parametrize("state, shuffle, refused", [
    (None, False, False),                 # before #424: read in order
    (None, True, True),
    ({"corpus": "data"}, False, False),   # before the knob: read in order
    ({"corpus": "data"}, True, True),
    ({"corpus": "data", "shuffle": False}, True, True),
    ({"corpus": "data", "shuffle": True}, False, True),
    ({"corpus": "data", "shuffle": True}, True, False),
])
def test_a_checkpoint_is_only_resumed_in_the_order_it_was_read(state, shuffle, refused, branch):
    """A branch too: it continues every bucket where the checkpoint stopped."""
    if state is not None:
        state = {**state, "names": ["web"]}
    why = data_refusal(state, "runs/data", "web=1.0", branch=branch, shuffle=shuffle)
    assert ("DATA_SHUFFLE" in why) if refused else why is None


def test_a_resume_under_the_other_setting_stops_before_it_touches_a_reader(monkeypatch):
    from trm.train import trainer
    monkeypatch.setattr(trainer, "DATA_ROOT", "runs/data")
    in_order = {"corpus": "data", "sources": []}
    with pytest.raises(SystemExit, match="DATA_SHUFFLE"):
        trainer.resume_stream(Config.from_env({"DATA_SHUFFLE": "1"}), None, None, None, 1, 0, in_order)
    with pytest.raises(SystemExit, match="DATA_SHUFFLE"):
        trainer.resume_stream(Config.from_env({}), None, None, None, 1, 0, {**in_order, "shuffle": True})


def test_a_reader_refuses_the_other_readers_position(tmp_path):
    directory = _corpus(tmp_path, "web", _ragged([10, 8]))
    in_order = TextDataGenerator(directory, max_seq_len=MAX_SEQ_LEN, rng=np.random.default_rng(0))
    shuffled = _shuffled(directory)
    with pytest.raises(ValueError, match="DATA_SHUFFLE is on"):
        shuffled.load_state(in_order.state())
    with pytest.raises(ValueError, match="DATA_SHUFFLE is off"):
        in_order.load_state(shuffled.state())


def test_the_pipeline_records_the_order_with_every_batch(tmp_path, monkeypatch):
    """What `data_refusal` reads on a resume is written here, beside the corpus name."""
    from trm.train import trainer
    _corpus(tmp_path / "data" / "pretrain", "web", _ragged([20, 20]))
    monkeypatch.setattr(trainer, "DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(trainer, "VAL_TAIL_ROWS", HELDOUT)
    config = Config.from_env({"DATA_SHUFFLE": "1", "DATA_MIXTURE": "pretrain/web=1.0",
                              "MAX_SEQ_LEN": str(MAX_SEQ_LEN)})
    _, _, state = trainer.setup_data_pipeline(config, 0).get(timeout=30)
    assert state["shuffle"] is True and state["corpus"] == "data"
    assert data_refusal(state, str(tmp_path / "data"), config.DATA_MIXTURE, branch=False, shuffle=True) is None
    assert "DATA_SHUFFLE" in data_refusal(state, str(tmp_path / "data"), config.DATA_MIXTURE,
                                          branch=False, shuffle=False)


def test_the_run_records_whether_it_shuffled():
    from trm.runtime.run_tracker import RunTracker
    assert RunTracker.get_hyperparameters(Config.from_env({"DATA_SHUFFLE": "1"}))["DATA_SHUFFLE"] is True
