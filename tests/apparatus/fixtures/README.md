# Recorded fixtures

Small copies of real artifacts. Tests read these so they run everywhere, CI included,
instead of reading `runs/`, which exists on one machine. The `recorded_run(name)` fixture
in `tests/conftest.py` lays a run out as `runs/<name>/` in a temp dir; `champion_run` is
`recorded_run("run_20260813_214725")`.

**Bytes are the trainer's.** Every CSV line is a byte-identical line of its source,
checked by streaming the source. That includes the CRLF line endings the trainer's
`csv` writer produces. `.gitattributes` marks these CSVs `-text` so git never
normalises them.

## `champion_run_metadata.json`

The full `run_metadata.json` of the 4B champion, `runs/run_20260813_214725` (added in
#301). Byte-for-byte identical to the source when checked again on 2026-09-16 (#325).

## `run_20260813_214725/metrics.csv`

An excerpt of the same run's `metrics.csv`, cut on 2026-09-16 (#325). The source is
6,080 lines, sha256 prefix `dc2895fe9cbe2488`. The excerpt has the header and 195 rows,
unchanged and in file order. It was cut by a streaming script that read the source line
by line:

- **Only rows `instruments.runlog` itself keeps.** Those are rows with a parseable,
  strictly advancing step: 6,079 of them. Replayed rows were never copied, so the
  excerpt loads to exactly the rows it holds.
- **The first 10 and last 10 of those rows** (steps 5 and 30,520 bound the excerpt), for
  `test_runlog`'s read of a real run.
- **Every row the invariants flag, plus 2 rows on each side**, for `test_invariants`'
  live-run pair. The source has 32 suspect rows under today's bounds
  (`MAX_STEPS_LIMIT=8`, `ACCUMULATION_STEPS=128`, 5σ): 24 with `depth_avg` 8.44–9.20,
  above the maximum depth of 8, and 8 below the corridor (#355). Selection used a looser
  test (`|depth_avg − 4.5| > 0.5`, plus every other invariant), so all 32 are kept, and
  `test_the_live_run_flags_rows_only_for_known_reasons` asserts exactly 32. The strict
  xfail on #196's `<= 5` bound therefore records the real failure here, not a trimmed one.

To re-cut after the bounds change, rerun the same selection on the source run.
Otherwise these are records: fixing #355's cause will not change them.

This excerpt is in the August format: it has no `wall_clock`, `mix`,
`arena_peak_mib`, `act_max` or `applied_*` columns. The next excerpt covers those.

## `run_287_adamw_lr0.0001_s0/`

A finished arm of the #287 peak-LR pair (plain, AdamW, seed 0, commit `57df54c`), in
today's metrics format, so `test_runlog`'s read check covers `wall_clock` (a datetime),
`mix` (text) and `arena_peak_mib`. Cut on 2026-09-16 (#325), with the bytes kept:

- **`metrics.csv`**: the header, the first 3 and last 2 of 103 data rows (steps 5, 10,
  15, 510, 515). Steps 10 and 515 carry `val_ce`. The source is 104 lines, sha256 prefix
  `8b5fc53791ecf8b7`.
- **`run_metadata.json`**: the run's full file, copied unchanged (877 bytes).

The trade: five rows prove the format parses, not that a whole current run does. The
champion excerpt above is the one with depth.
