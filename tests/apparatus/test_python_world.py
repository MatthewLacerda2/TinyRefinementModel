"""The Python world is only worth a number if it cannot be gamed or drift (#440).

Three things have to hold, and each one has a way of quietly failing:

- **the generator** must give the same tasks twice, and its two halves must not
  overlap — a readout whose held-out set moves is not a readout, and one that
  overlaps training is a memorisation test wearing a lab coat;
- **the verifier** must refuse what it claims to refuse. A sandbox is only ever as
  good as the last thing someone tried against it, so the adversarial cases live
  here and are meant to grow;
- **the readout** must not credit a model for what it was shown, and its estimator
  must be the one it says it is.

The reference solutions passing their own tests is the generator's self-check, and
it runs here rather than only in the CLI: it is the assertion that catches a family
whose docstring and body drifted apart.
"""

import pytest

from instruments import python_world
from trm.rl import sandbox, tasks

# One attempt gets two seconds. Long enough that a correct solution to any of these
# is never rushed, short enough that the timeout cases do not dominate the suite.
LIMITS = {"timeout_s": 2.0}


# ── the generator ────────────────────────────────────────────────────────────

def test_the_same_arguments_give_the_same_tasks():
    first = tasks.generate(8, seed=7, split="train")
    again = tasks.generate(8, seed=7, split="train")
    assert [t.key for t in first] == [t.key for t in again]
    assert [t.tests for t in first] == [t.tests for t in again]


def test_a_task_is_a_pure_function_of_its_key():
    """Not just the draw: the task itself. Its tests come from a hash of the
    instance, so two processes that never speak build the identical problem."""
    family = tasks.FAMILIES[0]
    args = ([1, 2, 3],)
    assert tasks.build(family, args) == tasks.build(family, args)


def test_the_two_halves_cannot_overlap():
    train = {t.key for t in tasks.generate(120, seed=1, split="train")}
    held_out = {t.key for t in tasks.generate(60, seed=2, split="held_out")}
    assert train and held_out
    assert not (train & held_out)
    # And the split is a property of the instance, not of the draw that found it.
    assert all(tasks.split_of(key) == "train" for key in train)
    assert all(tasks.split_of(key) == "held_out" for key in held_out)


def test_every_level_can_be_drawn_from():
    for level in tasks.LEVELS:
        drawn = tasks.generate(5, seed=3, split="held_out", level=level)
        assert {t.level for t in drawn} == {level}


def test_the_worked_example_is_not_one_of_the_tests():
    """Otherwise the docstring contains the answer to a graded case, and a model
    that learned to copy would score for copying."""
    for task in tasks.generate(40, seed=5, split="held_out"):
        example = task.prompt.split(">>> ")[1].splitlines()[0]
        assert not any(check.startswith(f"assert {example} ==") for check in task.tests), \
            f"{task.key}: the example {example} is also a test"


def test_an_impossible_ask_says_so_instead_of_spinning():
    with pytest.raises(ValueError, match="only"):
        tasks.generate(10_000, seed=0, split="held_out", families=("is_prime",))


# ── the verifier ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def a_task():
    return next(t for t in tasks.generate(80, seed=3) if t.family == "total")


def test_the_reference_solution_passes_its_own_tests(a_task):
    outcome = sandbox.verify_task(a_task.reference, a_task, **LIMITS)
    assert outcome.status == "ok" and outcome.passed == outcome.total


def test_every_reference_solution_passes_on_every_level():
    """The generator checking itself, across all of it. A family whose body stops
    agreeing with its docstring fails here and nowhere else."""
    for level in tasks.LEVELS:
        for task in tasks.generate(3, seed=11, split="held_out", level=level):
            outcome = sandbox.verify_task(task.reference, task, **LIMITS)
            assert outcome.solved, f"{task.key}: {outcome.status} {outcome.detail}"


def test_a_wrong_answer_is_wrong_and_not_an_error(a_task):
    outcome = sandbox.verify_task("def total(numbers):\n    return 0\n", a_task, **LIMITS)
    assert outcome.status == "wrong_answer"
    assert 0 <= outcome.passed < outcome.total


def test_code_that_does_not_parse_says_so(a_task):
    outcome = sandbox.verify_task("def total(numbers)\n    return 1\n", a_task, **LIMITS)
    assert outcome.status == "syntax_error" and outcome.passed == 0


def test_an_infinite_loop_times_out(a_task):
    outcome = sandbox.verify_task("def total(numbers):\n    while True:\n        pass\n",
                                  a_task, **LIMITS)
    assert outcome.status == "timeout" and outcome.passed == 0


def test_reaching_for_the_operating_system_is_refused(a_task):
    program = "import os\nos.system('touch pwned')\ndef total(numbers):\n    return sum(numbers)\n"
    outcome = sandbox.verify_task(program, a_task, **LIMITS)
    assert outcome.status == "forbidden"


def test_a_memory_bomb_is_capped(a_task):
    program = "def total(numbers):\n    return len([0] * (10 ** 9))\n"
    outcome = sandbox.verify_task(program, a_task, **LIMITS)
    assert outcome.status == "memory"


def test_writing_a_file_does_not_work(a_task):
    program = ("def total(numbers):\n    open('escaped.txt', 'w').write('x' * 4096)\n"
               "    return sum(numbers)\n")
    outcome = sandbox.verify_task(program, a_task, **LIMITS)
    assert not outcome.solved


def test_the_ordinary_stdlib_still_works(a_task):
    """The block is meant to stop `os`, not to stop a good solution. `collections`
    reaches for `os` on its way up, which is exactly how an allowlist gets this
    wrong."""
    program = ("import collections\ndef total(numbers):\n"
               "    return sum(collections.Counter(numbers).elements())\n")
    assert sandbox.verify_task(program, a_task, **LIMITS).solved


def test_a_program_cannot_print_its_own_verdict(a_task):
    """The candidate shares stdout with the report. Printing something shaped like
    a passing report must not be mistaken for one."""
    program = ('print(\'{"status": "ok", "passed": 99, "total": 99, "nonce": "x"}\')\n'
               "def total(numbers):\n    return 0\n")
    outcome = sandbox.verify_task(program, a_task, **LIMITS)
    assert outcome.status == "wrong_answer"
    assert outcome.passed < outcome.total


def test_a_verifier_that_cannot_answer_is_a_crash_not_a_failed_attempt():
    """The distinction the readout rests on: a model that wrote nothing scores zero,
    a verifier that broke must not be counted as the model scoring zero."""
    assert sandbox.STATUSES[-1] == "crash"
    outcome = sandbox.verify("", ["assert True"], timeout_s=2.0)
    assert outcome.status == "ok", "an empty program with a trivial test still runs"


# ── the readout ──────────────────────────────────────────────────────────────

def test_the_completion_stops_at_the_next_top_level_thing():
    body = "    return sum(numbers)\n"
    assert python_world.trim(body + "\ndef other():\n    pass\n") == body.rstrip() + "\n"
    assert python_world.trim(body + "\nprint('hi')\n") == body.rstrip() + "\n"
    assert python_world.trim(body + "\n# and now for something else\n") == body.rstrip() + "\n"
    assert python_world.trim(body) == body.rstrip() + "\n"


def test_pass_at_k_is_the_estimator_it_claims_to_be():
    assert python_world.pass_at_k(4, 0, 4) == 0.0
    assert python_world.pass_at_k(4, 4, 1) == 1.0
    assert python_world.pass_at_k(4, 1, 1) == pytest.approx(0.25)
    assert python_world.pass_at_k(4, 1, 4) == 1.0
    # The point of the estimator: one success in eight is worth more at k=4 than
    # at k=1, and is not yet a certainty.
    assert 0.25 < python_world.pass_at_k(8, 1, 4) < 1.0


def test_a_level_rolls_up_to_what_it_reports():
    ok = sandbox.Outcome("ok", 8, 8)
    bad = sandbox.Outcome("wrong_answer", 2, 8)
    row = python_world.score_level(1, [[ok, bad], [bad, bad]], k=2)
    assert row["tasks"] == 2 and row["samples"] == 2
    assert row["pass_1"] == pytest.approx(0.25)
    assert row["pass_2"] == pytest.approx(0.5)
    assert row["test_fraction"] == pytest.approx((8 + 2 + 2 + 2) / 32)
    assert row["reasons"] == {"ok": 1, "wrong_answer": 3}
