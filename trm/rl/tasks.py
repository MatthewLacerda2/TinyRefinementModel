"""Procedural Python tasks, graded by difficulty, with their own tests.

A fixed problem set is a finite resource: LeetCode and freeCodeCamp together are a
few thousand problems, and a model that has seen them has seen them. A *generator*
scales with compute, which is the only kind of curriculum the bitter lesson lets us
keep. So a task here is not a file — it is a family plus a seed.

**The reference solution is the only source of truth.** Each family is one ordinary
Python function in this module. Its signature and docstring become the prompt, its
body becomes the reference, and its return value on sampled inputs becomes the
expected output of every test. Nothing is written twice, so nothing can drift, and
"the reference passes its own tests" is true by construction rather than by care.

**Train and held-out are disjoint by construction, not by bookkeeping.** The split
is a hash of the concrete instance (family + arguments), so the same instance lands
in the same half forever, no matter how many tasks anyone draws, in what order, or
on which machine. There is no file of held-out ids to keep in sync.

The levels are a rough ladder, not a claim about what is hard for a transformer:

1. one operation over a list or a number
2. a filter, a map, or a single pass over a string
3. dictionaries, grouping, sorting by a key
4. small named algorithms
5. two ideas composed

What levels are *for* is separating two models. HumanEval and MBPP both read ~0 at
135M parameters and so rank a bad model equal to a worse one; a ladder always has a
rung where two models differ.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import random
import textwrap
from dataclasses import dataclass
from typing import Callable

# Held-out share. Ten percent of the instance space is plenty: the space is
# effectively unbounded, so the split's job is only to keep training from touching
# what the readout will ask, and a tenth of infinity is still infinity.
HELD_OUT_PERCENT = 10
SPLITS = ("train", "held_out")

# Every string family draws from this list. It is short and dull on purpose: the
# task is the transformation, and rare words would make it a tokenizer test.
WORDS = ("apple", "bread", "cat", "door", "echo", "fox", "green", "house", "ice",
         "jam", "kite", "lamp", "moon", "nest", "open", "pear", "queen", "rain",
         "sun", "tree", "up", "vase", "wind", "xray", "yarn", "zebra")


# ── the families ─────────────────────────────────────────────────────────────
# Each function is the reference solution AND the prompt AND the test oracle. Keep
# them short, keep them obviously correct, and keep the return value comparable
# with `==` — no floats, no sets, nothing whose repr depends on hash order.

def total(numbers):
    """Return the sum of the numbers in the list."""
    return sum(numbers)


def largest(numbers):
    """Return the largest number in the list, or None if the list is empty."""
    return max(numbers) if numbers else None


def count_value(numbers, value):
    """Return how many times value appears in the list."""
    return numbers.count(value)


def add_to_each(numbers, amount):
    """Return a new list with amount added to every number."""
    return [n + amount for n in numbers]


def last_digit(number):
    """Return the last digit of a non-negative whole number."""
    return number % 10


def sum_even(numbers):
    """Return the sum of only the even numbers in the list."""
    return sum(n for n in numbers if n % 2 == 0)


def squares(numbers):
    """Return a new list holding the square of every number."""
    return [n * n for n in numbers]


def reverse_text(text):
    """Return the text written backwards."""
    return text[::-1]


def count_vowels(text):
    """Return how many vowels (a, e, i, o, u) the text contains."""
    return sum(1 for ch in text if ch in "aeiou")


def longer_than(words, length):
    """Return the words that are strictly longer than length, in their original order."""
    return [w for w in words if len(w) > length]


def drop_duplicates(numbers):
    """Return the numbers with later repeats removed, keeping the first of each."""
    seen, out = [], []
    for n in numbers:
        if n not in seen:
            seen.append(n)
            out.append(n)
    return out


def word_count(words):
    """Return a dictionary mapping each word to how many times it appears."""
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts


def group_by_parity(numbers):
    """Return {'even': [...], 'odd': [...]} with the numbers in their original order."""
    return {"even": [n for n in numbers if n % 2 == 0],
            "odd": [n for n in numbers if n % 2 != 0]}


def most_common(numbers):
    """Return the number that appears most often; on a tie return the smallest of them."""
    if not numbers:
        return None
    counts = {}
    for n in numbers:
        counts[n] = counts.get(n, 0) + 1
    best = max(counts.values())
    return min(n for n, c in counts.items() if c == best)


def sort_by_length(words):
    """Return the words sorted by length, and alphabetically among equal lengths."""
    return sorted(words, key=lambda w: (len(w), w))


def invert_mapping(mapping):
    """Return a dictionary with the keys and values swapped."""
    return {value: key for key, value in mapping.items()}


def fizzbuzz(n):
    """Return the fizzbuzz list for 1..n: 'Fizz' for multiples of 3, 'Buzz' for
    multiples of 5, 'FizzBuzz' for both, otherwise the number itself as a string."""
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out


def fibonacci(n):
    """Return the first n Fibonacci numbers, starting 0, 1."""
    out, a, b = [], 0, 1
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out


def is_prime(number):
    """Return True if the number is prime, otherwise False."""
    if number < 2:
        return False
    divisor = 2
    while divisor * divisor <= number:
        if number % divisor == 0:
            return False
        divisor += 1
    return True


def run_length_encode(text):
    """Return a list of (character, run length) pairs for the text's runs."""
    out = []
    for ch in text:
        if out and out[-1][0] == ch:
            out[-1] = (ch, out[-1][1] + 1)
        else:
            out.append((ch, 1))
    return out


def balanced(text):
    """Return True if the round brackets in the text are balanced, otherwise False."""
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def two_sum(numbers, target):
    """Return the indices of the first two numbers that add up to target, as a list
    of two indices in increasing order, or None if no pair does."""
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] + numbers[j] == target:
                return [i, j]
    return None


def merge_sorted(left, right):
    """Return one sorted list holding every number from both sorted input lists."""
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    return out + left[i:] + right[j:]


def common_prefix(words):
    """Return the longest string every word starts with, or '' if there is none."""
    if not words:
        return ""
    prefix = words[0]
    for word in words[1:]:
        while not word.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


def transpose(rows):
    """Return the rectangular grid of numbers with its rows and columns swapped."""
    if not rows:
        return []
    return [[row[i] for row in rows] for i in range(len(rows[0]))]


def flatten_once(rows):
    """Return one list holding every element of the inner lists, in order."""
    return [item for row in rows for item in row]


def roman_to_int(text):
    """Return the value of an uppercase Roman numeral (I, V, X, L, C, D, M)."""
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    out = 0
    for i, ch in enumerate(text):
        if i + 1 < len(text) and values[ch] < values[text[i + 1]]:
            out -= values[ch]
        else:
            out += values[ch]
    return out


# ── how each family is sampled ───────────────────────────────────────────────
# A sampler returns the argument tuple for one instance. `edges` are the cases a
# generated draw would almost never produce and a wrong solution almost always
# trips on — empty, zero, negative, a tie. Every instance gets all of them, so a
# solution that ignores the empty list cannot pass by luck.

def _ints(rng, low=1, high=8, lo=-20, hi=20):
    return [rng.randint(lo, hi) for _ in range(rng.randint(low, high))]


def _words(rng, low=1, high=6):
    return [rng.choice(WORDS) for _ in range(rng.randint(low, high))]


def _letters(rng, low=1, high=12, alphabet="abcde"):
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(low, high)))


def _grid(rng):
    width, height = rng.randint(1, 4), rng.randint(1, 4)
    return ([[rng.randint(0, 9) for _ in range(width)] for _ in range(height)],)


@dataclass(frozen=True)
class Family:
    """One problem shape. `solution` is the reference, the prompt and the oracle."""
    solution: Callable
    level: int
    sampler: Callable[[random.Random], tuple]
    edges: tuple[tuple, ...] = ()

    @property
    def name(self) -> str:
        return self.solution.__name__


FAMILIES = (
    Family(total, 1, lambda r: (_ints(r),), (([],), ([0],), ([-3, 3],))),
    Family(largest, 1, lambda r: (_ints(r),), (([],), ([-5],), ([2, 2],))),
    Family(count_value, 1, lambda r: (_ints(r, lo=0, hi=4), r.randint(0, 4)),
           (([], 0), ([1, 1, 1], 1), ([1, 2], 3))),
    Family(add_to_each, 1, lambda r: (_ints(r), r.randint(-5, 5)),
           (([], 1), ([0], 0), ([-1, 1], -1))),
    Family(last_digit, 1, lambda r: (r.randint(0, 9999),), ((0,), (7,), (100,))),

    Family(sum_even, 2, lambda r: (_ints(r),), (([],), ([1, 3],), ([-2, 2],))),
    Family(squares, 2, lambda r: (_ints(r),), (([],), ([0],), ([-3],))),
    Family(reverse_text, 2, lambda r: (_letters(r),), (("",), ("a",), ("aba",))),
    Family(count_vowels, 2, lambda r: (_letters(r, alphabet="aeioubcd"),),
           (("",), ("bcd",), ("aeiou",))),
    Family(longer_than, 2, lambda r: (_words(r), r.randint(2, 5)),
           (([], 3), (["up"], 2), (["apple", "cat"], 0))),
    Family(drop_duplicates, 2, lambda r: (_ints(r, lo=0, hi=4),),
           (([],), ([1, 1, 1],), ([3, 1, 3],))),

    Family(word_count, 3, lambda r: (_words(r),), (([],), (["cat"],), (["cat", "cat"],))),
    Family(group_by_parity, 3, lambda r: (_ints(r),), (([],), ([2, 4],), ([-1, -2],))),
    Family(most_common, 3, lambda r: (_ints(r, lo=0, hi=4),),
           (([],), ([2, 1, 1, 2],), ([5],))),
    Family(sort_by_length, 3, lambda r: (_words(r),), (([],), (["cat", "jam"],), (["up"],))),
    Family(invert_mapping, 3,
           lambda r: ({w: i for i, w in enumerate(dict.fromkeys(_words(r)))},),
           (({},), ({"a": 1},), ({"a": 1, "b": 2},))),

    Family(fizzbuzz, 4, lambda r: (r.randint(1, 40),), ((1,), (3,), (15,))),
    Family(fibonacci, 4, lambda r: (r.randint(0, 20),), ((0,), (1,), (2,))),
    Family(is_prime, 4, lambda r: (r.randint(0, 200),), ((0,), (1,), (2,), (4,))),
    Family(run_length_encode, 4, lambda r: (_letters(r, alphabet="aab"),),
           (("",), ("a",), ("aab",))),
    Family(balanced, 4, lambda r: (_letters(r, alphabet="()ab"),),
           (("",), ("()",), (")(",))),
    Family(two_sum, 4, lambda r: (_ints(r, low=2, lo=0, hi=9), r.randint(0, 18)),
           (([], 0), ([1, 2], 3), ([1, 2], 9))),

    Family(merge_sorted, 5, lambda r: (sorted(_ints(r)), sorted(_ints(r))),
           (([], []), ([1], []), ([1, 1], [1]))),
    Family(common_prefix, 5, lambda r: (_words(r),),
           (([],), (["cat"],), (["cat", "car"],), (["a", "b"],))),
    Family(transpose, 5, _grid, (([],), ([[1]],), ([[1, 2]],))),
    Family(flatten_once, 5,
           lambda r: ([_ints(r, low=0, high=3) for _ in range(r.randint(0, 3))],),
           (([],), ([[]],), ([[1], []],))),
    Family(roman_to_int, 5,
           lambda r: ("".join(r.choice("IVXLCDM") for _ in range(r.randint(1, 4))),),
           (("I",), ("IV",), ("MCM",))),
)

LEVELS = tuple(sorted({family.level for family in FAMILIES}))


# ── a task ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Task:
    """One concrete problem: what the model is shown, and what decides the attempt.

    `family` is also the name of the function the attempt has to define — the
    reference solution is the family, so the two can never be different names.
    """
    family: str
    level: int
    key: str
    prompt: str
    tests: tuple[str, ...]
    reference: str

    @property
    def split(self) -> str:
        return split_of(self.key)


def split_of(key: str) -> str:
    """Which half an instance belongs to, from the instance itself.

    A hash, not a list: the same instance lands in the same half on every machine
    and in every process, so a training loop and a readout that never speak to each
    other still cannot overlap.
    """
    digest = hashlib.sha256(key.encode()).digest()
    return "held_out" if digest[0] * 100 // 256 < HELD_OUT_PERCENT else "train"


def _body(function) -> str:
    """The reference solution's body — its source with the def line and docstring cut."""
    source = textwrap.dedent(inspect.getsource(function))
    node = ast.parse(source).body[0]
    statements = node.body
    if (isinstance(statements[0], ast.Expr)
            and isinstance(statements[0].value, ast.Constant)
            and isinstance(statements[0].value.value, str)):
        statements = statements[1:]
    return "\n".join(source.splitlines()[statements[0].lineno - 1:])


def _signature(function) -> str:
    return f"def {function.__name__}{inspect.signature(function)}:"


def _call(name: str, args: tuple) -> str:
    return f"{name}({', '.join(repr(a) for a in args)})"


def build(family: Family, args: tuple, tests: int = 5) -> Task:
    """Turn one argument tuple into a task — a pure function of (family, args).

    The tuple is the task's identity and its worked example; the tests come from
    the identity's own hash, so the same instance always carries the same tests and
    "which half is this in" can be decided from the key alone.

    The worked example is deliberately *not* one of the tests. A model that copies
    the answer out of the docstring should score zero, not one.
    """
    name = family.name
    key = f"{name}:{args!r}"
    rng = random.Random(hashlib.sha256(key.encode()).digest())
    cases = list(family.edges) + [family.sampler(rng) for _ in range(tests)]

    docstring = "\n    ".join(inspect.getdoc(family.solution).splitlines())
    prompt = (f"{_signature(family.solution)}\n"
              f'    """{docstring}\n\n'
              f"    >>> {_call(name, args)}\n"
              f"    {family.solution(*args)!r}\n"
              f'    """\n')
    checks = tuple(f"assert {_call(name, case)} == {family.solution(*case)!r}"
                   for case in cases)
    return Task(family=name, level=family.level, key=key, prompt=prompt,
                tests=checks, reference=prompt + _body(family.solution))


def generate(count: int, *, seed: int = 0, split: str = "train", level: int | None = None,
             families: tuple[str, ...] | None = None) -> list[Task]:
    """`count` distinct tasks, the same ones every time for the same arguments.

    Instances that fall in the other half are drawn and discarded, which is what
    makes the two halves disjoint without either side keeping a list. A family whose
    instance space is genuinely small (there are only 200 `is_prime` draws) can run
    out; the loop gives up rather than spinning, and says so.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    pool = [f for f in FAMILIES
            if (level is None or f.level == level)
            and (families is None or f.name in families)]
    if not pool:
        raise ValueError(f"no family matches level={level} families={families}")

    rng = random.Random(seed)
    tasks: dict[str, Task] = {}
    for _ in range(count * 200):
        if len(tasks) >= count:
            break
        family = pool[rng.randrange(len(pool))]
        args = family.sampler(rng)
        key = f"{family.name}:{args!r}"
        if key in tasks or split_of(key) != split:
            continue
        tasks[key] = build(family, args)
    if len(tasks) < count:
        raise ValueError(
            f"only {len(tasks)} distinct {split} tasks exist for level={level} "
            f"families={families}; ask for fewer, or widen the pool")
    return list(tasks.values())
