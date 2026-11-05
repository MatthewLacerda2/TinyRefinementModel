"""Whether a checkpoint's data position can be read on this run's data (#597, #607).

Jax-free, so the trainer and the supervisor ask the same question: the supervisor
before it launches or relaunches a trainer (`rewind.unresumable`, #505), the trainer
before it touches a reader. A position is refused on a corpus it was not read from,
and an exact resume on a mixture it was not written for. A branch (#489) rebuilds the
stream by bucket name, so only the corpus binds it.
"""

import os

from trm.settings import LEGACY_DATA_MIXTURE

# A rebuilt corpus goes in a new folder (trm/data/prefill.py), so the folder's name says
# which corpus a run read. Every state saved before the name was recorded read this one.
LEGACY_CORPUS = "data"


def corpus_name(data_root):
    return os.path.basename(os.path.normpath(str(data_root)))


def mixture_buckets(mixture):
    """DATA_MIXTURE's buckets, in mixer order: the names `parse_mixture` returns, without
    its checks (tests/core/test_data_mixture.py keeps the two equal)."""
    return [entry.partition("=")[0].strip() for entry in mixture.split(",") if entry.strip()]


def data_refusal(data_state, data_root, mixture, branch):
    """Why this run must not read `data_state`'s position, or None. A state saved before
    #439 names no buckets; every run then read LEGACY_DATA_MIXTURE."""
    if data_state is None:
        return None
    written_on = data_state.get("corpus", LEGACY_CORPUS)
    if data_root and written_on != corpus_name(data_root):
        return (f"this checkpoint read the corpus '{written_on}', and DATA_ROOT is "
                f"'{corpus_name(data_root)}': its data position means nothing there")
    if branch:
        return None
    written_for = list(data_state.get("names") or mixture_buckets(LEGACY_DATA_MIXTURE))
    reads = mixture_buckets(mixture)
    if written_for != reads:
        return (f"data state was written for the mixture {written_for}, and this run reads "
                f"{reads}: a different DATA_MIXTURE cannot resume a run's exact data stream")
    return None
