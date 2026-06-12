"""The decoder's slot read positions vs the reasoning loop's write positions.

The decoder keys slots at negative positions, which wrap around the RoPE cache.
Today that wrap lands exactly on the write positions of the FINAL reasoning step
at FULL depth — an accident that happens to work, currently load-bearing while
random-depth training writes slots at shallower positions every micro-step.
These tests pin the accident down so any change to the cache size or position
scheme fails loudly instead of silently shifting where slots get keyed.
"""

import numpy as np

from config import MAX_SEQ_LEN, MAX_STEPS_LIMIT, SHARED_SLOTS

CACHE_LEN = MAX_SEQ_LEN + MAX_STEPS_LIMIT * SHARED_SLOTS


def test_decoder_slot_reads_land_on_full_depth_write_positions():
    # model.py: past_shared_pos = arange(-SHARED_SLOTS, 0), resolved by Python
    # negative indexing into the cos/sin caches of length CACHE_LEN.
    read_pos = CACHE_LEN + np.arange(-SHARED_SLOTS, 0)

    # model.py _reasoning_loop: step i writes at MAX_SEQ_LEN + i*SHARED_SLOTS ...
    final_step_write_pos = MAX_SEQ_LEN + (MAX_STEPS_LIMIT - 1) * SHARED_SLOTS + np.arange(SHARED_SLOTS)

    np.testing.assert_array_equal(
        read_pos, final_step_write_pos,
        err_msg="Decoder slot read positions no longer wrap onto the final-step "
                "write positions. If this is intentional, update model.py's "
                "past_shared_pos comment and this test together.",
    )


def test_rope_cache_covers_every_position_used():
    deepest_write = MAX_SEQ_LEN + MAX_STEPS_LIMIT * SHARED_SLOTS - 1
    slot_kv_home = MAX_SEQ_LEN + SHARED_SLOTS - 1  # base_shared_pos in the reasoning context
    assert deepest_write < CACHE_LEN
    assert slot_kv_home < CACHE_LEN
    assert MAX_SEQ_LEN - 1 < CACHE_LEN
