import numpy as np
import jax.numpy as jnp
import fsspec
from trm.config import MAX_SEQ_LEN, DATA_SEED, VOCAB_SIZE

class TextDataGenerator:
    def __init__(self, directory, max_seq_len=MAX_SEQ_LEN, rng=None):
        self.max_seq_len = max_seq_len
        self.directory = directory
        self.rng = rng if rng is not None else np.random.default_rng(DATA_SEED)

        self.fs, _ = fsspec.core.url_to_fs(directory)

        all_files = self.fs.ls(directory)
        self.files = sorted([f for f in all_files if f.endswith('.npy')])

        self.current_file_idx = 0
        self.data = None
        self.pointer = 0
        self.exhausted = False
        self.skip_count = 0
        self.is_new_file = False

    def _load_next_file(self):
        while True:
            if self.current_file_idx >= len(self.files):
                self.exhausted = True
                return False

            file_path = self.files[self.current_file_idx]
            print(f"📖 Memory-mapping {file_path} (lazy host-RAM paging)...")

            try:
                # Attempt direct OS-level memory mapping for zero-copy lazy paging
                self.data = np.load(file_path, mmap_mode='r')
            except (ValueError, TypeError, OSError):
                # Fallback to fsspec wrapper for remote or virtual file systems
                with self.fs.open(file_path, 'rb') as f:
                    self.data = np.load(f)

            self.pointer = 0
            stride = 2 * self.max_seq_len + 1

            if self.skip_count > 0:
                # Checkpoint recovery: Skip exact number of previously seen tokens
                tokens_to_skip = self.skip_count * stride
                if tokens_to_skip < len(self.data):
                    self.pointer = tokens_to_skip
                    self.skip_count = 0
                else:
                    self.skip_count -= (len(self.data) // stride)
                    self.current_file_idx += 1
                    continue  # Loop to load the next file
            else:
                # Dynamic Token Boundary Augmentation: apply a random starting offset
                random_offset = int(self.rng.integers(0, stride))
                if random_offset < len(self.data):
                    self.pointer = random_offset

            self.current_file_idx += 1
            self.is_new_file = True
            return True

    def state(self):
        """Where this reader is, exactly (#424): enough for `load_state` to continue
        with the very next row an uninterrupted reader would serve. JSON-safe."""
        return {"file_idx": int(self.current_file_idx), "pointer": int(self.pointer),
                "open": self.data is not None, "is_new_file": self.is_new_file,
                "exhausted": self.exhausted, "rng": self.rng.bit_generator.state}

    def load_state(self, state):
        """Continue from a `state()` snapshot. The open file is mapped again; the
        rng resumes mid-stream, so every later file draws the offset it would have."""
        self.rng.bit_generator.state = state["rng"]
        self.current_file_idx, self.pointer = state["file_idx"], state["pointer"]
        self.is_new_file, self.exhausted = state["is_new_file"], state["exhausted"]
        self.skip_count = 0
        self.data = None
        if state["open"]:
            path = self.files[self.current_file_idx - 1]
            print(f"📖 Memory-mapping {path} (resuming at token {self.pointer:,})...")
            self.data = np.load(path, mmap_mode='r')

    def get_batch(self, batch_size):
        if self.exhausted:
            return None, None

        stride = 2 * self.max_seq_len + 1
        total_tokens = batch_size * stride

        if self.data is None or self.pointer + total_tokens > len(self.data):
            if not self._load_next_file():
                return None, None
            if self.exhausted or self.pointer + total_tokens > len(self.data):
                return self.get_batch(batch_size)

        batch = self.data[self.pointer : self.pointer + total_tokens]
        self.pointer += total_tokens

        # The one place a bad id can be caught LOUDLY. The model clamps out-of-range
        # ids (#233) because it cannot raise from inside jit, and a clamp turns a
        # destroyed window into a wrong token — but it also hides the cause. Here we
        # are in numpy, outside any trace, so a corrupted shard or a tokenizer change
        # fails with its own name instead of surfacing days later as f16 instability.
        # One max() over ~1k ints per batch; unmeasurable against a forward pass.
        if batch.size:
            worst = int(batch.max())
            if worst >= VOCAB_SIZE:
                raise ValueError(
                    f"token id {worst} in {self.directory} exceeds VOCAB_SIZE "
                    f"{VOCAB_SIZE}. The model would clamp it to a wrong token rather "
                    f"than crash (#233), so this is the only place it can be caught: "
                    f"the shard is corrupt, or it was written by a different tokenizer.")

        # doc_boundary: marks batches that start a new file, i.e. the previous
        # document's carried state is no longer relevant downstream.
        doc_boundary = np.zeros((batch_size,), dtype=bool)
        if self.is_new_file:
            doc_boundary[:] = True
            self.is_new_file = False

        return jnp.array(batch.reshape(batch_size, stride), dtype=jnp.int32), jnp.array(doc_boundary)

class DataMixer:
    def __init__(self, sources, weights, rng=None, names=None):
        self.sources = list(sources)
        # What each source is called, in the same order (#439): the buckets of the
        # run's DATA_MIXTURE. Saved with the state, so a resume can tell whether it
        # is reading the mixture the checkpoint was written with.
        self.names = list(names) if names is not None else None
        if self.names is not None and len(self.names) != len(self.sources):
            raise ValueError(f"{len(self.names)} names for {len(self.sources)} sources")
        # Every source by its original index, alive or not: what `state()` saves.
        self._all = list(sources)
        self.weights = list(weights)
        self.rng = rng if rng is not None else np.random.default_rng(DATA_SEED)
        # Original index of each surviving source, so full-length weight lists
        # supplied via set_weights can be mapped after sources exhaust.
        self._alive = list(range(len(self.sources)))
        # The original index of the source the last batch came from, or None when
        # it mixed several (#364: per-source gradient telemetry).
        self.last_source = None

    def state(self):
        """Every reader's `state()` plus the mixer's own draw stream, which sources
        are still alive (#424), and what they are called (#439). JSON-safe."""
        state = {"rng": self.rng.bit_generator.state, "alive": list(self._alive),
                 "weights": [float(w) for w in self.weights], "sources": [s.state() for s in self._all]}
        if self.names is not None:
            state["names"] = list(self.names)
        return state

    def load_state(self, state):
        """Restore a snapshot, refusing one written for a different mixture.

        The state is positional — source i's reader, source i's weight — so the
        buckets have to be the same ones *in the same order*. A resume that silently
        mapped a code reader's position onto a web shard would not crash; it would
        train on a stream nobody chose. A state saved before #439 carries no names
        and is matched by count alone, as it always was.
        """
        saved = state.get("names")
        if saved is not None and self.names is not None and list(saved) != self.names:
            raise ValueError(f"data state was written for the mixture {list(saved)}, and this "
                             f"run reads {self.names}: a different DATA_MIXTURE cannot resume "
                             f"a run's exact data stream")
        if len(state["sources"]) != len(self._all):
            raise ValueError(f"data state has {len(state['sources'])} sources, "
                             f"the mixer {len(self._all)}: a different DATA_MIXTURE")
        self.rng.bit_generator.state = state["rng"]
        for source, source_state in zip(self._all, state["sources"]):
            source.load_state(source_state)
        self._alive = list(state["alive"])
        self.sources = [self._all[i] for i in self._alive]
        self.weights = list(state["weights"])

    def set_weights(self, weights):
        """Update mixture weights with a full-length list (one weight per
        *original* source). Weights of exhausted sources are dropped and the
        remainder renormalized — never mutate self.weights from outside, as the
        internal list shrinks when sources exhaust."""
        mapped = [weights[i] for i in self._alive]
        total = sum(mapped)
        if total > 0:
            self.weights = [w / total for w in mapped]

    def get_batch(self, batch_size):
        while len(self.sources) > 0:
            counts = self.rng.multinomial(batch_size, self.weights)
            batch_list = []
            exhausted_indices = []
            drawn_from = []

            for i, (source, count) in enumerate(zip(self.sources, counts)):
                if count > 0:
                    res = source.get_batch(count)
                    if res[0] is None or getattr(source, "exhausted", False):
                        exhausted_indices.append(i)
                    else:
                        batch_list.append(res)
                        drawn_from.append(self._alive[i])

            if exhausted_indices:
                new_sources, new_weights, new_alive = [], [], []
                for i, (s, w) in enumerate(zip(self.sources, self.weights)):
                    if i not in exhausted_indices:
                        new_sources.append(s)
                        new_weights.append(w)
                        new_alive.append(self._alive[i])
                self.sources = new_sources
                self._alive = new_alive
                if not self.sources:
                    return None, None
                total_w = sum(new_weights)
                self.weights = [w / total_w for w in new_weights]
                continue

            if batch_list:
                self.last_source = drawn_from[0] if len(drawn_from) == 1 else None
                batches, masks = zip(*batch_list)
                return jnp.concatenate(batches, axis=0), jnp.concatenate(masks, axis=0)
        return None, None
