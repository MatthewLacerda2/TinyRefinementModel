import numpy as np
import jax.numpy as jnp
import fsspec
from train_local import MAX_SEQ_LEN

class TextDataGenerator:
    def __init__(self, directory, max_seq_len=MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len
        self.directory = directory
        
        self.fs, self.path_prefix = fsspec.core.url_to_fs(directory)
        
        all_files = self.fs.ls(directory)
        self.files = sorted([f for f in all_files if f.endswith('.npy')])
        
        self.current_file_idx = 0
        self.data = None
        self.pointer = 0
        self.exhausted = False
        self.skip_count = 0
        self.is_new_file = False

    def _load_next_file(self):
        if self.current_file_idx >= len(self.files):
            self.exhausted = True
            return False
        
        file_path = self.files[self.current_file_idx]
        print(f"📖 Streaming {file_path} into VRAM...")
        
        with self.fs.open(file_path, 'rb') as f:
            self.data = np.load(f)
            
        self.pointer = 0
        
        if self.skip_count > 0:
            # Skip logic must account for the doubled sequence length
            stride = 2 * self.max_seq_len + 1
            tokens_to_skip = self.skip_count * stride
            if tokens_to_skip < len(self.data):
                self.pointer = tokens_to_skip
                self.skip_count = 0
            else:
                self.skip_count -= (len(self.data) // stride)
                self.current_file_idx += 1
                return self._load_next_file()
                
        self.current_file_idx += 1
        self.is_new_file = True
        return True

    def get_batch(self, batch_size):
        if self.exhausted: return None, None
        
        stride = 2 * self.max_seq_len + 1
        total_tokens = batch_size * stride
        
        if self.data is None or self.pointer + total_tokens > len(self.data):
            if not self._load_next_file():
                return None, None
            if self.exhausted or self.pointer + total_tokens > len(self.data):
                return self.get_batch(batch_size)

        batch = self.data[self.pointer : self.pointer + total_tokens]
        self.pointer += total_tokens
        
        reset_mask = np.zeros((batch_size,), dtype=bool)
        if self.is_new_file:
            reset_mask[:] = True
            self.is_new_file = False
            
        return jnp.array(batch.reshape(batch_size, stride), dtype=jnp.int32), jnp.array(reset_mask)

class DataMixer:
    def __init__(self, sources, weights):
        self.sources = list(sources)
        self.weights = list(weights)
        
    def get_batch(self, batch_size):
        while len(self.sources) > 0:
            counts = np.random.multinomial(batch_size, self.weights)
            batch_list = []
            exhausted_indices = []
            
            for i, (source, count) in enumerate(zip(self.sources, counts)):
                if count > 0:
                    res = source.get_batch(count)
                    if res is None or getattr(source, "exhausted", False):
                        exhausted_indices.append(i)
                    else:
                        batch_list.append(res)
            
            if exhausted_indices:
                new_sources, new_weights = [], []
                for i, (s, w) in enumerate(zip(self.sources, self.weights)):
                    if i not in exhausted_indices:
                        new_sources.append(s); new_weights.append(w)
                self.sources = new_sources
                if not self.sources: return None, None
                total_w = sum(new_weights)
                self.weights = [w / total_w for w in new_weights]
                continue
                
            if batch_list:
                batches, masks = zip(*batch_list)
                return jnp.concatenate(batches, axis=0), jnp.concatenate(masks, axis=0)
        return None, None
