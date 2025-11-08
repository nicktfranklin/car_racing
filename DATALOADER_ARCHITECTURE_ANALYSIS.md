# DataLoader Architecture Analysis

## Current Architecture (INEFFICIENT)

```
Dataset: ImageDataset with 200 chunk files
├─ image_indices: [(file_idx, ep_idx, frame_idx), ...]  # All 20M images
│
Sampler: RandomSampler (NO SHARDING)
├─ Generates random indices from 0 to 20M
├─ All workers see the SAME sampler output
│
Workers: 8 parallel processes
├─ Worker 0: Gets indices [5, 199, 44, 177, ...]  # Random across ALL files
├─ Worker 1: Gets indices [88, 3, 156, 92, ...]   # Random across ALL files
├─ Worker 2: Gets indices [123, 67, 8, 188, ...] # Random across ALL files
└─ ...

Result: Every worker needs access to ALL 200 files!
```

**Memory usage:**
- Each worker opens files on-demand as random indices request them
- With persistent workers, all 200 files eventually get opened and cached
- 8 workers × 200 files = RAM explosion

## The Problem

**RandomSampler does NOT shard data across workers by default!**

From PyTorch docs:
```python
# RandomSampler behavior with num_workers > 1:
# - ALL workers receive batches from the same shuffled index pool
# - Worker assignment is round-robin by BATCH, not by data partition
# - Each worker can access ANY sample in the dataset
```

This means:
- Worker 0 might need image from file 5
- Next batch, Worker 0 might need image from file 199
- Worker must keep both files open OR constantly open/close
- With 200 files, this is a disaster!

## Better Architecture Options

### Option 1: Shard Files Across Workers (BEST)

Partition the 200 files among workers:

```python
class ShardedImageDataset:
    def __init__(self, chunk_files, worker_id, num_workers):
        # Each worker gets a subset of files
        my_files = chunk_files[worker_id::num_workers]
        # Worker 0: files [0, 8, 16, 24, ...]  (25 files)
        # Worker 1: files [1, 9, 17, 25, ...]  (25 files)
        # ...

        # Only build index for MY files
        self.image_indices = build_index(my_files)
```

**Benefits:**
- Each worker: 200/8 = 25 files max
- Can cache all 25 files: 25 × 16MB = 400MB per worker
- 8 workers × 400MB = 3.2GB total (vs 25.6GB)
- No random file thrashing

**Trade-off:**
- Slightly less random (samples from file 0 and file 8 never in same batch)
- But batches are still very diverse

### Option 2: Memory-Mapped Dataset (GOOD)

Load ALL data into a single memory-mapped file:

```python
# One-time preprocessing: Consolidate 200 files into 1
consolidate_chunks_to_mmap()

# At training time:
class MmapDataset:
    def __init__(self):
        self.data = np.memmap('data.mmap', mode='r', shape=(20M, 64, 64, 3))

    def __getitem__(self, idx):
        return self.data[idx]  # Zero copy, instant access
```

**Benefits:**
- Zero file opening overhead
- OS manages memory automatically (pages in/out as needed)
- One file handle total
- Very fast random access

**Trade-off:**
- Requires preprocessing step
- Single large file (but that's actually better)

### Option 3: Pre-load Chunks to RAM (SIMPLE)

Just load several chunks into RAM at a time:

```python
class ChunkedDataset:
    def __init__(self, chunk_files):
        self.all_chunks = chunk_files
        self.active_chunks = []
        self.chunk_cache_size = 10  # Keep 10 chunks in RAM

    def __getitem__(self, idx):
        chunk_idx = idx // samples_per_chunk
        if chunk_idx not in self.active_chunks:
            self._swap_chunk(chunk_idx)
        return self.chunks[chunk_idx][local_idx]
```

**Benefits:**
- Simple to implement
- Predictable memory usage
- Good balance of speed and RAM

**Trade-off:**
- Some samples slower when swapping chunks

## Current Implementation Issues

**In `ImageDataset._build_lazy_index()`:**
```python
# Builds index over ALL 200 files
for file_idx, chunk_file in enumerate(self.chunk_files):
    self.image_indices.append((file_idx, ep_idx, frame_idx))
```

**In `__getitem__()`:**
```python
file_idx, ep_idx, frame_idx = self.image_indices[idx]
f = self._get_file_handle(file_idx)  # Can be ANY of 200 files!
```

**With RandomSampler:**
- idx can be any value from 0 to 20M
- file_idx can be any value from 0 to 200
- Each worker sees random file_idx values
- **No locality, no caching benefit**

## Recommendation

Implement **Option 1: Shard Files Across Workers**

Why:
1. Minimal code changes (add worker sharding logic)
2. 8x reduction in files per worker (200 → 25)
3. Can cache all files per worker comfortably
4. Maintains randomness at batch level
5. No preprocessing required

Implementation:
- Add `worker_init_fn` to DataLoader
- Modify `_build_lazy_index()` to only index worker's files
- Each worker independently samples from its file subset

Expected RAM: **~3-4GB total** (vs current 40GB)
Expected GPU util: **85-95%** (no change)
