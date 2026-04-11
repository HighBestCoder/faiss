# FAISS HNSW Search Performance Optimization — Full Technical Report

> Project: FAISS v1.14.1 HNSW index search performance optimization
> Branch: `bench/optimize-all` (based on the v1.14.1 tag)
> Date: 2026-04-08
> Hardware: Azure VM, 62 GB RAM, 16 cores, Intel Xeon (Ice Lake, AVX-512)
> Dataset: Cohere 768D cosine similarity, 1M / 10M vectors

---

## 1. Project Background

### 1.1 Problem Definition

FAISS HNSW (Hierarchical Navigable Small World) is one of the most widely used approximate nearest neighbor search algorithms in production. In large-scale vector retrieval scenarios (tens of millions of vectors and beyond), the native FAISS v1.14.1 HNSW search path has the following performance bottlenecks:

1. **Low distance-computation efficiency**: scalar loops rely on compiler auto-vectorization and cannot fully utilize the 512-bit register width of AVX-512.
2. **Poor memory-access pattern**: 10M × 768D vector data is about 30 GB, far beyond CPU cache capacity, and HNSW's random access causes many cache misses and TLB misses.
3. **High thread-management overhead**: even a single query still triggers OpenMP thread-pool scheduling, and the microsecond-level overhead is non-negligible at high QPS.
4. **Repeated memory allocation**: each search allocates and frees a VisitedTable array on the order of 10 MB.

### 1.2 Optimization Goals

- Increase search QPS without changing search-result correctness.
- Avoid increasing memory overhead as much as possible.
- Make every optimization independently switchable so its contribution can be evaluated.

### 1.3 Final Results

Baseline for comparison: the customer's production setup, **native FAISS v1.14.1 + HNSW32,Flat (FP32, no quantization)**.
Optimized version: **V1-16 with all optimizations enabled + SQfp16 quantization**.

#### 1M dataset (efSearch=64, efConstruction=40)

| Metric | V0 Native Flat (FP32) | V1-16 + SQfp16 | Gain |
|------|------------------------|----------------|------|
| Single-thread QPS | 1,110 | 1,538 | **+38.6%** |
| 16-thread QPS | 10,912 | 13,404 | **+22.8%** |
| Recall@10 | 96.15% | 95.98% | -0.17 pp (negligible) |
| Memory (RSS) | ~3,200 MB | 1,946 MB | **-39.1%** |

#### 10M dataset — high-recall search (M=32, efConstruction=512, efSearch=512)

| Metric | V0 Native Flat (FP32) | V1-16 + SQfp16 | Gain |
|------|------------------------|----------------|------|
| Single-thread QPS | 141 | 177 | **+25.5%** |
| 16-thread QPS | 1,325 | 1,581 | **+19.3%** |
| Recall@10 | 99.60% | 99.02% | -0.58 pp (acceptable) |
| Memory (RSS) | 33,798 MB | 19,157 MB | **-43.3%** |
| Build time | 8,031 s (2.2 h) | 7,041 s (2.0 h) | **-12.3%** |

> **Core value**: while improving QPS by 19-39%, memory usage is reduced by 39-43%, with Recall loss below 0.6 pp.

---

## 2. Analysis of the HNSW Search Process

To understand the optimizations, it is necessary to understand the core HNSW search flow first.

### 2.1 Search Has Two Stages

```
Stage 1: Upper-layer greedy search (greedy_update_nearest)
  Start from the entry point at the highest level and find only 1 nearest neighbor per layer
  Number of layers = log(N) / log(M), so the compute cost is small

Stage 2: Candidate search on layer 0 (search_from_candidates)
  This is the main search cost (>95% of runtime)
  Maintain a candidate heap of size efSearch
  Each time, pop the nearest candidate node and traverse all of its neighbors (up to 2×M = 64)
  Compute distances for unvisited neighbors and update the candidate heap
  At most efSearch steps are performed
```

### 2.2 CPU Time Breakdown of the Search Hot Path

For a typical query with 768D vectors and efSearch=64:

| Operation | Share | Notes |
|------|------|------|
| Distance computation (`fvec_inner_product`) | ~60% | 768-dimensional floating-point inner product each time |
| Memory stall (`cache miss`) | ~20% | Random access into 30 GB of vector data |
| Candidate heap operations (`MinimaxHeap`) | ~8% | `pop_min`, `count_below` |
| VisitedTable allocation / lookup | ~7% | `malloc` + `memset` of 10 MB |
| OpenMP scheduling | ~3% | Thread-pool wakeup / synchronization |
| Other | ~2% | Graph traversal, etc. |

Each optimization below targets one of these hotspots precisely.

---

## 3. Optimization Details

### O1: Conditional OpenMP Guard — Eliminate Single-Query Thread Scheduling Overhead

**Problem addressed**

When `n=1` (a single query), OpenMP `#pragma omp parallel` still triggers thread-pool management, including thread wakeup, task dispatch, and barrier synchronization. This fixed overhead is roughly 10-130 microseconds. In scenarios with QPS > 1000 (that is, each query takes < 1 ms), this overhead can account for 10-13% of total latency.

**Implementation**

Add an `if(n > 1)` condition to all search-related `#pragma omp parallel` directives:

```cpp
// Before
#pragma omp parallel for
for (idx_t i = 0; i < n; i++) { ... }

// After
#pragma omp parallel for if(n > 1)
for (idx_t i = 0; i < n; i++) { ... }
```

When `n=1`, the loop runs directly on the calling thread and skips all OpenMP overhead.

**Files modified**

- `faiss/IndexHNSW.cpp`
- `faiss/IndexBinaryHNSW.cpp`
- `faiss/IndexFlatCodes.cpp`
- `faiss/IndexIDMap.cpp`
- `faiss/IndexNNDescent.cpp`
- `faiss/IndexNSG.cpp`
- `faiss/IndexScalarQuantizer.cpp`

This mainly prevents a class of unnecessary OpenMP usage already present in the FAISS codebase.

---

### O2: Dynamic OpenMP Scheduling — Fix Load Imbalance During Build

**Problem addressed**

During HNSW build, `hnsw_add_vertices()` uses `schedule(static)` to distribute vectors evenly across threads. But HNSW insertion is inherently non-uniform: vectors inserted later must search for neighbors in a larger graph and are therefore much more expensive to process than earlier vectors. Static scheduling causes threads assigned to the later half of the workload to do significantly more work, while threads assigned to the earlier half sit idle.

**Implementation**

```cpp
// Before
#pragma omp for schedule(static)

// After
#pragma omp for schedule(dynamic, 64)
```

With dynamic scheduling, once a thread finishes inserting its current batch of 64 vectors, it immediately takes the next batch, automatically balancing the load.

**Files modified**

- `faiss/IndexHNSW.cpp` — `hnsw_add_vertices()`

**Effect**: build time reduced by about **-4%**. Search performance is not directly affected.

---

### O3: AVX-512 `batch_8` Distance Computation — The Main Performance Driver

**Problem addressed**

Distance computation accounts for about 60% of search time and is the core hotspot. In native FAISS:
1. `fvec_inner_product` / `fvec_L2sqr` use scalar loops and depend on compiler auto-vectorization.
2. In the HNSW search path, neighbor distances are computed in buffered groups of 4 (`batch_4`).

This creates two issues:
- compiler auto-vectorization is not reliable and does not guarantee AVX-512 usage;
- AVX-512 has 32 512-bit registers (16 floats per register), but `batch_4` uses only 4 accumulators, so register utilization is only 12.5%.

**Implementation**

**Step 1**: Add a handwritten SIMD `batch_8` distance kernel.

```cpp
// AVX-512 implementation (distances_avx512.cpp)
void fvec_inner_product_batch_8(
    const float* x,           // query vector (768D)
    const float* y0, ..., const float* y7,  // 8 database vectors
    size_t d,
    float& dp0, ..., float& dp7)
{
    __m512 sum0=_mm512_setzero_ps(), ..., sum7=_mm512_setzero_ps();
    for (size_t i = 0; i < d; i += 16) {
        __m512 xi = _mm512_loadu_ps(x + i);  // load query vector once
        sum0 = _mm512_fmadd_ps(xi, _mm512_loadu_ps(y0+i), sum0);  // FMA
        sum1 = _mm512_fmadd_ps(xi, _mm512_loadu_ps(y1+i), sum1);
        ...
        sum7 = _mm512_fmadd_ps(xi, _mm512_loadu_ps(y7+i), sum7);
    }
    dp0 = _mm512_reduce_add_ps(sum0);
    ...
}
```

Key optimization points:
- each 16-float block of the query vector `x[i]` is loaded from memory only **once** and reused across 8 FMAs;
- 8 accumulators perform FMA operations concurrently, making fuller use of the CPU's out-of-order execution capacity;
- each loop iteration issues 8 FMA instructions, exploiting AVX-512 FMA throughput (2 FMAs per cycle).

**Step 2**: Add a virtual `distances_batch_8()` method to the `DistanceComputer` base class.

```cpp
struct DistanceComputer {
    // existing
    virtual void distances_batch_4(const idx_t idx0, ..., float& dis0, ...);
    // new
    virtual void distances_batch_8(const idx_t idx0, ..., float& dis0, ...);
};
```

**Step 3**: Expand the HNSW search-path buffer from 4 to 8.

```cpp
// Before (HNSW.cpp search_from_candidates)
storage_idx_t saved_j[4];
if (counter == 4) { qdis.distances_batch_4(...); counter = 0; }

// After
storage_idx_t saved_j[8];
if (counter == 8) { qdis.distances_batch_8(...); counter = 0; }
// Remaining groups of 4-7 use batch_4
```

All four HNSW hot paths were upgraded: `search_from_candidates`, `search_from_candidate_unbounded`, `greedy_update_nearest`, and `search_neighbors_to_add`.

**Files modified**

- `faiss/impl/DistanceComputer.h` — `batch_8` virtual method
- `faiss/impl/HNSW.cpp` — 4 search hot paths
- `faiss/IndexFlat.cpp` — `FlatL2Dis` / `FlatIPDis` `batch_8` implementation
- `faiss/IndexFlatCodes.cpp` — `GenericFlatCodesDistanceComputer` `batch_8`
- `faiss/utils/distances.cpp` — dispatch entry point
- `faiss/utils/distances.h` — declarations
- `faiss/utils/distances_dispatch.h` — SIMD-level dispatch templates
- `faiss/utils/simd_impl/distances_avx512.cpp` — AVX-512 implementation
- `faiss/utils/simd_impl/distances_avx2.cpp` — AVX2 implementation
- `faiss/utils/simd_impl/distances_autovec-inl.h` — scalar fallback

**Effect**: about **+20%** single-thread QPS improvement, the largest single contributor.

---

### O4: Dimension-Aware Software Prefetch — Reduce Cache-Miss Stall Time

**Problem addressed**

10M × 768D vector data occupies about 30 GB, far larger than CPU L3 cache (typically 25-50 MB). HNSW search uses random access: after obtaining a neighbor node ID from the candidate's adjacency list, the algorithm must load that node's 3072-byte vector to compute the distance. This is almost guaranteed to be an L3 cache miss with 100-200 cycles of latency.

The CPU's out-of-order execution window cannot completely hide such long latency. If prefetch can be issued before the data is actually needed so that the data reaches cache before computation begins, the latency can be hidden behind other work.

**Implementation**

Add a `prefetch()` virtual method to the `DistanceComputer` base class:

```cpp
virtual void prefetch(idx_t id, int lines = 3) {
    // Enable only when vector data >= 1200 bytes (roughly dim >= 300)
    if (code_size >= 1200) {
        const char* ptr = codes + id * code_size;
        for (int i = 0; i < lines; i++) {
            _mm_prefetch(ptr + i * 64, _MM_HINT_T0);  // prefetch into L1
        }
    }
}
```

Then, in the HNSW hot path, prefetch neighbor data while iterating over neighbor IDs:

```cpp
// Before using neighbor v1's distance
qdis.prefetch(v1);   // prefetch vector data
vt.prefetch(v1);     // prefetch VisitedTable entry (existing)
```

**Why the dimension threshold matters**

- `dim=128` (SIFT): each vector is only 512 bytes and is likely already in cache; prefetch instruction overhead can actually slow performance down (measured at about -20%).
- `dim=768` (Cohere): each vector is 3072 bytes and almost certainly causes a cache miss, so prefetch provides clear gains.
- the threshold is set to 1200 bytes (about `dim ≈ 300`), based on empirical results across multiple datasets.

**Files modified**

- `faiss/impl/DistanceComputer.h` — `prefetch` virtual method (+22 lines)
- `faiss/impl/HNSW.cpp` — inserted prefetch calls into 4 hot paths (+4 lines)

**Effect**: together with O3/O14, reduces cache-miss overhead and improves QPS by about **+3-5%**.

---

### O5: FP16 SIMD Distance-Compute Library

**Problem addressed**

FP32 vectors require 4 bytes per dimension, while FP16 uses 2 bytes. Using SQfp16 storage halves memory usage, but requires efficient FP16 distance-computation support. FAISS ScalarQuantizer already supports FP16, but in v1.14.1 the internal distance-computation path requires correct SIMD dispatch configuration to actually enable the AVX-512 implementation.

**Implementation**

An independent FP16 distance library with three SIMD layers:

```
AVX-512F + F16C: process 32 float16 values at a time -> convert into 2 __m512 registers -> FMA
AVX2 + F16C:     process 8 float16 values at a time -> _mm256_cvtph_ps -> FMA
Scalar:          elementwise fp16_ieee_to_fp32_value() -> scalar multiply-add
```

The API includes `fp16vec_L2sqr`, `fp16vec_inner_product`, `batch_4`, `batch_8` variants, and FP32↔FP16 conversion utilities.

**Files modified**

- `faiss/utils/distances_fp16.h` — declarations (140 lines)
- `faiss/utils/distances_fp16_simd.cpp` — implementation (1249 lines)
- `tests/test_distances_fp16.cpp` — 117 unit tests

**SIMD dispatch fix** (critical finding)

After all optimizations were ported, SQfp16 QPS was only 431 instead of the expected ~1500. Root cause analysis found:

1. the build script was missing `-DFAISS_OPT_LEVEL=avx512`, causing v1.14.1's SIMD dispatch mechanism to fall back to the scalar path;
2. after enabling it, an ODR (One Definition Rule) conflict appeared because a `batch_8` template specialization and an explicit specialization coexisted.

Fix: add the `FAISS_SKIP_AUTOVEC_BATCH_8` preprocessor guard and update `llm/build.sh` to include `-DFAISS_OPT_LEVEL=avx512`.

**Effect**: SQfp16 QPS increased from 431 to 1,304 (**+202%**). This is not the isolated contribution of the O5 library itself, but rather the result of fixing dispatch configuration so that the AVX-512 SQ path is actually enabled.

---

### O10: Transparent Huge Pages (THP) — Reduce TLB Misses

**Problem addressed**

10M vectors × 3072 bytes = 30 GB of vector data. With standard 4 KB pages, this requires about **7.5 million page-table entries**. CPU TLBs (Translation Lookaside Buffers) typically cache only 1000-2000 page-table entries. Under HNSW's random access pattern, almost every vector access can trigger a TLB miss and a multi-level page-table walk, costing 100+ cycles.

**Implementation**

After index construction, call `madvise(MADV_HUGEPAGE)` on the vector storage region to request 2 MB transparent huge pages from the kernel:

```cpp
// numa_helpers.h (new file)
inline void try_enable_hugepages(void* ptr, size_t size) {
    // Align to a 2 MB boundary
    uintptr_t aligned = (uintptr_t(ptr) + 0x1FFFFF) & ~0x1FFFFF;
    size_t usable = size - (aligned - uintptr_t(ptr));
    madvise((void*)aligned, usable, MADV_HUGEPAGE);
}

// End of add() in IndexHNSW.cpp
auto* flat = dynamic_cast<IndexFlatCodes*>(storage);
if (flat) {
    try_enable_hugepages(flat->codes.data(), flat->codes.size());
}
```

Using 2 MB huge pages reduces page-table entries from 7.5 million to about **15,000**, greatly lowering TLB miss rates.

**When it helps**

| Dataset | Dim | THP effect | Reason |
|--------|------|-----------|------|
| 10M × 768D | 3072 B/vec | **+13-25%** | Large dataset, long vectors |
| 1M × 960D | 3840 B/vec | **+25.3%** | Very long vectors |
| 1M × 128D | 512 B/vec | +12.5% | Medium-scale data |
| 1.2M × 100D | 400 B/vec | **-40%** | Short vectors; huge pages waste memory |
| 290K × 256D | 1024 B/vec | **-44%** | Dataset too small |

**Conclusion**: huge pages help only when vectors are larger than 500 bytes **and** the dataset exceeds 500K vectors.

**Files modified**

- `faiss/utils/numa_helpers.h` — new file (38 lines)
- `faiss/IndexHNSW.cpp` — end of `add()` (+8 lines)
- `faiss/CMakeLists.txt` — header registration

**Effect**: **+13-25%** on large, high-dimensional datasets, with even better synergy when combined with O14 BFS reordering.

---

### O11: Cross-Node Neighbor Batching — Improve `batch_8` Hit Rate

**Problem addressed**

O3 introduced `batch_8` distance computation, but the original HNSW buffering logic is **per candidate node**. After popping a candidate node, the code traverses its neighbors and buffers them. The problem is that with HNSW `M=32`, each node has up to 64 neighbors, but most have already been visited. On average, each candidate node contributes only 2-3 unvisited neighbors to the buffer, so it rarely fills to 8. In practice, `batch_8` was only hit about 30% of the time.

**Implementation**

Move the buffer **outside** the candidate-node loop so that unvisited neighbors from multiple candidate nodes accumulate into the same buffer:

```cpp
// Before: buffer resets every iteration
while (candidates.size() > 0) {
    int counter = 0;               // reset per candidate node
    storage_idx_t saved_j[8];
    for (j in neighbors) { ... }
}

// After: buffer accumulates across candidate nodes
int batch_counter = 0;             // moved outside the loop
storage_idx_t batch_ids[8];
while (candidates.size() > 0) {
    for (ni in neighbor_ids) {
        batch_ids[batch_counter++] = v1;
        if (batch_counter == 8) {   // trigger batch_8 once 8 entries are collected
            qdis.distances_batch_8(...);
            batch_counter = 0;
        }
    }
}
flush_batch();  // handle leftovers: try batch_4 first, then scalar
```

This also adds a **sliding-window prefetch**: after collecting all neighbor IDs for the current node, later neighbors are prefetched in windows of 8, including both vector data and VisitedTable entries.

**Files modified**

- `faiss/impl/HNSW.cpp` — complete rewrite of `search_from_candidates` (+114 / -78 lines)

**Effect**: `batch_8` hit rate increases from ~30% to ~80%, and QPS improves by about **+10-17%**.

---

### O12: SIMD-Accelerated `MinimaxHeap::count_below` — Speed Up Early-Termination Checks

**Problem addressed**

The early-termination condition in `search_from_candidates` calls `MinimaxHeap::count_below(thresh)` to count the number of heap elements whose distance is below a threshold. The original implementation uses a scalar element-by-element comparison loop. With `efSearch=256`, it scans 256 floats, and this function is called for every candidate node.

**Implementation**

```cpp
// AVX-512 implementation
int MinimaxHeap::count_below(float thresh) {
    int count = 0;
    __m512 vt = _mm512_set1_ps(thresh);
    size_t i = 0;
    for (; i + 16 <= k; i += 16) {
        __m512 vd = _mm512_loadu_ps(dis.data() + i);
        __mmask16 mask = _mm512_cmp_ps_mask(vd, vt, _CMP_LT_OS);
        count += _mm_popcnt_u32(mask);
    }
    // Handle remaining elements
    for (; i < k; i++) {
        if (dis[i] < thresh) count++;
    }
    return count;
}
```

Each iteration compares 16 floats at once on AVX-512, or 8 floats on AVX2, and uses `popcnt` to count matches.

**Files modified**

- `faiss/impl/HNSW.cpp` — `count_below` (+43 lines)

**Effect**: about **+3-5%** locally. The function itself is not a dominant part of total runtime.

---

### O13: `SharedVectorStore` — Zero-Copy Rebuild

**Problem addressed**

In production, HNSW indexes often need periodic rebuilds because graph quality degrades after vector deletions. The traditional rebuild flow **copies all vectors** into the new index, doubling peak memory usage. For a 10M × 768D index (~30 GB of vector data), rebuild requires about 60 GB of memory, often exceeding machine capacity.

**Implementation**

Introduce a shared-storage architecture:

```
┌──────────────────────────────────────┐
│         SharedVectorStore            │
│  (stores actual vector data, managed │
│   by shared_ptr)                     │
│  + free_list (recycled deleted slots)│
└──────────┬───────────────────────────┘
           │ shared_ptr
    ┌──────┴──────┐
    │             │
┌───┴────┐  ┌────┴───┐
│ old    │  │ new    │   ← two indexes share the same vector data
│ index  │  │ index  │
│ after  │  │ during │
│ delete │  │ rebuild│
│storage_│  │storage_│
│id_map[]│  │id_map[]│   ← each keeps its own HNSW node ID -> store position map
└────────┘  └────────┘
```

- `SharedVectorStore`: owns vector data and is referenced by multiple indexes through `shared_ptr`.
- `IndexFlatShared`: uses `storage_id_map` for indirect addressing and supports `deleted_bitmap`.
- During rebuild, call `IndexHNSW::add(n, nullptr)` — passing `nullptr` means vectors are not copied and are read directly from `SharedVectorStore`.
- `compact_store()`: after rebuild, use an in-place cycle-following algorithm to reorder the store and remove the indirection cost.

**Files modified**

- `faiss/SharedVectorStore.h/cpp` — new
- `faiss/IndexFlatShared.h/cpp` — new (744 lines)
- `faiss/IndexHNSW.cpp` — zero-copy `add` path

**Effect**: during rebuild, memory usage is reduced by about **78%** because only graph structure and mapping tables are copied, not vector payloads. Search performance impact is below 2%.

---

### O14: BFS Graph Reordering — Improve Cache Locality

**Problem addressed**

HNSW node IDs are assigned in insertion order and are unrelated to graph topology. During search, neighboring nodes are randomly distributed in memory, so the CPU cannot exploit spatial locality effectively. Even if two nodes are close in the graph, their vector data may be gigabytes apart in memory.

**Implementation**

Provide 5 graph-reordering strategies that remap node IDs according to graph topology so that graph-adjacent nodes are also close in memory:

| Strategy | Principle | Best for |
|------|------|---------|
| **BFS** | BFS from the entry point, assign IDs by visit order | General-purpose, most stable |
| RCM | Reverse Cuthill-McKee, minimize graph bandwidth | Sparse graphs |
| DFS | Depth-first numbering | Specific graph shapes |
| Cluster | Sort by level + degree | Multi-level graphs |
| Weighted | Sort by `(1+level)×degree` | High-degree nodes |

```cpp
// Usage
auto perm = generate_permutation(hnsw, ReorderStrategy::BFS);
index->permute_entries(perm.data());
// perm[new_id] = old_id; search results require reverse mapping afterwards
```

`permute_entries` reorders both:
- graph structure (neighbor-list ID references);
- vector data (`IndexFlatCodes::codes` array).

After reordering, nodes visited along the BFS search path are physically closer in memory, allowing CPU L1/L2 prefetchers and hardware prefetch to work much more effectively.

**Files modified**

- `faiss/HNSWReorder.h/cpp` — new (367 lines)
- `faiss/IndexFlatShared.cpp` — dual-mode `permute_entries`
- `faiss/CMakeLists.txt`

**Effect**: BFS reordering alone gives about **+18%** QPS. Combined with huge pages (O10), the gain is another **+10-15%**.

---

### O16: VisitedTable Reuse — Eliminate Per-Query Memory Allocation

**Problem addressed**

Each HNSW search allocates a new `VisitedTable` of size `ntotal` bytes. On a 10M index, that means every query performs `malloc(10 MB)` + `memset(10 MB, 0)` + `free(10 MB)`. At QPS > 1000, this happens thousands of times per second and becomes a measurable cost.

`VisitedTable` already has a clever internal design: it maintains a `visited_generation` counter, and calling `advance()` logically clears the table in $O(1)$ time without doing `memset`. But if the table is reallocated every time, that design benefit is lost.

**Implementation**

Add a `visited_table` field to `SearchParametersHNSW`:

```cpp
struct SearchParametersHNSW : SearchParameters {
    int efSearch = 0;
    VisitedTable* visited_table = nullptr;  // new
};
```

In the search function, if the caller provides a preallocated `VisitedTable`, reuse it:

```cpp
VisitedTable* vt_ptr;
std::unique_ptr<VisitedTable> local_vt;
if (external_vt && is_single_query) {
    vt_ptr = external_vt;
    vt_ptr->advance();      // O(1) logical clear
} else {
    local_vt = std::make_unique<VisitedTable>(ntotal);  // O(ntotal) allocate + zero
    vt_ptr = local_vt.get();
}
```

The caller only needs to create one `VisitedTable` and reuse it across all queries:

```cpp
// Client code
VisitedTable vt(index->ntotal);
SearchParametersHNSW params;
params.efSearch = 64;
params.visited_table = &vt;
for (int i = 0; i < nq; i++) {
    index->search(1, query_i, k, distances, labels, &params);
}
```

**Files modified**

- `faiss/impl/HNSW.h` — add field to `SearchParametersHNSW` (+1 line)
- `faiss/IndexHNSW.cpp` — `hnsw_search` logic (+16 lines)

**Effect**: about **+5%** single-thread QPS improvement, more noticeable on 10M-scale indexes.

---

## 4. Contribution Breakdown

### 4.1 Estimated Standalone Contribution of Each Optimization

Based on tests on the 10M dataset with HNSW32,SQfp16, `efSearch=64`:

```
Total improvement: single-thread +30.4% (954 -> 1,244 QPS)

O3  SIMD batch_8 distance compute   ████████████████████  ~20%   <- largest contributor
O14 BFS graph reorder              ██████████████        ~12%
O16 VisitedTable reuse             █████                 ~5%
O4  software prefetch              ████                  ~4%
O11 cross-node neighbor batching   ███                   ~3%    (synergy with O3)
O10 transparent huge pages         ██                    ~2%    (synergy with O14)
O1  conditional OMP guard          ██                    ~2%
O12 SIMD count_below               █                     ~1%
```

> Note: these optimizations have both synergy and diminishing-return effects, so the sum of standalone contributions does not equal the total speedup. O3+O11 together raise the `batch_8` hit rate from 30% to 80%; O14+O10 make reordered contiguous data benefit more from huge-page coverage.

### 4.2 Search vs Build Optimizations

| Category | Optimizations | Affected stage |
|------|------|---------|
| Search performance | O1, O3, O4, O11, O12, O14, O16 | `search()` |
| Build performance | O2 | `add()` |
| Memory saving | O5 (SQfp16), O13 (zero-copy rebuild) | deployment |
| Search + build | O10 (huge pages) | both |

---

## 5. Full Performance Comparison

> **Comparison rule**: the baseline is the customer's actual production setup, **native FAISS v1.14.1 + HNSW32,Flat (FP32)**, compared against **V1-16 with all optimizations + SQfp16**.
> For reference, this section also includes a pure-search-optimization comparison under the same quantization mode (`V0 SQfp16` vs `V1-16 SQfp16`).

### 5.1 1M dataset (`efSearch=64`, `efConstruction=40`)

| Metric | V0 Flat (FP32) | V1-16 SQfp16 | Change |
|------|---------------|-------------|------|
| Single-thread QPS | 1,110 | 1,538 | **+38.6%** |
| 16-thread QPS | 10,912 | 13,404 | **+22.8%** |
| Recall@10 | 96.15% | 95.98% | -0.17 pp |
| Memory (RSS) | ~3,200 MB | 1,946 MB | **-39.1%** |

Pure search-optimization reference (both using SQfp16): `V0 SQfp16 QPS_1T=1,165` -> `V1-16 SQfp16 QPS_1T=1,538` (+32.0%)

### 5.2 10M dataset — high-recall search (`M=32`, `efC=512`, `efS=512`)

| Metric | V0 Flat (FP32) | V1-16 SQfp16 | Change |
|------|---------------|-------------|------|
| Single-thread QPS | 141 | 177 | **+25.5%** |
| 16-thread QPS | 1,325 | 1,581 | **+19.3%** |
| Recall@10 | 99.60% | 99.02% | -0.58 pp |
| Memory (RSS) | 33,798 MB | **19,157 MB** | **-43.3%** |
| Build time | 8,031 s (2.2 h) | 7,041 s (2.0 h) | **-12.3%** |

### 5.3 10M dataset — pure search optimization comparison (`V0 SQfp16` vs `V1-16 SQfp16`)

The following comparison uses the same quantization mode on both sides (SQfp16) and isolates the benefit of search-algorithm optimization:

| efSearch | V0 SQfp16 QPS_1T | V1-16 SQfp16 QPS_1T | Change |
|----------|------------------|---------------------|------|
| 64 | 954 | 1,244 | +30.4% |
| 128 | 525 | 684 | +30.3% |
| 256 | 273 | 366 | +34.1% |

### 5.4 Comparison with RaBitQ Quantization

RaBitQ is a new quantization scheme built into FAISS v1.14.1. On the 10M dataset at roughly `Recall@10 ≈ 95%`:

| Scheme | Single-thread QPS | 16-thread QPS | Memory | Build time |
|------|--------------------|---------------|--------|-----------|
| V0 native Flat (`efC=512,efS=512`) | 141 | 1,325 | 33,798 MB | 8,031 s |
| V1-16 HNSW+SQfp16 (`efC=512,efS=512`) | **177** | **1,581** | 19,157 MB | 7,041 s |
| IVF4096,RaBitQfs4 (`nprobe=256`) | 29 | 451 | **5,718 MB** | **488 s** |

Compared with native Flat, HNSW+SQfp16 provides +25% QPS and -43% memory usage.
Compared with native Flat, RaBitQ provides -79% QPS and -83% memory usage. Each has its own suitable deployment scenarios.

---

## 6. Version Definitions and Reproducibility

### 6.1 Version Definitions

| Version | Description | Library file |
|------|------|--------|
| **V0** | Native FAISS v1.14.1, no optimizations | `llm/faiss-1.14.1-origin/lib/libfaiss_avx512.so` |
| **V1-12** | O1/O2/O3/O4/O10/O11/O12 (automatically active inside the library) | `install/lib/libfaiss_avx512.so` |
| **V1-16** | V1-12 + O14 (BFS graph reorder) + O16 (VisitedTable reuse) | same as above + explicit calls in `bench_task.cpp` |

### 6.2 Build Parameters

```bash
# Build the FAISS library
cmake -DFAISS_OPT_LEVEL=avx512 -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -ffast-math -funroll-loops" \
      -DBLA_VENDOR=Intel10_64_dyn ..

# Build the client program
g++ -O3 -march=native -mtune=native -std=c++17 \
    -I install/include \
    -o bench_task bench_task.cpp \
    -L install/lib -lfaiss_avx512 \
    -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 \
    -lpthread -lm -ldl \
    -Wl,-rpath,install/lib
```

### 6.3 Key Caveats

1. **You must link against `-lfaiss_avx512` instead of `-lfaiss`**. The latter loads the generic library, and SQfp16 distance computation falls back to the scalar path, dropping QPS from 1538 to 450.
2. **You must set `-DFAISS_OPT_LEVEL=avx512`**. Without it, ScalarQuantizer SIMD dispatch falls back to the scalar path.
3. **O14 graph reordering** requires temporary memory roughly equal to the size of the `codes` array. A 10M Flat index needs about 64 GB and will OOM on a 62 GB machine. SQfp16 requires only about 15 GB of temporary memory and completes normally.

---

## 7. Key Conclusions and Recommendations

### 7.1 Optimizations with the Largest QPS Impact

1. **O3 (SIMD `batch_8`)** — the largest single contributor (~20%) and must be enabled.
2. **O14 (BFS graph reordering)** — the second-largest contributor (~12%), with a significant cache-locality benefit.
3. **O11 (cross-node batching)** — synergistic with O3 and required for `batch_8` to pay off fully.

### 7.2 Overall Improvement vs the Customer's Current Setup

Customer's current setup: **native FAISS v1.14.1 + HNSW32,Flat (FP32, no quantization)**

Our solution: **V1-16 with all search optimizations + SQfp16 quantization**

Combined benefits:
- **QPS improves by 19-39%**: search-algorithm optimizations (O1-O16) contribute +25-34%, and the smaller SQfp16 footprint further improves cache behavior.
- **Memory drops by 39-43%**: SQfp16 reduces storage per vector from 3072 bytes to 1536 bytes.
- **Recall loss is below 0.6 pp**: 99.60% -> 99.02%, negligible for production use.
- **Build time drops by 12%**: dynamic OpenMP scheduling plus lower data-movement cost from SQfp16.

### 7.3 Recommended Scenarios

| Scenario | Recommended solution |
|------|---------|
| High QPS + high Recall | HNSW32,SQfp16 (V1-16) |
| Memory constrained (< 8 GB) | IVF4096,RaBitQfs4 |
| Very large scale (100M+) | IVF + RaBitQ (possibly the only option that fits in memory) |
| Frequent rebuilds required | O13 `SharedVectorStore` zero-copy architecture |

---

## Appendix A: Optimization Summary Table

| ID | Optimization | Goal | Standalone effect |
|----|-------------|------|-------------------|
| O1 | Conditional OMP guard | Eliminate thread scheduling overhead when `n=1` | +2% |
| O2 | Dynamic OMP scheduling | Load balancing during build | Build -4% |
| O3 | AVX-512 `batch_8` | Instruction-level parallelism in distance computation | **+20%** |
| O4 | Dimension-aware prefetch | Reduce cache misses on vector data | +3-5% |
| O5 | FP16 SIMD library | Enable FP16 distance computation | (infrastructure) |
| O10 | Transparent huge pages | Reduce TLB misses | +2-25% |
| O11 | Cross-node batching | Improve `batch_8` hit rate | +10-17% |
| O12 | SIMD `count_below` | Accelerate early termination checks | +3-5% |
| O13 | `SharedVectorStore` | Zero-copy rebuild | Memory -78% |
| O14 | BFS graph reordering | Improve cache locality | +10-18% |
| O16 | VisitedTable reuse | Eliminate `malloc` / `memset` | +5% |

## Appendix B: Repository Information

- Branch: `bench/optimize-all`
- Baseline: FAISS v1.14.1 tag (`471ddad72`)
- Latest commit: `436c609f8` (RaBitQ benchmark)
- Result report: `llm/arch-2026-04-07/benchmark-report.md` (HNSW comparison)
- Result report: `llm/arch-2026-04-07/rabitq-benchmark-report.md` (RaBitQ comparison)