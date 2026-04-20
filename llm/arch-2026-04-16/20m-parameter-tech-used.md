# FAISS HNSW Optimization Techniques — V1-16 Build

> Base: FAISS v1.14.1 (upstream main, April 2026)
> All optimizations are self-developed on top of the upstream codebase.

---

## Overview

Our optimized FAISS build (internally called **V1-16**) integrates 11 search/index-level optimizations plus 1 application-level optimization, for a total of **12 techniques**. These are categorized below by the area they target.

---

## 1. Threading & Parallelism

### O1 — OpenMP Conditional Guards

When a caller issues a single query (`n=1`), the upstream code still spawns the OpenMP thread pool, incurring 10–130 µs of thread-management overhead per call. We add `if(n > 1)` guards to `#pragma omp parallel` directives across six Index implementations, so that single-query requests bypass the thread pool entirely and execute on the calling thread.

**Files**: `IndexBinaryHNSW.cpp`, `IndexFlatCodes.cpp`, `IndexIDMap.cpp`, `IndexNNDescent.cpp`, `IndexNSG.cpp`, `IndexScalarQuantizer.cpp`

### O2 — Dynamic OMP Scheduling for HNSW Build

The upstream HNSW build loop uses `schedule(static)`, which distributes vertices evenly across threads. In practice, vertices inserted into higher HNSW layers are significantly more expensive because they require longer search paths. We switch to `schedule(dynamic, 64)` so that threads that finish their chunk early can steal work from the remaining queue, improving CPU utilization during index construction.

**Files**: `IndexHNSW.cpp`

---

## 2. SIMD Distance Computation

### O3 — AVX-512 / AVX2 Batch-8 Distance Kernels

The upstream HNSW search evaluates neighbor distances one at a time. We introduce `fvec_L2sqr_batch_8` and `fvec_inner_product_batch_8` hand-written SIMD kernels (AVX-512: 16 floats/iteration, AVX2: 8 floats/iteration) and upgrade the four hot paths in `HNSW.cpp` (`search_from_candidates`, `greedy_update_nearest`, etc.) from a buffer-of-4 to a buffer-of-8 design. This allows the CPU to process 8 distance computations per loop iteration using wide SIMD registers, amortizing loop overhead and improving instruction-level parallelism.

**Files**: `distances_simd.cpp`, `HNSW.cpp`, `IndexFlat.cpp`, `DistanceComputer.h`

### O4 — Software Prefetch Pipeline

Each HNSW neighbor visit requires loading a full vector from main memory (768 × 4 = 3 KB for FP32, or 1.5 KB for FP16). At 20 million vectors the dataset far exceeds any CPU cache. We add a `DistanceComputer::prefetch()` virtual method and insert prefetch instructions in the HNSW search hot paths so that while the CPU is computing distances for the current batch, it is simultaneously fetching the next batch's vector data into L1/L2 cache. This overlaps memory latency with computation and works synergistically with O3's batch-8 design.

**Files**: `DistanceComputer.h`, `HNSW.cpp`

### O5 — FP16 SIMD Distance Computation Library

We implement a standalone FP16/BF16 distance computation library with three-tier SIMD dispatch: AVX-512F+F16C, AVX2+F16C, and scalar fallback. The library provides a complete API surface (`fp16vec_L2sqr`, `fp16vec_inner_product`, `_batch_4`, `_batch_8`, etc.) and includes 117 unit tests. This library is used by the `SQfp16` storage backend to compute distances directly on half-precision data without upconverting to FP32.

**Files**: `distances_fp16.h` (new), `distances_fp16_simd.cpp` (new)

### O12 — SIMD-Accelerated MinimaxHeap::count_below

`MinimaxHeap::count_below()` is called in the HNSW search inner loop to decide whether to terminate early. The upstream implementation is a scalar loop over the heap array. We replace it with AVX-512 / AVX2 vectorized comparison and popcount, reducing the cost of this frequently called function.

**Files**: `HNSW.cpp`

---

## 3. Memory & Cache Optimization

### O10 — Transparent Huge Pages (THP)

HNSW search exhibits a random-access pattern across a large, contiguous vector storage array (30–70 GB at 20M scale). With the default 4 KB page size, TLB misses become a significant bottleneck. After index construction completes, we call `madvise(MADV_HUGEPAGE)` on the vector storage region to promote it to 2 MB huge pages, dramatically reducing TLB pressure.

**Files**: `IndexHNSW.cpp`, `numa_helpers.h` (new)

### O11 — Cross-Node Neighbor Batching

In the upstream `search_from_candidates`, the batch buffer for distance computation is reset at each candidate node boundary. If a node has fewer than 8 unvisited neighbors, the batch is flushed partially, losing the SIMD efficiency gained by O3. We make the batch buffer persistent across candidate nodes, so neighbor IDs from consecutive candidates accumulate into full batches of 8 before being dispatched to the SIMD kernel. This maximizes the batch-8 fill rate.

**Files**: `HNSW.cpp`

### O13 — SharedVectorStore Architecture

We introduce `SharedVectorStore`, a storage layer that decouples vector data ownership from the HNSW graph. It supports `storage_id_map` indirect addressing, a `free_list` for O(1) slot recycling, and a `deleted_bitmap` for logical deletion. This enables zero-copy HNSW rebuild: after deleting vectors, the graph can be reconstructed in-place via `add(n, nullptr)` without copying the remaining vector data.

**Files**: `SharedVectorStore.cpp/h` (new), `IndexFlatShared.cpp/h` (new), `IndexHNSW.cpp`

### O14 — BFS Graph Reorder for Cache Locality

HNSW nodes are stored in insertion order, which is essentially random with respect to the graph topology. During search, traversing graph edges causes random jumps across the adjacency list array, leading to poor cache utilization. After index construction, we apply a BFS (Breadth-First Search) traversal starting from the entry point and permute both the adjacency lists and the vector storage to match the BFS order. This ensures that nodes likely to be visited in sequence during search are stored in adjacent memory locations, significantly improving spatial locality and cache hit rates.

**Files**: `HNSWReorder.cpp/h` (new), `IndexFlatShared.cpp/h`

---

## 4. Search-Path Optimization

### O16 — VisitedTable Reuse

The upstream HNSW search allocates and frees a `VisitedTable` (a byte array sized to the number of vectors — 20 MB at 20M scale) on every single search call. This causes repeated `malloc`/`free` of a large allocation on every query, which is expensive and fragments the heap. We extend `SearchParametersHNSW` to accept a pre-allocated `VisitedTable` and reuse it across queries within the same thread, replacing the per-query allocation with a simple `memset` reset.

**Files**: `HNSW.h`, `HNSW.cpp`, `IndexHNSW.cpp`

---

## 5. Application-Level Optimization (TODO)

### O17 — gRPC Thread Pool Tuning

At the application layer (VDE gRPC server), the default gRPC configuration uses `MAX_POLLERS=2`, which limits the number of threads actively polling for incoming RPCs. Under high-concurrency search workloads, this becomes a bottleneck as requests queue up waiting for a poller thread. We increase `MAX_POLLERS` to 8 to match the expected concurrency level, reducing request queuing latency.

**Files**: `vde_grpc_server.cpp` (application-level, not in FAISS library)

---

## Summary

| # | Technique | Category | Scope |
|---|-----------|----------|-------|
| O1 | OpenMP conditional guards | Threading | Search |
| O2 | Dynamic OMP scheduling | Threading | Build |
| O3 | AVX-512/AVX2 batch-8 distance | SIMD | Search |
| O4 | Software prefetch pipeline | SIMD + Memory | Search |
| O5 | FP16 SIMD distance library | SIMD | Search (SQfp16) |
| O10 | Transparent huge pages | Memory | Search |
| O11 | Cross-node neighbor batching | Memory + SIMD | Search |
| O12 | SIMD count_below | SIMD | Search |
| O13 | SharedVectorStore | Memory | Build + Runtime |
| O14 | BFS graph reorder | Memory | Post-build |
| O16 | VisitedTable reuse | Search-path | Search |
| O17 | gRPC thread pool tuning | Application | Serving |
