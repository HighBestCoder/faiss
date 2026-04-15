# HNSW Optimization Techniques Overview

> Source: `QPS分析-2026-04-02.md`
> Scenario: IndexHNSWFlat, 768D, float32, M=32

---

| ID | Name | What It Does | Affected Stage |
|----|------|-------------|----------------|
| A1 | AVX2 handwritten `fvec_L2sqr_batch_4` | Replace per-neighbor distance computation in HNSW traversal with a batched version that computes L2 distances for 4 neighbors at once using handwritten AVX2 intrinsics | search |
| A2 | AVX-512 Batch 8/16 extension | Extend batch_4 to compute 8 or 16 neighbor distances at once, utilizing the full width of AVX-512 512-bit registers | search |
| A3 | Vector data prefetch | While computing the current neighbor's distance, prefetch the next batch of neighbor vector data into L2 cache | search |
| A4 | OpenMP dynamic scheduling | Change the OpenMP scheduling strategy in HNSW build from static to dynamic, reducing load imbalance across threads | build |
| A5 | FP16 storage + SIMD distance computation | Store vectors as float16 instead of float32; use F16C instructions for distance computation. Halves memory usage and improves cache hit rate | search + storage |
| A6 | Spatial locality sort | After building, reorder vector storage so that graph-adjacent nodes are also adjacent in memory | post-build |
| A7 | Aggressive early termination | Exit the search/build loop early when the candidate heap quality is good enough, reducing unnecessary distance computations | build / search |
| A8 | Neighbor distance cache | Cache previously computed neighbor distances to avoid redundant calculations | build |
| A9 | Symmetric distance reuse | Exploit the symmetry of L2 distance d(a,b)=d(b,a) by caching and reusing previously computed distance pairs | build |
| A10 | NUMA-aware memory allocation | Ensure vector data and graph structures are allocated on the same NUMA node as the query thread, avoiding cross-NUMA access | deployment |
| A11 | OpenMP conditional guard | Skip OpenMP thread pool scheduling when search batch size=1, avoiding the fixed overhead of entering a parallel region for a single query | search |
| B1 | VisitedTable thread-local reuse | Replace per-search malloc+memset+free of the VisitedTable with thread_local reuse, eliminating large memory allocation on the hot path | search |
| B2 | gRPC thread pool expansion | Increase gRPC service concurrency threads from 2 to 16+, allowing multiple search requests to execute in true parallelism | deployment |
| B3 | OMP thread pool governance | Set OMP_NUM_THREADS=1 and KMP_BLOCKTIME=0 to prevent OMP threads from spin-waiting in single-query scenarios | deployment |
| B4 | Remove IndexIDMap2 rev_map | If remove() is not needed, replace IndexIDMap2 with IndexIDMap to eliminate the unordered_map reverse mapping and its memory overhead | deployment |
| B5 | Eliminate dynamic_cast | Replace 3 dynamic_cast calls on the search path (used to set efSearch) with direct field writes on the object | search |
| C1 | Graph Reorder — BFS | Renumber HNSW graph nodes in BFS order so that nodes visited consecutively during search are also contiguous in memory | post-build |
| C2 | Graph Reorder — RCM | Reorder nodes using the Reverse Cuthill-McKee algorithm to minimize the ID gap between adjacent nodes, improving cache line utilization | post-build |
| C3 | Graph Reorder — Weighted | Reorder nodes weighted by access frequency, placing frequently visited nodes close together | post-build |
| C4 | HugePages (2 MB pages) | Use 2 MB huge pages instead of 4 KB pages to significantly reduce TLB misses | deployment |
| C5 | THP (Transparent Huge Pages) | Let the kernel automatically coalesce 4 KB pages into 2 MB transparent huge pages via madvise, without pre-allocation | deployment |
| D1 | SharedVectorStore zero-copy rebuild | During index rebuild, the old and new indexes share the same vector memory; only the graph structure is rebuilt, vector data is not copied | rebuild |
| D2 | DPDK-style 4-way batch prefetch | In indirect-addressing distance computation, issue prefetch for 4 vectors simultaneously, following DPDK's pipeline processing pattern | search |
| D3 | GraphReorder + Storage Compact pipeline | First compact storage to eliminate holes left by deletions, then apply graph reordering, forming a complete optimization pipeline | post-build |
