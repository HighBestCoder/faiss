# SIFT-1M Benchmark Results

## Dataset Information

| Property | Value |
|----------|-------|
| Dataset | SIFT-1M |
| Dimensions | 128 |
| Training Vectors | 1,000,000 |
| Query Vectors | 10,000 |
| Distance Metric | L2 (Euclidean) |
| Source | [ann-benchmarks.com](http://ann-benchmarks.com/) |

## Index Parameters

| Parameter | Value |
|-----------|-------|
| M (connections per layer) | 16 |
| efConstruction | 100 |
| efSearch | 50, 100 |

## Benchmark Results

### Full Comparison Table

| Index Type | ef=50 QPS | ef=100 QPS | Recall@10 | vs Baseline |
|------------|-----------|------------|-----------|-------------|
| **Baseline IndexHNSWFlat** | 23,058 | 13,342 | 0.9404 | - |
| CacheAligned | 21,719 | 13,638 | 0.9409 | -5.8% |
| Reorder-BFS | 24,706 | 14,985 | 0.9404 | +7.1% |
| Reorder-RCM | 23,576 | 14,818 | 0.9405 | +2.2% |
| Reorder-DFS | 24,835 | 15,027 | 0.9411 | +7.7% |
| Reorder-Cluster | 24,924 | 14,639 | 0.9400 | +8.1% |
| Reorder-Weighted | 23,138 | 13,809 | 0.9405 | +0.3% |
| Chunked | 26,954 | 14,826 | 0.9406 | +16.9% |
| Hugepage | 25,951 | 15,530 | 0.9404 | +12.5% |
| **RCM+Hugepage** | **30,284** | **17,926** | 0.9396 | **+31.3%** |
| **Weighted+Hugepage** | 28,582 | **18,118** | 0.9397 | +23.9% |
| VSAG HGraph | 4,814 | 2,877 | 0.8444 | -79.1% |

### Key Findings

1. **Best Overall Performance**: RCM+Hugepage achieves **31.3% speedup** over baseline
   - ef=50: 30,284 QPS (vs 23,058 baseline)
   - ef=100: 17,926 QPS (vs 13,342 baseline)
   - Maintains recall at 0.9396 (negligible 0.08% drop)

2. **Second Best**: Weighted+Hugepage at **23.9% speedup**
   - Slightly better at ef=100 (18,118 QPS)
   - Good alternative when RCM overhead is a concern

3. **Individual Optimizations**:
   - Hugepage alone: +12.5% (reduces TLB misses)
   - Chunked storage: +16.9% (better cache locality)
   - Graph reordering (BFS/DFS/Cluster): +7-8% (improved traversal locality)
   - RCM reordering: +2.2% (bandwidth matrix optimization)

4. **VSAG HGraph Comparison**: 
   - FAISS baseline is **4.8x faster** than VSAG
   - FAISS optimized is **6.3x faster** than VSAG

## Optimization Techniques

### RCM (Reverse Cuthill-McKee) Algorithm
Reduces matrix bandwidth by reordering graph nodes, placing connected nodes closer together in memory.

### Hugepages
Uses 2MB pages instead of 4KB pages, reducing TLB misses during large vector traversals.

### Combined Approach
RCM reordering + Hugepage allocation provides synergistic benefits:
- RCM improves cache line utilization
- Hugepages reduce page table overhead

## Test Environment

- **CPU**: (Container environment)
- **Threads**: 8 (OMP_NUM_THREADS=8)
- **Compiler**: GCC with -O3 optimization
- **Date**: 2026-02-07

## Reproduction

```bash
# Build
cd /src/faiss-dev/faiss/build
make bench_hnsw_compare -j8

# Run
export LD_LIBRARY_PATH=/src/faiss-dev/vsag/build/src:$LD_LIBRARY_PATH
OMP_NUM_THREADS=8 ./benchs/bench_hnsw_compare /tmp/data/sift-128-euclidean.hdf5 -M 16 -efConstruction 100 -efSearch 50,100
```
