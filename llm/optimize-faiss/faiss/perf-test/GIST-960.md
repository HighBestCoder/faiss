# GIST-960 Benchmark Results

## Dataset Information

| Property | Value |
|----------|-------|
| Dataset | GIST-960 |
| Dimensions | 960 |
| Training Vectors | 1,000,000 |
| Query Vectors | 1,000 |
| Distance Metric | L2 (Euclidean) |
| Source | [ann-benchmarks.com](http://ann-benchmarks.com/) |

## Index Parameters

| Parameter | Value |
|-----------|-------|
| M (connections per layer) | 16 |
| efConstruction | 100 |
| efSearch | 50, 100 |
| Threads | 8 |

## Benchmark Results (C++ Full Test)

### efSearch = 50

| Index Type | QPS | Recall@10 | Build Time (s) | vs Baseline |
|------------|-----|-----------|----------------|-------------|
| IndexHNSWFlat (Baseline) | 5,726 | 0.6759 | 412.78 | - |
| CacheAligned | 6,104 | 0.6751 | 435.11 | +6.6% |
| Reorder-BFS | 6,446 | 0.6787 | 418.98 | **+12.6%** |
| Reorder-RCM | 6,531 | 0.6719 | 428.46 | **+14.1%** |
| Reorder-DFS | 4,191 | 0.6719 | 428.84 | -26.8% |
| Reorder-Cluster | 5,199 | 0.6764 | 427.35 | -9.2% |
| Reorder-Weighted | 6,839 | 0.6707 | 419.68 | **+19.5%** |
| Chunked | 6,788 | 0.6767 | 412.84 | **+18.5%** |
| Hugepage | 6,317 | 0.6742 | 410.96 | **+10.3%** |
| RCM+Hugepage | 6,686 | 0.6810 | 831.63 | **+16.8%** |
| Weighted+Hugepage | 7,237 | 0.6740 | 825.21 | **+26.4%** |

### efSearch = 100

| Index Type | QPS | Recall@10 | Build Time (s) | vs Baseline |
|------------|-----|-----------|----------------|-------------|
| IndexHNSWFlat (Baseline) | 3,732 | 0.7833 | 412.78 | - |
| CacheAligned | 3,632 | 0.7827 | 435.11 | -2.7% |
| Reorder-BFS | 4,032 | 0.7839 | 418.98 | +8.1% |
| Reorder-RCM | 4,300 | 0.7830 | 428.46 | **+15.2%** |
| Reorder-DFS | 3,017 | 0.7817 | 428.84 | -19.1% |
| Reorder-Cluster | 2,985 | 0.7848 | 427.35 | -20.0% |
| Reorder-Weighted | 4,120 | 0.7800 | 419.68 | **+10.4%** |
| Chunked | 3,659 | 0.7845 | 412.84 | -2.0% |
| Hugepage | 3,982 | 0.7810 | 410.96 | +6.7% |
| RCM+Hugepage | 4,125 | 0.7860 | 831.63 | **+10.5%** |
| Weighted+Hugepage | 4,383 | 0.7821 | 825.21 | **+17.4%** |

### VSAG Comparison

| Index Type | efSearch | QPS | Recall@10 |
|------------|----------|-----|-----------|
| VSAG HGraph | 50 | 2,251 | 0.4654 |
| VSAG HGraph | 100 | 1,411 | 0.5548 |

FAISS is 2.5-3x faster than VSAG with significantly higher recall.

## Analysis

### Key Observations

1. **Weighted+Hugepage Best for High Dimensions**:
   - ef=50: **+26.4%** improvement (7,237 vs 5,726 QPS)
   - ef=100: **+17.4%** improvement (4,383 vs 3,732 QPS)

2. **Multiple Optimizations Work Well**:
   - Reorder-Weighted: +19.5% / +10.4%
   - Chunked: +18.5% (ef=50 only)
   - Reorder-RCM: +14.1% / +15.2%
   - Reorder-BFS: +12.6% / +8.1%

3. **Some Strategies Hurt Performance**:
   - DFS: -26.8% / -19.1% (unexpected regression)
   - Cluster: -9.2% / -20.0%

4. **High-Dim Benefits from Memory Optimizations**:
   - Unlike GloVe/NYTimes, GIST-960 benefits from Hugepage
   - Large vector size (3.84KB) → high TLB pressure

### Comparison with Other Datasets

| Dataset | Dimensions | Vector Size | Best Strategy | Improvement |
|---------|------------|-------------|---------------|-------------|
| SIFT-1M | 128 | 512B | RCM+Hugepage | **+31.3%** |
| GIST-960 | 960 | 3.84KB | Weighted+Hugepage | **+26.4%** |
| GloVe-100 | 100 | 400B | Reorder-RCM | +5.4% |
| NYTimes-256 | 256 | 1KB | Reorder-BFS | +10.0% |

### Pattern: When Does Hugepage Help?

| Factor | Helps | Hurts |
|--------|-------|-------|
| Large vectors (>1KB) | ✅ | |
| Large dataset (1M+) | ✅ | |
| Small vectors (<500B) | | ❌ |
| Small dataset (<500K) | | ❌ |

**Hugepage is beneficial when TLB pressure is high (large vectors * large dataset).**

## Recommendations

For GIST-960 and similar high-dimensional L2 datasets:
1. Use **Weighted+Hugepage** for best performance (+26%)
2. Alternative: **Reorder-RCM** if build time is critical (+14%)
3. **Avoid** DFS and Cluster reordering - they hurt performance

## Memory Usage

| Strategy | Memory (MB) | Notes |
|----------|-------------|-------|
| IndexHNSWFlat | 3,999 | Full vector storage |
| CacheAligned | 68 | Reduced memory footprint |
| Reorder strategies | 0-14 | Minimal overhead |
| RCM/Weighted+Hugepage | 3,771-3.6 | Varies |

## Test Environment

- **CPU**: Container environment
- **Threads**: 8 (OMP_NUM_THREADS=8)
- **FAISS Version**: Latest from source
- **Date**: 2026-02-07
- **Benchmark**: C++ bench_hnsw_compare with L2 metric
