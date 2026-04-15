# GloVe-100 Benchmark Results

## Dataset Information

| Property | Value |
|----------|-------|
| Dataset | GloVe-100 |
| Dimensions | 100 |
| Training Vectors | 1,183,514 |
| Query Vectors | 10,000 |
| Distance Metric | Angular (Inner Product) |
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
| IndexHNSWFlat (Baseline) | 18,793 | 0.7272 | 193.79 | - |
| CacheAligned | 7,846 | 0.7268 | 318.85 | -58.2% |
| Reorder-BFS | 18,627 | 0.7274 | 194.45 | -0.9% |
| Reorder-RCM | 19,799 | 0.7267 | 191.99 | **+5.4%** |
| Reorder-DFS | 18,953 | 0.7261 | 191.49 | +0.9% |
| Reorder-Cluster | 18,933 | 0.7271 | 194.97 | +0.7% |
| Reorder-Weighted | 19,634 | 0.7280 | 194.06 | **+4.5%** |
| Chunked | 12,702 | 0.7258 | 228.73 | -32.4% |
| Hugepage | 12,977 | 0.7260 | 222.58 | -30.9% |
| RCM+Hugepage | 14,139 | 0.7247 | 394.72 | -24.8% |
| Weighted+Hugepage | 14,213 | 0.7262 | 402.70 | -24.4% |

### efSearch = 100

| Index Type | QPS | Recall@10 | Build Time (s) | vs Baseline |
|------------|-----|-----------|----------------|-------------|
| IndexHNSWFlat (Baseline) | 10,985 | 0.8018 | 193.79 | - |
| CacheAligned | 4,727 | 0.8030 | 318.85 | -57.0% |
| Reorder-BFS | 11,667 | 0.8026 | 194.45 | **+6.2%** |
| Reorder-RCM | 10,941 | 0.8020 | 191.99 | -0.4% |
| Reorder-DFS | 11,800 | 0.8018 | 191.49 | **+7.4%** |
| Reorder-Cluster | 11,943 | 0.8028 | 194.97 | **+8.7%** |
| Reorder-Weighted | 12,075 | 0.8025 | 194.06 | **+9.9%** |
| Chunked | 7,335 | 0.8021 | 228.73 | -33.2% |
| Hugepage | 7,661 | 0.8020 | 222.58 | -30.3% |
| RCM+Hugepage | 8,108 | 0.8023 | 394.72 | -26.2% |
| Weighted+Hugepage | 8,299 | 0.8021 | 402.70 | -24.4% |

### VSAG Comparison

| Index Type | efSearch | QPS | Recall@10 |
|------------|----------|-----|-----------|
| VSAG HGraph | 50 | 3,586 | 0.3795 |
| VSAG HGraph | 100 | 2,039 | 0.4820 |

## Analysis

### Key Observations

1. **Graph Reordering Works Best at Higher efSearch**: 
   - At ef=50: RCM best (+5.4%), Weighted (+4.5%)
   - At ef=100: Weighted best (+9.9%), Cluster (+8.7%), DFS (+7.4%)

2. **Chunked/Hugepage Underperforms on GloVe**:
   - Unlike SIFT-1M (+31.3%), GloVe shows -24% to -33% regression
   - Likely due to different memory access patterns in low-dim (100D) vectors
   - Overhead of chunked storage outweighs cache benefits for smaller vectors

3. **Best Configuration for GloVe-100**:
   - ef=50: Use **Reorder-RCM** (+5.4% QPS)
   - ef=100: Use **Reorder-Weighted** (+9.9% QPS)

4. **VSAG Very Poor on Angular**:
   - 5x slower than FAISS baseline
   - 50% lower recall - likely metric mismatch or configuration issue

### Why Different from SIFT-1M?

| Factor | SIFT-1M | GloVe-100 |
|--------|---------|-----------|
| Dimensions | 128 | 100 |
| Vector Size | 512 bytes | 400 bytes |
| Vectors per Cache Line | 0.125 | 0.16 |
| TLB Pressure | Higher | Lower |
| Hugepage Benefit | High | Low/Negative |

The smaller vector size (100D vs 128D) means:
- Less TLB pressure → Hugepage benefit reduced
- Chunked storage overhead dominates
- Simple reordering provides best results

## Recommendations

For GloVe-100 and similar low-dimensional angular datasets:
1. Use **graph reordering** (RCM or Weighted) for ~5-10% improvement
2. **Avoid** Chunked/Hugepage storage - adds overhead without benefit
3. Standard IndexHNSWFlat with reordering is optimal

## Test Environment

- **CPU**: Container environment
- **Threads**: 8 (OMP_NUM_THREADS=8)
- **FAISS Version**: Latest from source with IP metric support
- **Date**: 2026-02-07
- **Benchmark**: C++ bench_hnsw_compare with Inner Product metric
