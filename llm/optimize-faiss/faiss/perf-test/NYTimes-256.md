# NYTimes-256 Benchmark Results

## Dataset Information

| Property | Value |
|----------|-------|
| Dataset | NYTimes-256 |
| Dimensions | 256 |
| Training Vectors | 290,000 |
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
| IndexHNSWFlat (Baseline) | 13,641 | 0.8065 | 134.87 | - |
| CacheAligned | 6,543 | 0.8074 | 178.38 | -52.0% |
| Reorder-BFS | 15,000 | 0.8044 | 135.53 | **+10.0%** |
| Reorder-RCM | 14,548 | 0.8063 | 134.05 | **+6.6%** |
| Reorder-DFS | 14,732 | 0.8065 | 134.26 | **+8.0%** |
| Reorder-Cluster | 14,530 | 0.8061 | 134.17 | **+6.5%** |
| Reorder-Weighted | 14,346 | 0.8022 | 133.80 | +5.2% |
| Chunked | 8,207 | 0.8052 | 162.61 | -39.8% |
| Hugepage | 8,417 | 0.8049 | 155.00 | -38.3% |
| RCM+Hugepage | 8,899 | 0.8065 | 287.43 | -34.8% |
| Weighted+Hugepage | 8,919 | 0.8039 | 284.36 | -34.6% |

### efSearch = 100

| Index Type | QPS | Recall@10 | Build Time (s) | vs Baseline |
|------------|-----|-----------|----------------|-------------|
| IndexHNSWFlat (Baseline) | 8,078 | 0.8515 | 134.87 | - |
| CacheAligned | 3,835 | 0.8512 | 178.38 | -52.5% |
| Reorder-BFS | 8,494 | 0.8526 | 135.53 | +5.2% |
| Reorder-RCM | 8,512 | 0.8503 | 134.05 | +5.4% |
| Reorder-DFS | 8,501 | 0.8510 | 134.26 | +5.2% |
| Reorder-Cluster | 8,769 | 0.8514 | 134.17 | **+8.6%** |
| Reorder-Weighted | 8,831 | 0.8481 | 133.80 | **+9.3%** |
| Chunked | 4,950 | 0.8507 | 162.61 | -38.7% |
| Hugepage | 5,126 | 0.8508 | 155.00 | -36.5% |
| RCM+Hugepage | 5,213 | 0.8516 | 287.43 | -35.5% |
| Weighted+Hugepage | 5,378 | 0.8497 | 284.36 | -33.4% |

### VSAG Comparison

| Index Type | efSearch | QPS | Recall@10 |
|------------|----------|-----|-----------|
| VSAG HGraph | 50 | 10,249 | 0.0059 |
| VSAG HGraph | 100 | 5,178 | 0.0088 |

**Note**: VSAG has near-zero recall on NYTimes - likely metric configuration issue.

## Analysis

### Key Observations

1. **Graph Reordering Consistently Helps**:
   - At ef=50: BFS best (+10.0%), DFS (+8.0%), RCM (+6.6%)
   - At ef=100: Weighted best (+9.3%), Cluster (+8.6%)

2. **Chunked/Hugepage Severely Underperforms**:
   - -33% to -40% regression across all configurations
   - Similar pattern to GloVe-100

3. **Best Configuration for NYTimes-256**:
   - ef=50: Use **Reorder-BFS** (+10.0% QPS)
   - ef=100: Use **Reorder-Weighted** (+9.3% QPS)

### Pattern Emerging: Dataset Size Matters

| Dataset | Vectors | Dimensions | Hugepage Benefit |
|---------|---------|------------|------------------|
| SIFT-1M | 1,000,000 | 128 | **+31%** |
| GloVe-100 | 1,183,514 | 100 | -25% |
| NYTimes-256 | 290,000 | 256 | -35% |

**Hypothesis**: Hugepage optimization benefits large datasets with moderate dimensions.
- SIFT-1M: Large dataset → high TLB pressure → Hugepage helps
- GloVe/NYTimes: Smaller or low-dim → less TLB pressure → overhead dominates

### Reordering Always Helps

| Dataset | Best Reorder Strategy | Improvement |
|---------|----------------------|-------------|
| SIFT-1M (ef=50) | Cluster | +8.1% |
| GloVe-100 (ef=50) | RCM | +5.4% |
| NYTimes-256 (ef=50) | BFS | +10.0% |

Graph reordering provides consistent 5-10% improvement across all datasets.

## Recommendations

For NYTimes-256 and similar medium-sized angular datasets:
1. Use **BFS or DFS graph reordering** for ~8-10% improvement
2. **Avoid** Chunked/Hugepage storage entirely
3. Keep standard IndexHNSWFlat with reordering

## Test Environment

- **CPU**: Container environment
- **Threads**: 8 (OMP_NUM_THREADS=8)
- **FAISS Version**: Latest from source with IP metric support
- **Date**: 2026-02-07
- **Benchmark**: C++ bench_hnsw_compare with Inner Product metric
