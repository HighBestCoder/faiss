# FAISS HNSW Cache Optimization - Performance Test Results

## Overview

This directory contains benchmark results for FAISS HNSW cache optimization techniques tested on multiple standard ANN benchmark datasets.

## Key Findings

| Dataset | Best Strategy | Improvement | Notes |
|---------|---------------|-------------|-------|
| **SIFT-1M** | RCM+Hugepage | **+31.3%** | Large dataset + moderate dim |
| **GIST-960** | Weighted+Hugepage | **+26.4%** | High-dim benefits from Hugepage |
| **NYTimes-256** | Reorder-BFS | **+10.0%** | Small dataset, reorder only |
| **GloVe-100** | Reorder-Weighted | **+9.9%** | Low-dim, Hugepage hurts |

## Datasets Tested

| Dataset | Dimensions | Vectors | Metric | Status |
|---------|------------|---------|--------|--------|
| SIFT-1M | 128 | 1,000,000 | L2 | ✅ Full benchmark |
| GIST-960 | 960 | 1,000,000 | L2 | ✅ Full benchmark |
| GloVe-100 | 100 | 1,183,514 | Angular | ✅ Full benchmark |
| NYTimes-256 | 256 | 290,000 | Angular | ✅ Full benchmark |
| Deep-96 | 96 | 9,990,000 | Angular | ⏱️ Pending (10M vectors) |

## Optimization Techniques Summary

### When to Use Hugepage+Reorder

| Factor | Use Hugepage | Use Reorder Only |
|--------|--------------|------------------|
| Large vectors (>500B) | ✅ | |
| Large dataset (1M+) | ✅ | |
| Small vectors (<500B) | | ✅ |
| Small dataset (<500K) | | ✅ |
| High dimensions (>256) | ✅ | |

### Best Strategy by Dataset Characteristics

| Scenario | Best Strategy |
|----------|---------------|
| High-dim + Large dataset (GIST) | Weighted+Hugepage |
| Moderate-dim + Large dataset (SIFT) | RCM+Hugepage |
| Low-dim + Any size (GloVe) | Reorder-RCM or Weighted |
| Small dataset (NYTimes) | Reorder-BFS |

## Result Files

- [SIFT-1M.md](SIFT-1M.md) - L2, 128D, 1M vectors (+31.3%)
- [GIST-960.md](GIST-960.md) - L2, 960D, 1M vectors (+26.4%)
- [GloVe-100.md](GloVe-100.md) - Angular, 100D, 1.18M vectors (+9.9%)
- [NYTimes-256.md](NYTimes-256.md) - Angular, 256D, 290K vectors (+10.0%)

## Reproduction

### Using Makefile (Recommended)

```bash
cd /src/faiss-dev/faiss/benchs

# Run all benchmarks
make test

# Run individual datasets
make test-sift      # SIFT-1M (L2)
make test-gist      # GIST-960 (L2, high-dim)
make test-glove     # GloVe-100 (Angular)
make test-nytimes   # NYTimes-256 (Angular)
make test-deep      # Deep-96 (Angular, 10M vectors - slow!)

# Customize parameters
make test-sift M=32 EF_CONSTRUCTION=200 EF_SEARCH=50,100,200
```

### Manual Run

```bash
cd /src/faiss-dev/faiss/build
make bench_hnsw_compare -j8

OMP_NUM_THREADS=8 ./benchs/bench_hnsw_compare \
    /src/faiss-dev/dataset/sift-128-euclidean.hdf5 \
    -M 16 -efConstruction 100 -efSearch 50,100
```

## Implementation

The optimizations are implemented in:
- `benchs/bench_hnsw_compare.cpp` - 5 graph reordering algorithms + memory optimizations
- `benchs/Makefile` - Convenient test targets for all datasets

### Supported Features
- L2 (Euclidean) distance
- Inner Product / Angular / Cosine similarity (auto-detected from filename)
- Vector normalization for angular datasets
- All optimization strategies with metric support

## Date

2026-02-07
