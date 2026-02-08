#!/bin/bash
# Quick test of madvise THP vs MAP_HUGETLB on GloVe-100

cd /src/faiss-dev/faiss/build/benchs

echo "Running focused THP comparison on GloVe-100..."
echo "This compares:"
echo "  - Old approach: MAP_HUGETLB with 16384-vector chunks"  
echo "  - New approach: madvise(MADV_HUGEPAGE) with adaptive chunks"
echo ""

export OMP_NUM_THREADS=8

# Run full benchmark (takes ~30 minutes)
timeout 3600 ./bench_hnsw_compare /src/faiss-dev/dataset/glove-100-angular.hdf5 2>&1 | tee /tmp/thp_test.log

echo ""
echo "Key results:"
grep -E "(Hugepage|THP|Reorder-Weighted)" /tmp/thp_test.log | head -20
