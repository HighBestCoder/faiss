#!/usr/bin/env python3
"""
Quick benchmark script for HNSW cache optimizations.
Runs only the key configurations: baseline, RCM+Hugepage, Weighted+Hugepage
"""

import sys
import os
import time
import numpy as np
import h5py

# Add faiss to path
sys.path.insert(0, "/src/faiss-dev/faiss/build/faiss/python")
import faiss


def load_dataset(path):
    """Load HDF5 dataset from ann-benchmarks format"""
    with h5py.File(path, "r") as f:
        train = np.array(f["train"]).astype("float32")
        test = np.array(f["test"]).astype("float32")
        neighbors = np.array(f["neighbors"])

        # Get distance type from filename
        if "angular" in path:
            metric = "angular"
        else:
            metric = "euclidean"

        return train, test, neighbors, metric


def normalize_vectors(vectors):
    """Normalize vectors for angular/cosine similarity"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def compute_recall(I, gt, k=10):
    """Compute recall@k"""
    n = I.shape[0]
    correct = 0
    for i in range(n):
        correct += len(set(I[i, :k]) & set(gt[i, :k]))
    return correct / (n * k)


def benchmark_index(index, queries, gt, ef_search_values, k=10, warmup=True):
    """Benchmark search performance"""
    results = []

    for ef in ef_search_values:
        index.hnsw.efSearch = ef

        # Warmup
        if warmup:
            _ = index.search(queries[: min(100, len(queries))], k)

        # Benchmark
        t0 = time.time()
        D, I = index.search(queries, k)
        t1 = time.time()

        search_time = t1 - t0
        qps = len(queries) / search_time
        recall = compute_recall(I, gt, k)

        results.append(
            {"ef_search": ef, "search_time": search_time, "qps": qps, "recall": recall}
        )

    return results


def run_benchmark(dataset_path, output_path=None):
    """Run benchmark on a dataset"""
    print(f"\n{'=' * 60}")
    print(f"Dataset: {os.path.basename(dataset_path)}")
    print(f"{'=' * 60}")

    # Load dataset
    train, test, neighbors, metric = load_dataset(dataset_path)

    # Normalize if angular
    if metric == "angular":
        print("Normalizing vectors for angular distance...")
        train = normalize_vectors(train)
        test = normalize_vectors(test)

    nb, d = train.shape
    nq = test.shape[0]
    k = 10

    print(f"Train: {nb} x {d}")
    print(f"Test: {nq} x {d}")
    print(f"Metric: {metric}")

    # Parameters
    M = 16
    efConstruction = 100
    ef_search_values = [50, 100]

    results = {}

    # Metric type for FAISS
    if metric == "angular":
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    else:
        faiss_metric = faiss.METRIC_L2

    # 1. Baseline IndexHNSWFlat
    print("\n[1/3] Building Baseline IndexHNSWFlat...")
    t0 = time.time()
    index_baseline = faiss.IndexHNSWFlat(d, M, faiss_metric)
    index_baseline.hnsw.efConstruction = efConstruction
    index_baseline.add(train)
    build_time_baseline = time.time() - t0
    print(f"  Build time: {build_time_baseline:.2f}s")

    baseline_results = benchmark_index(
        index_baseline, test, neighbors, ef_search_values, k
    )
    results["baseline"] = {
        "build_time": build_time_baseline,
        "searches": baseline_results,
    }

    for r in baseline_results:
        print(f"  ef={r['ef_search']}: {r['qps']:.1f} QPS, Recall={r['recall']:.4f}")

    # Free memory
    del index_baseline

    # 2. IndexHNSWFlat + RCM Reordering simulation
    # Since we can't easily reorder in Python, we'll test with permuted data
    print("\n[2/3] Building with BFS-ordered data (simulating reorder)...")

    # Create a simple BFS-like ordering by building index and using neighbors
    t0 = time.time()
    index_reorder = faiss.IndexHNSWFlat(d, M, faiss_metric)
    index_reorder.hnsw.efConstruction = efConstruction

    # Build initial index to get graph structure
    index_reorder.add(train)

    # Get permutation from BFS traversal
    visited = set()
    queue = [0]
    permutation = []
    while queue and len(permutation) < nb:
        node = queue.pop(0)
        if node in visited or node >= nb:
            continue
        visited.add(node)
        permutation.append(node)

        # Get neighbors at level 0
        neighbors_list = faiss.hnsw.get_neighbor_at(index_reorder.hnsw, 0, node)
        for neighbor in neighbors_list:
            if neighbor not in visited and neighbor < nb:
                queue.append(neighbor)

    # Add remaining nodes
    for i in range(nb):
        if i not in visited:
            permutation.append(i)

    permutation = np.array(permutation)

    # Rebuild with reordered data
    del index_reorder
    train_reordered = train[permutation]

    index_reorder = faiss.IndexHNSWFlat(d, M, faiss_metric)
    index_reorder.hnsw.efConstruction = efConstruction
    index_reorder.add(train_reordered)

    build_time_reorder = time.time() - t0
    print(f"  Build time (with reorder): {build_time_reorder:.2f}s")

    # Create inverse permutation for result mapping
    inv_perm = np.zeros(nb, dtype=np.int64)
    inv_perm[permutation] = np.arange(nb)

    # Benchmark with result remapping
    reorder_results = []
    for ef in ef_search_values:
        index_reorder.hnsw.efSearch = ef

        # Warmup
        _ = index_reorder.search(test[: min(100, nq)], k)

        # Benchmark
        t0 = time.time()
        D, I = index_reorder.search(test, k)
        t1 = time.time()

        # Remap results back to original IDs
        I_remapped = permutation[I]

        search_time = t1 - t0
        qps = nq / search_time
        recall = compute_recall(I_remapped, neighbors, k)

        reorder_results.append(
            {"ef_search": ef, "search_time": search_time, "qps": qps, "recall": recall}
        )
        print(f"  ef={ef}: {qps:.1f} QPS, Recall={recall:.4f}")

    results["bfs_reorder"] = {
        "build_time": build_time_reorder,
        "searches": reorder_results,
    }

    del index_reorder

    # 3. Baseline again for reference (ensure fair comparison)
    print("\n[3/3] Final baseline verification...")
    t0 = time.time()
    index_final = faiss.IndexHNSWFlat(d, M, faiss_metric)
    index_final.hnsw.efConstruction = efConstruction
    index_final.add(train)
    build_time_final = time.time() - t0

    final_results = benchmark_index(index_final, test, neighbors, ef_search_values, k)
    for r in final_results:
        print(f"  ef={r['ef_search']}: {r['qps']:.1f} QPS, Recall={r['recall']:.4f}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    baseline_qps = results["baseline"]["searches"][0]["qps"]
    reorder_qps = results["bfs_reorder"]["searches"][0]["qps"]

    improvement = (reorder_qps - baseline_qps) / baseline_qps * 100

    print(f"Baseline QPS (ef=50): {baseline_qps:.1f}")
    print(f"BFS Reorder QPS (ef=50): {reorder_qps:.1f}")
    print(f"Improvement: {improvement:+.1f}%")

    return results


def main():
    datasets = [
        "/src/faiss-dev/dataset/sift-128-euclidean.hdf5",
        "/src/faiss-dev/dataset/gist-960-euclidean.hdf5",
        "/src/faiss-dev/dataset/glove-100-angular.hdf5",
        "/src/faiss-dev/dataset/nytimes-256-angular.hdf5",
    ]

    for ds in datasets:
        if os.path.exists(ds):
            try:
                run_benchmark(ds)
            except Exception as e:
                print(f"Error with {ds}: {e}")
                import traceback

                traceback.print_exc()
        else:
            print(f"Dataset not found: {ds}")


if __name__ == "__main__":
    main()
