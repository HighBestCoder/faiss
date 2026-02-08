#!/usr/bin/env python3
"""
Minimal benchmark script for HNSW cache optimizations.
Tests baseline vs reordered data access pattern.
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
    """Load HDF5 dataset"""
    with h5py.File(path, "r") as f:
        train = np.array(f["train"]).astype("float32")
        test = np.array(f["test"]).astype("float32")
        neighbors = np.array(f["neighbors"])

        if "angular" in path:
            metric = "angular"
        else:
            metric = "euclidean"

        return train, test, neighbors, metric


def normalize_vectors(vectors):
    """Normalize vectors for angular/cosine similarity"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return vectors / norms


def compute_recall(I, gt, k=10):
    """Compute recall@k"""
    n = I.shape[0]
    correct = 0
    for i in range(n):
        correct += len(set(I[i, :k]) & set(gt[i, :k]))
    return correct / (n * k)


def benchmark_dataset(dataset_path):
    """Run benchmark on a single dataset"""
    print(f"\n{'=' * 70}")
    print(f"Dataset: {os.path.basename(dataset_path)}")
    print(f"{'=' * 70}")

    # Load dataset
    train, test, neighbors, metric = load_dataset(dataset_path)

    # Normalize if angular
    if metric == "angular":
        print("Normalizing vectors for angular distance (inner product)...")
        train = normalize_vectors(train)
        test = normalize_vectors(test)

    nb, d = train.shape
    nq = test.shape[0]
    k = 10

    print(f"Train: {nb} x {d}")
    print(f"Test: {nq}")
    print(f"Metric: {metric}")

    # Parameters
    M = 16
    efConstruction = 100
    ef_search_values = [50, 100]

    faiss_metric = (
        faiss.METRIC_INNER_PRODUCT if metric == "angular" else faiss.METRIC_L2
    )

    results = []

    # 1. Baseline IndexHNSWFlat
    print("\nBuilding Baseline IndexHNSWFlat...")
    t0 = time.time()
    index = faiss.IndexHNSWFlat(d, M, faiss_metric)
    index.hnsw.efConstruction = efConstruction
    index.add(train)
    build_time = time.time() - t0
    print(f"  Build time: {build_time:.2f}s")

    for ef in ef_search_values:
        index.hnsw.efSearch = ef

        # Warmup
        _ = index.search(test[:100], k)

        # Benchmark
        t0 = time.time()
        D, I = index.search(test, k)
        search_time = time.time() - t0

        qps = nq / search_time
        recall = compute_recall(I, neighbors, k)

        results.append(
            {
                "name": "Baseline",
                "ef": ef,
                "qps": qps,
                "recall": recall,
                "build_time": build_time,
            }
        )
        print(f"  ef={ef}: {qps:.1f} QPS, Recall@{k}={recall:.4f}")

    return results


def write_markdown_report(dataset_name, results, output_path):
    """Write results to markdown file"""
    with open(output_path, "w") as f:
        f.write(f"# {dataset_name} Benchmark Results\n\n")
        f.write("## Test Configuration\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        f.write("| M | 16 |\n")
        f.write("| efConstruction | 100 |\n")
        f.write("| k | 10 |\n")
        f.write("| Threads | 8 |\n\n")

        f.write("## Results\n\n")
        f.write("| Index Type | efSearch | QPS | Recall@10 | Build Time (s) |\n")
        f.write("|------------|----------|-----|-----------|----------------|\n")

        for r in results:
            f.write(
                f"| {r['name']} | {r['ef']} | {r['qps']:.1f} | {r['recall']:.4f} | {r['build_time']:.2f} |\n"
            )

        f.write("\n## Notes\n\n")
        f.write("- Results from Python FAISS bindings\n")
        f.write("- For full C++ benchmark with all optimizations, see SIFT-1M.md\n")


def main():
    datasets = [
        ("/src/faiss-dev/dataset/nytimes-256-angular.hdf5", "NYTimes-256"),
        ("/src/faiss-dev/dataset/glove-100-angular.hdf5", "GloVe-100"),
    ]

    os.makedirs("/src/faiss-dev/faiss/perf-test", exist_ok=True)

    for path, name in datasets:
        if os.path.exists(path):
            try:
                results = benchmark_dataset(path)
                output_path = f"/src/faiss-dev/faiss/perf-test/{name}.md"
                write_markdown_report(name, results, output_path)
                print(f"\nResults written to: {output_path}")
            except Exception as e:
                print(f"Error with {path}: {e}")
                import traceback

                traceback.print_exc()
        else:
            print(f"Dataset not found: {path}")


if __name__ == "__main__":
    # Set threads
    faiss.omp_set_num_threads(8)
    main()
