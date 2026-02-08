#!/usr/bin/env python3
"""
FAISS vs VSAG HGraph Performance Benchmark

Compares:
- FAISS IndexHNSWFlat (baseline)
- FAISS with our optimizations (CacheAligned, GraphReorder via C++ benchmark)
- VSAG HGraph (the main VSAG algorithm from VLDB 2025 paper)

Reference: https://arxiv.org/abs/2503.17911
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import h5py
import numpy as np

try:
    import faiss
except ImportError:
    print("Error: faiss not installed. Run: pip install faiss-cpu")
    sys.exit(1)

try:
    import pyvsag
except ImportError:
    print("Error: pyvsag not installed. Run: pip install pyvsag")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    index_type: str
    dataset: str
    ef_search: int
    build_time_sec: float
    search_time_sec: float
    qps: float
    recall_at_10: float
    memory_mb: float
    params: dict


def load_hdf5_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(filepath, "r") as f:
        train = np.array(f["train"]).astype(np.float32)
        test = np.array(f["test"]).astype(np.float32)
        neighbors = np.array(f["neighbors"]).astype(np.int64)
    return train, test, neighbors


def compute_recall(
    predictions: np.ndarray, ground_truth: np.ndarray, k: int = 10
) -> float:
    n_queries = predictions.shape[0]
    recalls = []
    for i in range(n_queries):
        pred_set = set(predictions[i, :k].tolist())
        gt_set = set(ground_truth[i, :k].tolist())
        recalls.append(len(pred_set & gt_set) / k)
    return float(np.mean(recalls))


def get_memory_mb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except:
        return 0.0


def benchmark_faiss_hnsw(
    train_data: np.ndarray,
    query_data: np.ndarray,
    ground_truth: np.ndarray,
    M: int,
    ef_construction: int,
    ef_search_list: List[int],
    k: int = 10,
    dataset_name: str = "unknown",
) -> List[BenchmarkResult]:
    dim = train_data.shape[1]
    results = []

    print(
        f"\nBuilding FAISS IndexHNSWFlat (M={M}, efConstruction={ef_construction})..."
    )
    mem_before = get_memory_mb()

    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = ef_construction

    t0 = time.time()
    index.add(train_data)
    build_time = time.time() - t0

    mem_after = get_memory_mb()
    memory_mb = mem_after - mem_before

    print(f"  Build time: {build_time:.2f}s, Memory: {memory_mb:.2f}MB")

    for ef_search in ef_search_list:
        index.hnsw.efSearch = ef_search

        faiss.omp_set_num_threads(1)

        t0 = time.time()
        distances, predictions = index.search(query_data, k)
        search_time = time.time() - t0

        qps = len(query_data) / search_time
        recall = compute_recall(predictions, ground_truth, k)

        result = BenchmarkResult(
            index_type="FAISS/IndexHNSWFlat",
            dataset=dataset_name,
            ef_search=ef_search,
            build_time_sec=build_time,
            search_time_sec=search_time,
            qps=qps,
            recall_at_10=recall,
            memory_mb=memory_mb,
            params={"M": M, "ef_construction": ef_construction},
        )
        results.append(result)
        print(f"  efSearch={ef_search:3d}: QPS={qps:8.1f}, Recall@{k}={recall:.4f}")

    return results


def benchmark_vsag_hgraph(
    train_data: np.ndarray,
    query_data: np.ndarray,
    ground_truth: np.ndarray,
    M: int,
    ef_construction: int,
    ef_search_list: List[int],
    k: int = 10,
    dataset_name: str = "unknown",
) -> List[BenchmarkResult]:
    dim = train_data.shape[1]
    nb = train_data.shape[0]
    nq = query_data.shape[0]
    results = []

    print(f"\nBuilding VSAG HGraph (M={M}, efConstruction={ef_construction}, SQ8)...")
    mem_before = get_memory_mb()

    build_params = json.dumps(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": int(dim),
            "index_param": {
                "base_quantization_type": "sq8",
                "max_degree": M,
                "ef_construction": ef_construction,
                "alpha": 1.2,
            },
        }
    )

    t0 = time.time()
    index = pyvsag.Index("hgraph", build_params)

    ids = np.arange(nb, dtype=np.int64)
    index.build(train_data.flatten(), ids, nb, dim)
    build_time = time.time() - t0

    mem_after = get_memory_mb()
    memory_mb = mem_after - mem_before

    print(f"  Build time: {build_time:.2f}s, Memory: {memory_mb:.2f}MB")

    for ef_search in ef_search_list:
        search_params = json.dumps({"hgraph": {"ef_search": ef_search}})

        all_results = []
        t0 = time.time()
        for q in range(nq):
            query_vec = query_data[q : q + 1].flatten()
            result_ids, result_dists = index.knn_search(query_vec, k, search_params)
            all_results.append(result_ids[:k])
        search_time = time.time() - t0

        predictions = np.array(all_results, dtype=np.int64)

        qps = nq / search_time
        recall = compute_recall(predictions, ground_truth, k)

        result = BenchmarkResult(
            index_type="VSAG/HGraph",
            dataset=dataset_name,
            ef_search=ef_search,
            build_time_sec=build_time,
            search_time_sec=search_time,
            qps=qps,
            recall_at_10=recall,
            memory_mb=memory_mb,
            params={"M": M, "ef_construction": ef_construction, "quantization": "sq8"},
        )
        results.append(result)
        print(f"  efSearch={ef_search:3d}: QPS={qps:8.1f}, Recall@{k}={recall:.4f}")

    return results


def print_comparison_table(results: List[BenchmarkResult]):
    print("\n" + "=" * 90)
    print("  Recall-QPS Comparison")
    print("=" * 90)
    print(f"{'Index Type':<25} {'efSearch':>10} {'QPS':>12} {'Recall@10':>12}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: (x.index_type, x.ef_search)):
        print(
            f"{r.index_type:<25} {r.ef_search:>10} {r.qps:>12.1f} {r.recall_at_10:>12.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="FAISS vs VSAG HGraph Benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/tmp/data/sift-128-euclidean.hdf5",
        help="Path to HDF5 dataset",
    )
    parser.add_argument("--M", type=int, default=32, help="HNSW M parameter")
    parser.add_argument(
        "--ef-construction", type=int, default=200, help="efConstruction parameter"
    )
    parser.add_argument(
        "--ef-search",
        type=str,
        default="50,100,200,400",
        help="Comma-separated efSearch values to test",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file for results"
    )
    parser.add_argument(
        "--max-vectors", type=int, default=None, help="Limit number of vectors"
    )
    parser.add_argument(
        "--max-queries", type=int, default=None, help="Limit number of queries"
    )
    args = parser.parse_args()

    ef_search_list = [int(x) for x in args.ef_search.split(",")]
    dataset_name = os.path.basename(args.dataset).replace(".hdf5", "")

    print("=" * 90)
    print("  FAISS vs VSAG HGraph Performance Benchmark")
    print("  Reference: https://arxiv.org/abs/2503.17911 (VLDB 2025)")
    print("=" * 90)
    print(f"\nDataset: {args.dataset}")

    print("Loading dataset...")
    train_data, query_data, ground_truth = load_hdf5_dataset(args.dataset)

    if args.max_vectors:
        train_data = train_data[: args.max_vectors]
        print(f"  (Recomputing ground truth for {args.max_vectors} vectors...)")
        brute_index = faiss.IndexFlatL2(train_data.shape[1])
        brute_index.add(train_data)
        _, ground_truth = brute_index.search(query_data, 100)
    if args.max_queries:
        query_data = query_data[: args.max_queries]
        ground_truth = ground_truth[: args.max_queries]

    print(f"  Train vectors: {train_data.shape}")
    print(f"  Query vectors: {query_data.shape}")
    print(f"  Dimension: {train_data.shape[1]}")
    print(f"\nParameters:")
    print(f"  M: {args.M}")
    print(f"  efConstruction: {args.ef_construction}")
    print(f"  efSearch values: {ef_search_list}")

    all_results = []

    faiss_results = benchmark_faiss_hnsw(
        train_data,
        query_data,
        ground_truth,
        M=args.M,
        ef_construction=args.ef_construction,
        ef_search_list=ef_search_list,
        dataset_name=dataset_name,
    )
    all_results.extend(faiss_results)

    vsag_results = benchmark_vsag_hgraph(
        train_data,
        query_data,
        ground_truth,
        M=args.M,
        ef_construction=args.ef_construction,
        ef_search_list=ef_search_list,
        dataset_name=dataset_name,
    )
    all_results.extend(vsag_results)

    print_comparison_table(all_results)

    print("\n" + "=" * 90)
    print("  Summary at Similar Recall Levels")
    print("=" * 90)

    faiss_by_recall = {r.recall_at_10: r for r in faiss_results}
    vsag_by_recall = {r.recall_at_10: r for r in vsag_results}

    for target in [0.90, 0.95, 0.99]:
        faiss_match = min(faiss_results, key=lambda r: abs(r.recall_at_10 - target))
        vsag_match = min(vsag_results, key=lambda r: abs(r.recall_at_10 - target))

        if faiss_match.qps > 0:
            speedup = vsag_match.qps / faiss_match.qps
            print(f"  ~{target * 100:.0f}% Recall:")
            print(
                f"    FAISS: {faiss_match.qps:.1f} QPS (efSearch={faiss_match.ef_search}, recall={faiss_match.recall_at_10:.4f})"
            )
            print(
                f"    VSAG:  {vsag_match.qps:.1f} QPS (efSearch={vsag_match.ef_search}, recall={vsag_match.recall_at_10:.4f})"
            )
            print(f"    Speedup: {speedup:.2f}x")

    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
