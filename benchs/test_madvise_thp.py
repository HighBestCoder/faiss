#!/usr/bin/env python3
"""
Focused test to compare MAP_HUGETLB (old) vs madvise MADV_HUGEPAGE (new).
Tests on low-dimension datasets where the old approach showed regression.
"""

import subprocess
import os
import re
import sys

DATASETS = [
    ("/src/faiss-dev/dataset/glove-100-angular.hdf5", "GloVe-100"),
    ("/src/faiss-dev/dataset/nytimes-256-angular.hdf5", "NYTimes-256"),
]

BUILD_DIR = "/src/faiss-dev/faiss/build/benchs"


def run_benchmark(dataset_path):
    """Run benchmark and extract key results."""
    cmd = [f"{BUILD_DIR}/bench_hnsw_compare", dataset_path, "-efSearch", "100"]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "8"

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    output = result.stdout + result.stderr

    return output


def parse_results(output):
    """Extract QPS for relevant index types."""
    results = {}

    patterns = [
        (
            "Reorder-Weighted",
            r"FAISS Reorder-Weighted\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+[\d.]+",
        ),
        ("Hugepage", r"FAISS Hugepage\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+[\d.]+"),
        (
            "Weighted+Hugepage",
            r"FAISS Weighted\+Hugepage\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+[\d.]+",
        ),
        (
            "THP (madvise)",
            r"FAISS THP \(madvise\)\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+[\d.]+",
        ),
        (
            "Weighted+THP (madvise)",
            r"FAISS Weighted\+THP \(madvise\)\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+[\d.]+",
        ),
    ]

    for name, pattern in patterns:
        match = re.search(pattern, output)
        if match:
            results[name] = float(match.group(1))

    return results


def main():
    print("=" * 70)
    print("  madvise(MADV_HUGEPAGE) vs MAP_HUGETLB Comparison")
    print("=" * 70)
    print()

    for dataset_path, dataset_name in DATASETS:
        print(f"\n{'=' * 70}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'=' * 70}")

        if not os.path.exists(dataset_path):
            print(f"  SKIPPED: {dataset_path} not found")
            continue

        print("  Running benchmark (this may take 10-20 minutes)...")

        try:
            output = run_benchmark(dataset_path)
            results = parse_results(output)

            if results:
                print(f"\n  Results (QPS at efSearch=100):")
                print(f"  {'-' * 50}")

                for name, qps in sorted(results.items(), key=lambda x: -x[1]):
                    print(f"  {name:30s} {qps:10.1f} QPS")

                # Compare old vs new hugepage
                if (
                    "Weighted+Hugepage" in results
                    and "Weighted+THP (madvise)" in results
                ):
                    old = results["Weighted+Hugepage"]
                    new = results["Weighted+THP (madvise)"]
                    improvement = (new - old) / old * 100
                    print(f"\n  madvise vs MAP_HUGETLB: {improvement:+.1f}%")
            else:
                print("  No results parsed")
                print("  Raw output (last 2000 chars):")
                print(output[-2000:])

        except subprocess.TimeoutExpired:
            print("  TIMEOUT after 60 minutes")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
