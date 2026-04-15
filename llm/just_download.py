"""
Download benchmark datasets (Cohere Medium 1M, Cohere Large 10M, OpenAI Small 50K)
to llm/database/ directory.

Uses only Python standard library (no third-party dependencies).
"""

import urllib.request
import sys
from pathlib import Path


# All datasets to download
DATASETS = {
    "cohere_medium_1m": [
        "neighbors.parquet",
        "scalar_labels.parquet",
        "shuffle_train.parquet",
        "test.parquet",
    ],
    "cohere_large_10m": [
        "neighbors.parquet",
        "scalar_labels.parquet",
        "shuffle_train-00-of-10.parquet",
        "shuffle_train-01-of-10.parquet",
        "shuffle_train-02-of-10.parquet",
        "shuffle_train-03-of-10.parquet",
        "shuffle_train-04-of-10.parquet",
        "shuffle_train-05-of-10.parquet",
        "shuffle_train-06-of-10.parquet",
        "shuffle_train-07-of-10.parquet",
        "shuffle_train-08-of-10.parquet",
        "shuffle_train-09-of-10.parquet",
        "test.parquet",
    ],
    "openai_small_50k": [
        "neighbors.parquet",
        "scalar_labels.parquet",
        "shuffle_train.parquet",
        "test.parquet",
    ],
}

S3_BASE = "assets.zilliz.com/benchmark"


def download_file(url, filepath):
    """Download file with simple progress display"""
    filepath = Path(filepath)

    if filepath.exists():
        print(f"  [skip] {filepath.name} (already exists)")
        return filepath

    print(f"  Downloading {filepath.name} ...")

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)

        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(
                    f"\r    {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)",
                    end="",
                )

        urllib.request.urlretrieve(url, filepath, reporthook=show_progress)
        print()
        file_size = filepath.stat().st_size / (1024 * 1024)
        print(f"  [done] {filepath.name} ({file_size:.2f} MB)")
        return filepath

    except Exception as e:
        print(f"\n  [FAIL] {filepath.name}: {e}")
        if filepath.exists():
            filepath.unlink()
        return None


def main():
    data_dir = Path(__file__).parent / "database"

    # Allow selecting specific datasets via CLI args
    if len(sys.argv) > 1:
        selected = sys.argv[1:]
        for name in selected:
            if name not in DATASETS:
                print(f"Unknown dataset: {name}")
                print(f"Available: {', '.join(DATASETS.keys())}")
                sys.exit(1)
    else:
        selected = list(DATASETS.keys())

    print(f"\n{'='*60}")
    print(f"  Download Benchmark Datasets")
    print(f"{'='*60}")
    print(f"  Output: {data_dir}")
    print(f"  Datasets: {', '.join(selected)}")
    print(f"{'='*60}\n")

    total_success = 0
    total_fail = 0

    for dataset_name in selected:
        files = DATASETS[dataset_name]
        dataset_dir = data_dir / dataset_name

        print(f"[{dataset_name}] ({len(files)} files)")

        for filename in files:
            url = f"https://{S3_BASE}/{dataset_name}/{filename}"
            result = download_file(url, dataset_dir / filename)
            if result:
                total_success += 1
            else:
                total_fail += 1

        print()

    print(f"{'='*60}")
    print(f"  Done: {total_success} downloaded, {total_fail} failed")
    print(f"{'='*60}\n")

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
