"""
Convert cohere_large_10m parquet to fvecs/ivecs format.
Also generate a 20M dataset by duplicating with offset IDs.
"""
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import struct
import sys


def write_fvecs(fname, data):
    """Write float32 matrix to fvecs format."""
    n, d = data.shape
    with open(fname, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('<i', d))
            f.write(data[i].tobytes())
    print(f"  Wrote {fname}: {n} vectors, {d} dim")


def write_ivecs(fname, data):
    """Write int32 matrix to ivecs format."""
    n, d = data.shape
    data = data.astype(np.int32)
    with open(fname, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('<i', d))
            f.write(data[i].tobytes())
    print(f"  Wrote {fname}: {n} vectors, {d} dim")


def load_train_shards(src_dir, num_shards=10):
    """Load all train shards and concatenate."""
    all_embs = []
    for i in range(num_shards):
        fname = src_dir / f"shuffle_train-{i:02d}-of-{num_shards:02d}.parquet"
        print(f"  Loading {fname.name}...")
        t = pq.read_table(fname, columns=['emb'])
        embs = np.array([row.as_py() for row in t.column('emb')], dtype=np.float32)
        all_embs.append(embs)
        print(f"    {embs.shape[0]} vectors loaded")
    return np.concatenate(all_embs, axis=0)


def main():
    src_dir = Path("llm/database/cohere_large_10m")

    # === 10M dataset ===
    out_10m = Path("llm/database/cohere_10m")
    out_10m.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Converting cohere_large_10m to fvecs/ivecs")
    print("=" * 60)

    # Load base vectors
    print("\nLoading base vectors (10 shards)...")
    base = load_train_shards(src_dir)
    print(f"  Total: {base.shape[0]} vectors, {base.shape[1]} dim")

    # Load query vectors
    print("\nLoading query vectors...")
    t_test = pq.read_table(src_dir / "test.parquet", columns=['emb'])
    query = np.array([row.as_py() for row in t_test.column('emb')], dtype=np.float32)
    print(f"  {query.shape[0]} queries, {query.shape[1]} dim")

    # Load ground truth
    print("\nLoading ground truth...")
    t_gt = pq.read_table(src_dir / "neighbors.parquet")
    gt_lists = [row.as_py() for row in t_gt.column('neighbors_id')]
    gt_k = len(gt_lists[0])
    gt = np.array(gt_lists, dtype=np.int32)
    print(f"  {gt.shape[0]} queries, k={gt_k}")

    # Write 10M fvecs/ivecs
    print(f"\nWriting 10M dataset to {out_10m}/")
    write_fvecs(out_10m / "base.fvecs", base)
    write_fvecs(out_10m / "query.fvecs", query)
    write_ivecs(out_10m / "groundtruth.ivecs", gt)

    # === 20M dataset (duplicate base with shifted IDs) ===
    out_20m = Path("llm/database/cohere_20m")
    out_20m.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating 20M dataset to {out_20m}/")
    print("  Concatenating base vectors (10M + 10M)...")
    base_20m = np.concatenate([base, base], axis=0)
    print(f"  Total: {base_20m.shape[0]} vectors")

    write_fvecs(out_20m / "base.fvecs", base_20m)
    write_fvecs(out_20m / "query.fvecs", query)
    # Ground truth stays the same (original 10M IDs are still valid in first half)
    write_ivecs(out_20m / "groundtruth.ivecs", gt)

    print("\nDone!")
    print(f"  10M: {out_10m}/")
    print(f"  20M: {out_20m}/")


if __name__ == "__main__":
    main()
