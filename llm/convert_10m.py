"""
Convert cohere_large_10m parquet to fvecs/ivecs format.
Memory-efficient: reads parquet in chunks and writes directly.
"""
import pyarrow.parquet as pq
from pathlib import Path
import struct
import sys
import numpy as np


def main():
    src_dir = Path("llm/database/cohere_large_10m")
    out_dir = Path("llm/database/cohere_10m")
    out_dir.mkdir(parents=True, exist_ok=True)

    num_shards = 10
    d = 768

    # Write base vectors (one shard at a time, row by row)
    print("Converting base vectors (10 shards)...")
    total = 0
    dim_bytes = struct.pack('<i', d)
    with open(out_dir / "base.fvecs", 'wb') as f:
        for i in range(num_shards):
            fname = src_dir / f"shuffle_train-{i:02d}-of-{num_shards:02d}.parquet"
            print(f"  Shard {i}: {fname.name}...", end=" ", flush=True)
            table = pq.read_table(fname, columns=['emb'])
            col = table.column('emb')
            count = 0
            # Process in batches of 10000 to reduce per-row overhead
            batch_size = 10000
            for start in range(0, len(col), batch_size):
                end = min(start + batch_size, len(col))
                batch = np.array([col[j].as_py() for j in range(start, end)], dtype=np.float32)
                for row in batch:
                    f.write(dim_bytes)
                    f.write(row.tobytes())
                count += end - start
            total += count
            print(f"{count} vectors (total: {total})")
            del table, col

    print(f"  base.fvecs: {total} vectors")

    # Write query vectors
    print("\nConverting query vectors...")
    table = pq.read_table(src_dir / "test.parquet", columns=['emb'])
    col = table.column('emb')
    with open(out_dir / "query.fvecs", 'wb') as f:
        for j in range(len(col)):
            vec = np.array(col[j].as_py(), dtype=np.float32)
            f.write(dim_bytes)
            f.write(vec.tobytes())
    print(f"  query.fvecs: {len(col)} vectors")
    del table, col

    # Write ground truth
    print("\nConverting ground truth...")
    table = pq.read_table(src_dir / "neighbors.parquet")
    col = table.column('neighbors_id')
    with open(out_dir / "groundtruth.ivecs", 'wb') as f:
        for j in range(len(col)):
            ids = np.array(col[j].as_py(), dtype=np.int32)
            f.write(struct.pack('<i', len(ids)))
            f.write(ids.tobytes())
    print(f"  groundtruth.ivecs: {len(col)} queries, k={len(col[0].as_py())}")
    del table, col

    print(f"\nDone! Output: {out_dir}/")


if __name__ == "__main__":
    main()
