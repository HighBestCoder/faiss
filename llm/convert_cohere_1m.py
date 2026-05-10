#!/usr/bin/env python3
"""Convert Cohere 1M parquet -> base.fvecs / query.fvecs / groundtruth.ivecs.

base rows are sorted by id so that neighbors_id (which are global ids
0..N-1) directly index into base.fvecs row numbers.
"""
import os
import sys
import numpy as np
import pyarrow.parquet as pq

SRC = "/db/dataset/cohere/cohere_medium_1m"
DST = "/db/faiss/llm/database/cohere_medium_1m"
K_GT = 100

os.makedirs(DST, exist_ok=True)


def write_fvecs(path, mat):
    n, d = mat.shape
    out = np.empty((n, d + 1), dtype=np.float32)
    out[:, 0].view(np.int32)[:] = d
    out[:, 1:] = mat
    out.tofile(path)


def write_ivecs(path, mat):
    n, d = mat.shape
    out = np.empty((n, d + 1), dtype=np.int32)
    out[:, 0] = d
    out[:, 1:] = mat
    out.tofile(path)


def emb_to_matrix(col):
    arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    flat = np.asarray(arr.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    n = len(col)
    d = flat.size // n
    return flat.reshape(n, d), d


print(">> reading shuffle_train.parquet (1M base) ...", flush=True)
t = pq.read_table(f"{SRC}/shuffle_train.parquet")
ids = t["id"].to_numpy()
emb, d = emb_to_matrix(t["emb"])
print(f"   rows={len(ids)} dim={d}", flush=True)
order = np.argsort(ids, kind="stable")
assert ids[order[0]] == 0 and ids[order[-1]] == len(ids) - 1
base = emb[order]
print(">> writing base.fvecs ...", flush=True)
write_fvecs(f"{DST}/base.fvecs", base)
del t, emb, base, order, ids

print(">> reading test.parquet (1k queries) ...", flush=True)
t = pq.read_table(f"{SRC}/test.parquet")
qemb, qd = emb_to_matrix(t["emb"])
assert qd == d, f"query dim {qd} != base dim {d}"
print(f"   rows={qemb.shape[0]} dim={qd}", flush=True)
print(">> writing query.fvecs ...", flush=True)
write_fvecs(f"{DST}/query.fvecs", qemb)
del t, qemb

print(">> reading neighbors.parquet (1k * 1000) ...", flush=True)
t = pq.read_table(f"{SRC}/neighbors.parquet")
nb_lists = t["neighbors_id"].to_pylist()
gt = np.empty((len(nb_lists), K_GT), dtype=np.int32)
for i, row in enumerate(nb_lists):
    gt[i] = row[:K_GT]
print(f"   rows={gt.shape[0]} k={gt.shape[1]}  min={gt.min()} max={gt.max()}", flush=True)
print(">> writing groundtruth.ivecs ...", flush=True)
write_ivecs(f"{DST}/groundtruth.ivecs", gt)

print(">> done. files:")
for f in ("base.fvecs", "query.fvecs", "groundtruth.ivecs"):
    p = f"{DST}/{f}"
    print(f"   {p}  {os.path.getsize(p)} B")
