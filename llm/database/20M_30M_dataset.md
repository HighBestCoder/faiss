# 20M / 30M Dataset Notes

## Current Status

- 20M derived dataset: generated
- 30M derived dataset: not generated yet

## 20M Derived Dataset

Dataset directory:

- `/ceph/faiss-dev/llm/database/msmarco_websearch_20m_det_v1/`

Main files:

- `/ceph/faiss-dev/llm/database/msmarco_websearch_20m_det_v1/base.fvecs`
- `/ceph/faiss-dev/llm/database/msmarco_websearch_20m_det_v1/query.fvecs`
- `/ceph/faiss-dev/llm/database/msmarco_websearch_20m_det_v1/groundtruth.ivecs`

Sidecar / manifest files:

- `/ceph/faiss-dev/llm/database/msmarco_websearch_20m_det_v1/base.orig_ids.npy`
- `/ceph/faiss-dev/llm/database/msmarco_websearch_20m_det_v1/selection.selected_ids.npy`
- `/ceph/faiss-dev/llm/database/msmarco_websearch_20m_det_v1/selection_manifest.json`
- `/ceph/faiss-dev/llm/database/msmarco_websearch_20m_det_v1/manifest.json`

## Original Downloaded MS MARCO Data

Root directory:

- `/orion/diskann_bench/msmarco_websearch_simans/`

Passage vectors:

- `/orion/diskann_bench/msmarco_websearch_simans/passage_vectors/vectors.bin`
- `/orion/diskann_bench/msmarco_websearch_simans/passage_vectors/meta.bin`
- `/orion/diskann_bench/msmarco_websearch_simans/passage_vectors/metaidx.bin`

Query vectors:

- `/orion/diskann_bench/msmarco_websearch_simans/query_vectors/vectors.bin`
- `/orion/diskann_bench/msmarco_websearch_simans/query_vectors/meta.bin`
- `/orion/diskann_bench/msmarco_websearch_simans/query_vectors/metaidx.bin`

Official truth and related files:

- `/orion/diskann_bench/msmarco_websearch_simans/truth.txt`
- `/orion/diskann_bench/msmarco_websearch_simans/queries_test.tsv`
- `/orion/diskann_bench/msmarco_websearch_simans/qrels_test.tsv`

## Format Notes

- The 20M derived dataset is in FAISS-compatible `fvecs/ivecs` format.
- The original downloaded MS MARCO data is in SPTAG-style binary format.
