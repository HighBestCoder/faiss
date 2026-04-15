# Deterministic 20M MS MARCO Derived Dataset Rules

This document defines the required rules for deriving a reviewable 20M dataset from the downloaded MS MARCO Web Search SimANS release.

## Goal

Produce a derived dataset that:

- uses deterministic sampling
- supports valid `recall@100`
- matches the existing local dataset workflow
- is consumable by the current benchmark binaries without code changes

## Required Output Format

The derived dataset must be materialized as a directory containing exactly:

- `base.fvecs`
- `query.fvecs`
- `groundtruth.ivecs`

The binary format must remain standard FAISS-compatible `fvecs/ivecs`:

- `fvecs`: per row, little-endian `int32 dim` followed by `dim` float32 values
- `ivecs`: per row, little-endian `int32 k` followed by `k` int32 ids

Stored vectors must remain raw float32 embeddings. Do not pre-normalize vectors on disk. Benchmark code performs L2 normalization at load/index time and uses inner product for cosine-style evaluation.

## Canonical Source Definition

- Source dataset: downloaded MS MARCO Web Search SimANS release
- Base corpus source: `passage_vectors/vectors.bin`
- Query source: `query_vectors/vectors.bin`
- Canonical source id for a document vector is its original row index in the source base vector file
- If metadata is preserved, it is auxiliary only; benchmark ids are not metadata ids

## Deterministic Sampling Rule

The 20M subset must be selected deterministically from the full source corpus.

Required rule:

1. For every source row id, compute a stable hash using:
   - source row id
   - dataset version or source snapshot identifier
   - sampling spec version
   - fixed seed
2. Select the 20,000,000 rows with the smallest hash values.
3. After selection, sort the chosen source row ids in ascending original row-id order before materializing the derived base dataset.

Disallowed shortcuts:

- taking the first 20M rows as the primary dataset definition
- using non-deterministic RNG without a pinned seed and persisted selection
- relying on current file iteration order unless it is explicitly frozen as part of the dataset spec

## Dense ID Semantics

The benchmark expects implicit local ids equal to row order in `base.fvecs`.

Therefore:

- row `i` in `base.fvecs` must correspond to dense local id `i`
- `groundtruth.ivecs` must store these dense local ids
- original full-dataset ids must not appear in `groundtruth.ivecs`

Every derived dataset must include an audit sidecar mapping from dense local id to original source row id.

Recommended sidecar name:

- `base.orig_ids`

## Query Rules

- Query vectors must be reused from the official query package unchanged unless a different query benchmark is explicitly specified.
- Query vectors must be converted to `query.fvecs` in the same dimensionality as the derived base set.
- Query count must match the number of rows in `groundtruth.ivecs` exactly.

## Ground Truth Rules

Valid `recall@100` requires exact ground truth on the derived 20M base set.

Therefore:

- do not reuse the official full-dataset `truth.txt`
- do not filter the full-dataset truth to retained ids and call it final ground truth
- do not use ANN-generated neighbors as ground truth

Required rule:

1. Build the final 20M derived base set first.
2. Apply the same cosine semantics as the benchmark path (`L2 normalize + inner product`).
3. Recompute exact top-100 neighbors for every query against only the derived 20M base set.
4. Write the resulting top-100 dense local ids to `groundtruth.ivecs`.

Ground-truth requirements:

- one row per query
- at least 100 ids per row
- ids must be in the dense local id space of the derived `base.fvecs`
- tie-breaking must be deterministic; if equal scores occur, smaller dense local id wins

## Validation Gates

Before a derived dataset is accepted, all of the following must pass:

1. `base.fvecs` and `query.fvecs` have identical dimensions.
2. `groundtruth.ivecs` row count equals query count.
3. Every GT row contains at least 100 ids.
4. All GT ids are within `[0, nb)` for the final derived base size.
5. `base.orig_ids` length equals the number of base vectors.
6. Re-running the sampling step with the same inputs produces identical selected ids.
7. Spot-check exact search results against `groundtruth.ivecs` for sample queries.
8. Current benchmark binaries can load the dataset without code changes.

## Manifest Requirements

Each derived dataset must include a manifest recording at least:

- source dataset path/version/checksums
- source vector count
- target vector count (`20,000,000`)
- dimension
- metric semantics (`cosine via L2 normalize + inner product`)
- sampling rule
- seed
- sampling spec version
- canonical source id definition
- output ordering rule
- ground-truth generation method
- tie-break rule
- file checksums for output artifacts
- creation timestamp

## Non-Goals / Prohibited Simplifications

The following are not acceptable as the final reviewable 20M dataset definition:

- first-20M prefix subset as the primary benchmark artifact
- unchanged reuse of full-dataset truth
- filtered full-dataset truth presented as exact recall ground truth
- output in raw SPTAG format only
- changing benchmark code to match the dataset instead of matching the dataset to the current workflow

## Naming Recommendation

Recommended dataset identifier:

- `msmarco_websearch_20m_det_v1`

This name should correspond to a single frozen manifest, fixed sampling rule, and fixed GT generation method.
