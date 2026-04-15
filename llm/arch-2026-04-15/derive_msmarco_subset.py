#!/usr/bin/env python3
"""Derive a deterministic MS MARCO subset in fvecs/ivecs format.

Pipeline stages:
1. select      - deterministically choose target_count source row ids
2. materialize - write base.fvecs, query.fvecs, base.orig_ids.npy
3. groundtruth - recompute exact top-k ground truth on the derived base
4. all         - run the three stages above and write a manifest

Design goals:
- sequential I/O over large source files
- bounded memory independent of full source size
- contiguous dense local ids in groundtruth.ivecs
- deterministic behavior with explicit tie-breaking
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray


UINT64_MASK = np.uint64((1 << 64) - 1)
DEFAULT_SOURCE_ROOT = Path("/orion/diskann_bench/msmarco_websearch_simans")
DEFAULT_OUTPUT_DIR = Path("llm/database/msmarco_websearch_20m_det_v1")
DEFAULT_TARGET_COUNT = 20_000_000
DEFAULT_K = 100
DEFAULT_SELECTION_CHUNK = 5_000_000
DEFAULT_SOURCE_BLOCK = 250_000
DEFAULT_DERIVED_BASE_BLOCK = 20_000
DEFAULT_GT_QUERY_BATCH = 128
DEFAULT_SEED = 20_260_415
DEFAULT_SPEC_VERSION = "msmarco_websearch_20m_det_v1"


class ArgsNamespace(Protocol):
    command: str
    source_root: Path
    output_dir: Path
    target_count: int
    topk: int
    seed: int
    spec_version: str
    dataset_version: str
    selection_chunk_size: int
    source_block_size: int
    derived_base_block_size: int
    gt_query_batch_size: int
    source_count_limit: int | None
    temp_dir: Path | None
    compute_checksums: bool


@dataclass(frozen=True)
class Config:
    command: str
    source_root: Path
    output_dir: Path
    target_count: int
    topk: int
    seed: int
    spec_version: str
    dataset_version: str
    selection_chunk_size: int
    source_block_size: int
    derived_base_block_size: int
    gt_query_batch_size: int
    source_count_limit: int | None
    temp_dir: Path | None
    compute_checksums: bool


@dataclass(frozen=True)
class VectorFileInfo:
    path: Path
    count: int
    dim: int
    payload_bytes: int


@dataclass(frozen=True)
class DatasetPaths:
    output_dir: Path
    selected_ids_path: Path
    selection_manifest_path: Path
    base_path: Path
    query_path: Path
    groundtruth_path: Path
    orig_ids_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class SelectionManifest:
    created_at: str
    source_root: str
    source_vectors: str
    source_count: int
    source_dim: int
    target_count: int
    seed: int
    spec_version: str
    dataset_version: str
    salt: int
    selection_chunk_size: int
    source_count_limit: int | None


class BinaryWriteHandle(Protocol):
    def write(self, data: bytes) -> object: ...


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "command",
        choices=["inspect", "select", "materialize", "groundtruth", "manifest", "all"],
        help="Pipeline stage to run",
    )
    _ = parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"MS MARCO source root (default: {DEFAULT_SOURCE_ROOT})",
    )
    _ = parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Derived dataset output dir (default: {DEFAULT_OUTPUT_DIR})",
    )
    _ = parser.add_argument(
        "--target-count",
        type=int,
        default=DEFAULT_TARGET_COUNT,
        help=f"Number of base vectors to keep (default: {DEFAULT_TARGET_COUNT})",
    )
    _ = parser.add_argument(
        "--topk",
        type=int,
        default=DEFAULT_K,
        help=f"Ground-truth top-k depth (default: {DEFAULT_K})",
    )
    _ = parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic sampling seed (default: {DEFAULT_SEED})",
    )
    _ = parser.add_argument(
        "--spec-version",
        default=DEFAULT_SPEC_VERSION,
        help=f"Sampling spec version (default: {DEFAULT_SPEC_VERSION})",
    )
    _ = parser.add_argument(
        "--dataset-version",
        default="msmarco_websearch_simans_full_download_2026-04-15",
        help="Dataset snapshot identifier used in the sampling salt",
    )
    _ = parser.add_argument(
        "--selection-chunk-size",
        type=int,
        default=DEFAULT_SELECTION_CHUNK,
        help=f"Source ids per selection-sort chunk (default: {DEFAULT_SELECTION_CHUNK})",
    )
    _ = parser.add_argument(
        "--source-block-size",
        type=int,
        default=DEFAULT_SOURCE_BLOCK,
        help=f"Source rows per materialization block (default: {DEFAULT_SOURCE_BLOCK})",
    )
    _ = parser.add_argument(
        "--derived-base-block-size",
        type=int,
        default=DEFAULT_DERIVED_BASE_BLOCK,
        help=(
            "Derived base rows per GT search block "
            f"(default: {DEFAULT_DERIVED_BASE_BLOCK})"
        ),
    )
    _ = parser.add_argument(
        "--gt-query-batch-size",
        type=int,
        default=DEFAULT_GT_QUERY_BATCH,
        help=f"Queries per GT search batch (default: {DEFAULT_GT_QUERY_BATCH})",
    )
    _ = parser.add_argument(
        "--source-count-limit",
        type=int,
        default=None,
        help="Optional smaller source-count cap for testing the pipeline",
    )
    _ = parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="Optional directory for temporary selection chunk files",
    )
    _ = parser.add_argument(
        "--compute-checksums",
        action="store_true",
        help="Compute SHA256 checksums when writing the final manifest",
    )

    ns = cast(ArgsNamespace, cast(object, parser.parse_args()))
    return Config(
        command=ns.command,
        source_root=ns.source_root,
        output_dir=ns.output_dir,
        target_count=ns.target_count,
        topk=ns.topk,
        seed=ns.seed,
        spec_version=ns.spec_version,
        dataset_version=ns.dataset_version,
        selection_chunk_size=ns.selection_chunk_size,
        source_block_size=ns.source_block_size,
        derived_base_block_size=ns.derived_base_block_size,
        gt_query_batch_size=ns.gt_query_batch_size,
        source_count_limit=ns.source_count_limit,
        temp_dir=ns.temp_dir,
        compute_checksums=ns.compute_checksums,
    )


def resolve_paths(output_dir: Path) -> DatasetPaths:
    return DatasetPaths(
        output_dir=output_dir,
        selected_ids_path=output_dir / "selection.selected_ids.npy",
        selection_manifest_path=output_dir / "selection_manifest.json",
        base_path=output_dir / "base.fvecs",
        query_path=output_dir / "query.fvecs",
        groundtruth_path=output_dir / "groundtruth.ivecs",
        orig_ids_path=output_dir / "base.orig_ids.npy",
        manifest_path=output_dir / "manifest.json",
    )


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_target(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def atomic_replace(tmp_path: Path, final_path: Path) -> None:
    _ = tmp_path.replace(final_path)


def json_write_atomic(path: Path, payload: object) -> None:
    tmp_path = atomic_target(path)
    _ = tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    atomic_replace(tmp_path, path)


def npy_write_atomic(path: Path, array: NDArray[np.uint64]) -> None:
    tmp_path = atomic_target(path)
    with tmp_path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    atomic_replace(tmp_path, path)


def hash_string_u64(value: str) -> np.uint64:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return np.uint64(int.from_bytes(digest, byteorder="little", signed=False))


def sampling_salt(config: Config) -> np.uint64:
    return (
        np.uint64(config.seed)
        ^ hash_string_u64(config.dataset_version)
        ^ hash_string_u64(config.spec_version)
    )


def splitmix64(x: NDArray[np.uint64]) -> NDArray[np.uint64]:
    z = (x + np.uint64(0x9E3779B97F4A7C15)) & UINT64_MASK
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & UINT64_MASK
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & UINT64_MASK
    return (z ^ (z >> np.uint64(31))) & UINT64_MASK


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_default_vectors_info(path: Path) -> VectorFileInfo:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        header = handle.read(8)
    if len(header) != 8:
        raise RuntimeError(f"File too short for DEFAULT vector header: {path}")
    count = int.from_bytes(header[:4], byteorder="little", signed=True)
    dim = int.from_bytes(header[4:], byteorder="little", signed=True)
    if count <= 0 or dim <= 0:
        raise RuntimeError(f"Invalid DEFAULT vector header in {path}: count={count}, dim={dim}")
    payload_bytes = path.stat().st_size - 8
    expected_payload_bytes = count * dim * 4
    if payload_bytes != expected_payload_bytes:
        raise RuntimeError(
            "DEFAULT payload size mismatch for "
            + str(path)
            + ": "
            + "got "
            + str(payload_bytes)
            + ", expected "
            + str(expected_payload_bytes)
        )
    return VectorFileInfo(path=path, count=count, dim=dim, payload_bytes=payload_bytes)


def limited_count(info: VectorFileInfo, count_limit: int | None) -> int:
    if count_limit is None:
        return info.count
    return min(info.count, count_limit)


def read_all_default_vectors(path: Path) -> NDArray[np.float32]:
    info = read_default_vectors_info(path)
    with path.open("rb") as handle:
        _ = handle.read(8)
        raw = handle.read()
    data = np.frombuffer(raw, dtype="<f4")
    return data.reshape(info.count, info.dim)


def iter_default_vector_blocks(
    path: Path,
    dim: int,
    total_count: int,
    block_size: int,
) -> Iterator[tuple[int, NDArray[np.float32]]]:
    emitted = 0
    with path.open("rb") as handle:
        _ = handle.read(8)
        while emitted < total_count:
            current = min(block_size, total_count - emitted)
            raw = handle.read(current * dim * 4)
            if len(raw) != current * dim * 4:
                raise RuntimeError(
                    f"Short read while scanning {path}: expected {current * dim * 4}, got {len(raw)}"
                )
            block = np.frombuffer(raw, dtype="<f4").reshape(current, dim)
            yield emitted, block
            emitted += current


def read_fvecs_info(path: Path) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        raw_dim = handle.read(4)
    if len(raw_dim) != 4:
        raise RuntimeError(f"File too short for fvecs header: {path}")
    dim = int.from_bytes(raw_dim, byteorder="little", signed=True)
    if dim <= 0:
        raise RuntimeError(f"Invalid fvecs dim in {path}: {dim}")
    file_size = path.stat().st_size
    bytes_per_vec = 4 + dim * 4
    if file_size % bytes_per_vec != 0:
        raise RuntimeError(
            f"fvecs file size mismatch for {path}: file_size={file_size}, bytes_per_vec={bytes_per_vec}"
        )
    return dim, file_size // bytes_per_vec


def iter_fvecs_blocks(path: Path, block_size: int) -> Iterator[tuple[int, NDArray[np.float32]]]:
    dim, count = read_fvecs_info(path)
    bytes_per_vec = 4 + dim * 4
    emitted = 0
    with path.open("rb") as handle:
        while emitted < count:
            current = min(block_size, count - emitted)
            raw = handle.read(current * bytes_per_vec)
            if len(raw) != current * bytes_per_vec:
                raise RuntimeError(
                    f"Short read while scanning {path}: expected {current * bytes_per_vec}, got {len(raw)}"
                )
            ints = np.frombuffer(raw, dtype="<i4").reshape(current, dim + 1)
            if not np.all(ints[:, 0] == dim):
                raise RuntimeError(f"Per-row dim prefix mismatch while scanning {path}")
            block = ints[:, 1:].view("<f4")
            yield emitted, block
            emitted += current


def normalize_rows(x: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = cast(NDArray[np.float32], np.linalg.norm(x, axis=1, keepdims=True))
    safe = cast(NDArray[np.float32], np.where(norms > 0, norms, np.float32(1.0)))
    return cast(NDArray[np.float32], x / safe.astype(np.float32))


def pack_fvecs_batch(x: NDArray[np.float32], dim: int) -> NDArray[np.int32]:
    rows = x.shape[0]
    out = np.empty((rows, dim + 1), dtype=np.int32)
    out[:, 0] = dim
    out[:, 1:] = x.view(np.int32).reshape(rows, dim)
    return out


def write_fvecs_batch(handle: BinaryWriteHandle, x: NDArray[np.float32], dim: int) -> None:
    records = pack_fvecs_batch(x, dim)
    _ = handle.write(records.tobytes())


def write_ivecs_atomic(path: Path, ids: NDArray[np.int32]) -> None:
    rows, k = ids.shape
    tmp_path = atomic_target(path)
    out = np.empty((rows, k + 1), dtype=np.int32)
    out[:, 0] = k
    out[:, 1:] = ids
    out.tofile(tmp_path)
    atomic_replace(tmp_path, path)


def stable_sort_topk(
    scores: NDArray[np.float32],
    ids: NDArray[np.int64],
    topk: int,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    id_order = np.argsort(ids, axis=1, kind="stable")
    ids_by_id = np.take_along_axis(ids, id_order, axis=1)
    scores_by_id = np.take_along_axis(scores, id_order, axis=1)
    score_order = np.argsort(-scores_by_id, axis=1, kind="stable")[:, :topk]
    final_scores = np.take_along_axis(scores_by_id, score_order, axis=1)
    final_ids = np.take_along_axis(ids_by_id, score_order, axis=1)
    return final_scores, final_ids


def update_topk(
    top_scores: NDArray[np.float32],
    top_ids: NDArray[np.int64],
    cand_scores: NDArray[np.float32],
    cand_ids: NDArray[np.int64],
    topk: int,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    merged_scores = np.concatenate([top_scores, cand_scores], axis=1)
    merged_ids = np.concatenate([top_ids, cand_ids], axis=1)
    return stable_sort_topk(merged_scores, merged_ids, topk)


def inspect_source(config: Config) -> dict[str, object]:
    base_info = read_default_vectors_info(config.source_root / "passage_vectors" / "vectors.bin")
    query_info = read_default_vectors_info(config.source_root / "query_vectors" / "vectors.bin")
    source_count = limited_count(base_info, config.source_count_limit)
    if config.target_count > source_count:
        raise RuntimeError(
            f"target_count {config.target_count} exceeds usable source_count {source_count}"
        )
    return {
        "source_root": str(config.source_root),
        "base_vectors": {
            "count": base_info.count,
            "usable_count": source_count,
            "dim": base_info.dim,
            "payload_bytes": base_info.payload_bytes,
        },
        "query_vectors": {
            "count": query_info.count,
            "dim": query_info.dim,
            "payload_bytes": query_info.payload_bytes,
        },
        "target_count": config.target_count,
        "topk": config.topk,
        "seed": config.seed,
        "spec_version": config.spec_version,
        "dataset_version": config.dataset_version,
        "sampling_salt": int(sampling_salt(config)),
    }


def write_selection_chunk(
    path: Path,
    start: int,
    stop: int,
    salt: np.uint64,
) -> None:
    ids = np.arange(start, stop, dtype=np.uint64)
    hashes = splitmix64(ids ^ salt)
    order = cast(NDArray[np.intp], np.lexsort((ids, hashes)))
    records = np.empty((stop - start, 2), dtype=np.uint64)
    records[:, 0] = hashes[order]
    records[:, 1] = ids[order]
    with path.open("wb") as handle:
        np.save(handle, records, allow_pickle=False)


def merge_selection_chunks(chunk_paths: Sequence[Path], target_count: int) -> NDArray[np.uint64]:
    arrays = [cast(NDArray[np.uint64], np.load(path, mmap_mode="r")) for path in chunk_paths]
    positions = [0 for _ in arrays]
    heap: list[tuple[int, int, int]] = []
    for chunk_idx, array in enumerate(arrays):
        if array.shape[0] == 0:
            continue
        first_hash = int(np.asarray(array[0, 0]).item())
        first_id = int(np.asarray(array[0, 1]).item())
        heap.append((first_hash, first_id, chunk_idx))
    heapq.heapify(heap)

    selected_ids = np.empty(target_count, dtype=np.uint64)
    emitted = 0
    while emitted < target_count:
        if not heap:
            raise RuntimeError(
                f"Selection merge exhausted early after {emitted} ids; expected {target_count}"
            )
        _, row_id, chunk_idx = heapq.heappop(heap)
        selected_ids[emitted] = np.uint64(row_id)
        emitted += 1
        positions[chunk_idx] += 1
        pos = positions[chunk_idx]
        array = arrays[chunk_idx]
        if pos < array.shape[0]:
            next_hash = int(np.asarray(array[pos, 0]).item())
            next_id = int(np.asarray(array[pos, 1]).item())
            heapq.heappush(heap, (next_hash, next_id, chunk_idx))

    selected_ids.sort()
    return selected_ids


def run_select(config: Config, paths: DatasetPaths) -> None:
    ensure_output_dir(paths.output_dir)
    source_info = read_default_vectors_info(config.source_root / "passage_vectors" / "vectors.bin")
    usable_count = limited_count(source_info, config.source_count_limit)
    if config.target_count > usable_count:
        raise RuntimeError(
            f"target_count {config.target_count} exceeds usable source_count {usable_count}"
        )

    selection_manifest = SelectionManifest(
        created_at=datetime.now(timezone.utc).isoformat(),
        source_root=str(config.source_root),
        source_vectors=str(source_info.path),
        source_count=usable_count,
        source_dim=source_info.dim,
        target_count=config.target_count,
        seed=config.seed,
        spec_version=config.spec_version,
        dataset_version=config.dataset_version,
        salt=int(sampling_salt(config)),
        selection_chunk_size=config.selection_chunk_size,
        source_count_limit=config.source_count_limit,
    )
    json_write_atomic(paths.selection_manifest_path, asdict(selection_manifest))

    temp_parent = config.temp_dir if config.temp_dir is not None else paths.output_dir / "selection_tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    salt = sampling_salt(config)

    chunk_paths: list[Path] = []
    for chunk_idx, start in enumerate(range(0, usable_count, config.selection_chunk_size)):
        stop = min(start + config.selection_chunk_size, usable_count)
        chunk_path = temp_parent / f"selection_chunk_{chunk_idx:04d}.npy"
        write_selection_chunk(chunk_path, start, stop, salt)
        chunk_paths.append(chunk_path)

    selected_ids = merge_selection_chunks(chunk_paths, config.target_count)
    npy_write_atomic(paths.selected_ids_path, selected_ids)

    for chunk_path in chunk_paths:
        chunk_path.unlink()
    if config.temp_dir is None:
        temp_parent.rmdir()


def load_selected_ids(paths: DatasetPaths) -> NDArray[np.uint64]:
    if not paths.selected_ids_path.exists():
        raise FileNotFoundError(
            f"Selected ids not found: {paths.selected_ids_path}. Run 'select' first."
        )
    return cast(NDArray[np.uint64], np.load(paths.selected_ids_path, mmap_mode="r"))


def run_materialize(config: Config, paths: DatasetPaths) -> None:
    ensure_output_dir(paths.output_dir)
    selected_ids = load_selected_ids(paths)

    base_info = read_default_vectors_info(config.source_root / "passage_vectors" / "vectors.bin")
    query_info = read_default_vectors_info(config.source_root / "query_vectors" / "vectors.bin")
    usable_count = limited_count(base_info, config.source_count_limit)

    if selected_ids.shape[0] != config.target_count:
        raise RuntimeError(
            f"Selected-id count mismatch: got {selected_ids.shape[0]}, expected {config.target_count}"
        )
    max_selected_id = int(np.asarray(selected_ids[-1]).item())
    if max_selected_id >= usable_count:
        raise RuntimeError(
            f"Selected ids exceed usable source_count {usable_count}: max={max_selected_id}"
        )
    if base_info.dim != query_info.dim:
        raise RuntimeError(
            f"Dimension mismatch between base/query vectors: {base_info.dim} vs {query_info.dim}"
        )

    base_tmp = atomic_target(paths.base_path)
    query_tmp = atomic_target(paths.query_path)
    orig_ids_accumulator = np.empty(config.target_count, dtype=np.uint64)

    emitted = 0
    with base_tmp.open("wb") as base_handle:
        typed_base_handle = cast(BinaryWriteHandle, cast(object, base_handle))
        for block_start, block in iter_default_vector_blocks(
            base_info.path,
            dim=base_info.dim,
            total_count=usable_count,
            block_size=config.source_block_size,
        ):
            block_end = block_start + block.shape[0]
            lo = int(np.searchsorted(selected_ids, np.uint64(block_start), side="left"))
            hi = int(np.searchsorted(selected_ids, np.uint64(block_end), side="left"))
            if hi == lo:
                continue
            local_indices = (selected_ids[lo:hi] - np.uint64(block_start)).astype(np.int64)
            chosen = block[local_indices]
            write_fvecs_batch(
                typed_base_handle,
                np.ascontiguousarray(chosen, dtype=np.float32),
                base_info.dim,
            )
            orig_ids_accumulator[emitted : emitted + chosen.shape[0]] = selected_ids[lo:hi]
            emitted += chosen.shape[0]

    if emitted != config.target_count:
        raise RuntimeError(f"Materialized {emitted} base vectors, expected {config.target_count}")

    queries = read_all_default_vectors(query_info.path)
    with query_tmp.open("wb") as query_handle:
        typed_query_handle = cast(BinaryWriteHandle, cast(object, query_handle))
        write_fvecs_batch(
            typed_query_handle,
            np.ascontiguousarray(queries, dtype=np.float32),
            query_info.dim,
        )

    npy_write_atomic(paths.orig_ids_path, orig_ids_accumulator)
    atomic_replace(base_tmp, paths.base_path)
    atomic_replace(query_tmp, paths.query_path)


def run_groundtruth(config: Config, paths: DatasetPaths) -> None:
    if not paths.base_path.exists() or not paths.query_path.exists():
        raise FileNotFoundError("base.fvecs and query.fvecs must exist before recomputing ground truth")

    base_dim, base_count = read_fvecs_info(paths.base_path)
    query_dim, query_count = read_fvecs_info(paths.query_path)
    if base_dim != query_dim:
        raise RuntimeError(f"Dimension mismatch in derived dataset: base={base_dim}, query={query_dim}")
    if base_count != config.target_count:
        raise RuntimeError(
            f"Derived base count mismatch: base.fvecs has {base_count}, expected {config.target_count}"
        )
    if query_count < 100:
        raise RuntimeError(f"Query count too small for current benchmark assumptions: {query_count}")

    query_blocks = [
        block.copy() for _, block in iter_fvecs_blocks(paths.query_path, block_size=max(query_count, 1))
    ]
    query_vectors = np.ascontiguousarray(np.vstack(query_blocks), dtype=np.float32)
    query_vectors = normalize_rows(query_vectors)

    top_scores = np.full((query_count, config.topk), -np.inf, dtype=np.float32)
    top_ids = np.full((query_count, config.topk), -1, dtype=np.int64)

    for base_offset, base_block in iter_fvecs_blocks(paths.base_path, block_size=config.derived_base_block_size):
        normalized_block = normalize_rows(np.ascontiguousarray(base_block.copy(), dtype=np.float32))
        block_ids = np.arange(
            base_offset,
            base_offset + normalized_block.shape[0],
            dtype=np.int64,
        )

        for q_start in range(0, query_count, config.gt_query_batch_size):
            q_end = min(q_start + config.gt_query_batch_size, query_count)
            q_batch = query_vectors[q_start:q_end]
            scores = cast(NDArray[np.float32], np.matmul(q_batch, normalized_block.T))

            if normalized_block.shape[0] > config.topk:
                part = np.argpartition(-scores, kth=config.topk - 1, axis=1)[:, : config.topk]
                cand_scores = np.take_along_axis(scores, part, axis=1)
                cand_ids = part.astype(np.int64) + np.int64(base_offset)
            else:
                cand_scores = scores
                cand_ids = np.broadcast_to(block_ids[None, :], scores.shape).copy()

            cand_scores, cand_ids = stable_sort_topk(cand_scores, cand_ids, config.topk)
            merged_scores, merged_ids = update_topk(
                top_scores[q_start:q_end],
                top_ids[q_start:q_end],
                cand_scores,
                cand_ids,
                config.topk,
            )
            top_scores[q_start:q_end] = merged_scores
            top_ids[q_start:q_end] = merged_ids

    if np.any(top_ids < 0):
        raise RuntimeError("Ground-truth recomputation produced invalid ids")
    if np.any(top_ids >= np.int64(base_count)):
        raise RuntimeError("Ground-truth recomputation produced out-of-range ids")

    gt_ids = top_ids.astype(np.int32)
    write_ivecs_atomic(paths.groundtruth_path, gt_ids)


def write_manifest(config: Config, paths: DatasetPaths) -> None:
    base_dim, base_count = read_fvecs_info(paths.base_path)
    query_dim, query_count = read_fvecs_info(paths.query_path)
    gt_raw = cast(NDArray[np.int32], np.fromfile(paths.groundtruth_path, dtype="<i4"))
    gt_k = int(np.asarray(gt_raw[0]).item())
    gt_count = gt_raw.size // (gt_k + 1)
    if query_count != gt_count:
        raise RuntimeError(f"Query/GT row mismatch: query_count={query_count}, gt_count={gt_count}")

    file_info: dict[str, dict[str, object]] = {
        "base.fvecs": {
            "path": str(paths.base_path),
            "count": base_count,
            "dim": base_dim,
            "size_bytes": paths.base_path.stat().st_size,
        },
        "query.fvecs": {
            "path": str(paths.query_path),
            "count": query_count,
            "dim": query_dim,
            "size_bytes": paths.query_path.stat().st_size,
        },
        "groundtruth.ivecs": {
            "path": str(paths.groundtruth_path),
            "count": gt_count,
            "k": gt_k,
            "size_bytes": paths.groundtruth_path.stat().st_size,
        },
        "base.orig_ids.npy": {
            "path": str(paths.orig_ids_path),
            "size_bytes": paths.orig_ids_path.stat().st_size,
        },
        "selection.selected_ids.npy": {
            "path": str(paths.selected_ids_path),
            "size_bytes": paths.selected_ids_path.stat().st_size,
        },
    }
    if config.compute_checksums:
        for item in file_info.values():
            item["sha256"] = file_sha256(Path(cast(str, item["path"])))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(config.source_root),
        "dataset_version": config.dataset_version,
        "spec_version": config.spec_version,
        "seed": config.seed,
        "sampling_rule": "lowest splitmix64(source_row_id XOR salt)",
        "sampling_salt": int(sampling_salt(config)),
        "canonical_source_id": "original row index in source passage_vectors/vectors.bin",
        "output_order": "selected source row ids sorted ascending before materialization",
        "metric": "cosine via L2 normalize + inner product",
        "groundtruth": {
            "type": "exact",
            "k": config.topk,
            "id_space": "dense local row ids in base.fvecs",
            "tie_break": "smaller local id wins on equal score",
        },
        "source_count_limit": config.source_count_limit,
        "files": file_info,
    }
    json_write_atomic(paths.manifest_path, manifest)


def require_files(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))


def main() -> int:
    config = parse_args()
    paths = resolve_paths(config.output_dir)

    if config.command == "inspect":
        print(json.dumps(inspect_source(config), indent=2, sort_keys=True))
        return 0

    ensure_output_dir(paths.output_dir)

    if config.command == "select":
        run_select(config, paths)
        return 0

    if config.command == "materialize":
        require_files([paths.selected_ids_path])
        run_materialize(config, paths)
        return 0

    if config.command == "groundtruth":
        require_files([paths.base_path, paths.query_path])
        run_groundtruth(config, paths)
        return 0

    if config.command == "manifest":
        require_files(
            [
                paths.selected_ids_path,
                paths.base_path,
                paths.query_path,
                paths.groundtruth_path,
                paths.orig_ids_path,
            ]
        )
        write_manifest(config, paths)
        return 0

    if config.command == "all":
        run_select(config, paths)
        run_materialize(config, paths)
        run_groundtruth(config, paths)
        write_manifest(config, paths)
        return 0

    raise RuntimeError(f"Unhandled command: {config.command}")


if __name__ == "__main__":
    raise SystemExit(main())
