#!/usr/bin/env python3
"""
Download the official MS MARCO Web Search SimANS vector release into
/orion/diskann_bench with resumable per-file downloads.

Important: the official dataset is published for non-commercial research use.
This script requires --accept-license before it will download anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from typing import Protocol, TypedDict, cast


DEFAULT_TARGET_DIR = Path("/orion/diskann_bench/msmarco_websearch_simans")
OFFICIAL_README = "https://github.com/microsoft/MS-MARCO-Web-Search"
OFFICIAL_LANDING = (
    "https://www.microsoft.com/en-us/research/publication/"
    "ms-marco-web-search-a-large-scale-information-rich-web-dataset-with-"
    "millions-of-real-click-labels/"
)


@dataclass(frozen=True)
class DownloadItem:
    key: str
    relative_path: str
    url: str
    description: str
    required_for: str


@dataclass(frozen=True)
class Config:
    target_dir: Path
    mode: str
    include_doc_hash: bool
    dry_run: bool
    accept_license: bool
    reserve_gb: float
    connect_timeout: int
    retries: int


class OfficialSources(TypedDict):
    landing: str
    readme: str


class ManifestFile(TypedDict, total=False):
    key: str
    relative_path: str
    url: str
    description: str
    required_for: str
    expected_size: int | None
    expected_size_human: str
    path: str
    exists: bool
    size: int
    size_human: str
    sha256: str


class Manifest(TypedDict):
    dataset: str
    retrieved_at: str
    mode: str
    include_doc_hash: bool
    target_dir: str
    official_sources: OfficialSources
    license_notice: str
    files: list[ManifestFile]


class ArgsNamespace(Protocol):
    target_dir: Path
    mode: str
    include_doc_hash: bool
    dry_run: bool
    accept_license: bool
    reserve_gb: float
    connect_timeout: int
    retries: int


class UrlOpenResponse(Protocol):
    def __enter__(self) -> "UrlOpenResponse": ...
    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...
    def info(self) -> Message: ...


DOWNLOAD_ITEMS = {
    "passage_vectors_bin": DownloadItem(
        key="passage_vectors_bin",
        relative_path="passage_vectors/vectors.bin",
        url=(
            "https://msmarco.z22.web.core.windows.net/msmarcowebsearch/"
            "vectors/SimANS/passage_vectors/vectors.bin"
        ),
        description="Passage embeddings payload",
        required_for="building/searching the 100M document corpus",
    ),
    "passage_metaidx_bin": DownloadItem(
        key="passage_metaidx_bin",
        relative_path="passage_vectors/metaidx.bin",
        url=(
            "https://msmarco.z22.web.core.windows.net/msmarcowebsearch/"
            "vectors/SimANS/passage_vectors/metaidx.bin"
        ),
        description="Passage embedding metadata index",
        required_for="building/searching the 100M document corpus",
    ),
    "passage_meta_bin": DownloadItem(
        key="passage_meta_bin",
        relative_path="passage_vectors/meta.bin",
        url=(
            "https://msmarco.z22.web.core.windows.net/msmarcowebsearch/"
            "vectors/SimANS/passage_vectors/meta.bin"
        ),
        description="Passage embedding metadata payload",
        required_for="building/searching the 100M document corpus",
    ),
    "query_vectors_bin": DownloadItem(
        key="query_vectors_bin",
        relative_path="query_vectors/vectors.bin",
        url=(
            "https://msmarco.z22.web.core.windows.net/msmarcowebsearch/"
            "vectors/SimANS/query_vectors/vectors.bin"
        ),
        description="Query embeddings payload",
        required_for="official search queries",
    ),
    "query_metaidx_bin": DownloadItem(
        key="query_metaidx_bin",
        relative_path="query_vectors/metaidx.bin",
        url=(
            "https://msmarco.z22.web.core.windows.net/msmarcowebsearch/"
            "vectors/SimANS/query_vectors/metaidx.bin"
        ),
        description="Query embedding metadata index",
        required_for="official search queries",
    ),
    "query_meta_bin": DownloadItem(
        key="query_meta_bin",
        relative_path="query_vectors/meta.bin",
        url=(
            "https://msmarco.z22.web.core.windows.net/msmarcowebsearch/"
            "vectors/SimANS/query_vectors/meta.bin"
        ),
        description="Query embedding metadata payload",
        required_for="official search queries",
    ),
    "truth_txt": DownloadItem(
        key="truth_txt",
        relative_path="truth.txt",
        url=(
            "https://msmarco.z22.web.core.windows.net/msmarcowebsearch/"
            "vectors/SimANS/truth.txt"
        ),
        description="Ground truth for ANN evaluation",
        required_for="recall/evaluation",
    ),
    "queries_test_tsv": DownloadItem(
        key="queries_test_tsv",
        relative_path="queries_test.tsv",
        url=(
            "https://msmarco.z22.web.core.windows.net/msmarcowebsearch/"
            "100M_queries/queries_test.tsv"
        ),
        description="Human-readable test queries",
        required_for="inspecting the official test set",
    ),
    "qrels_test_tsv": DownloadItem(
        key="qrels_test_tsv",
        relative_path="qrels_test.tsv",
        url=(
            "https://msmarco.z22.web.core.windows.net/msmarcowebsearch/"
            "100M_queries/qrels_test.tsv"
        ),
        description="Official qrels file",
        required_for="TREC-style evaluation",
    ),
    "doc_hash_mapping_tsv": DownloadItem(
        key="doc_hash_mapping_tsv",
        relative_path="doc_hash_mapping.tsv",
        url=(
            "https://msmarco.z22.web.core.windows.net/msmarcowebsearch/"
            "100M_queries/doc_hash_mapping.tsv"
        ),
        description="Document hash mapping",
        required_for="mapping to upstream doc ids",
    ),
}


COMPONENT_GROUPS = {
    "eval": [
        "query_vectors_bin",
        "query_metaidx_bin",
        "query_meta_bin",
        "truth_txt",
        "queries_test_tsv",
        "qrels_test_tsv",
    ],
    "full": [
        "passage_vectors_bin",
        "passage_metaidx_bin",
        "passage_meta_bin",
        "query_vectors_bin",
        "query_metaidx_bin",
        "query_meta_bin",
        "truth_txt",
        "queries_test_tsv",
        "qrels_test_tsv",
    ],
}

LICENSE_NOTICE = """\
MS MARCO Web Search is published by Microsoft for non-commercial research use.
Before downloading, confirm that this matches your intended use.

Official sources:
  - {landing}
  - {readme}
""".format(landing=OFFICIAL_LANDING, readme=OFFICIAL_README)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Download the official MS MARCO Web Search SimANS release",
    )
    _ = parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help=f"Destination directory (default: {DEFAULT_TARGET_DIR})",
    )
    _ = parser.add_argument(
        "--mode",
        choices=sorted(COMPONENT_GROUPS),
        default="eval",
        help=(
            "eval = query vectors + truth + query text/qrels; "
            "full = passage vectors + eval files"
        ),
    )
    _ = parser.add_argument(
        "--include-doc-hash",
        action="store_true",
        help="Also download doc_hash_mapping.tsv (about 8.34 GB)",
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded and exit",
    )
    _ = parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Required: acknowledge the official non-commercial research notice",
    )
    _ = parser.add_argument(
        "--reserve-gb",
        type=float,
        default=20.0,
        help="Minimum free-space reserve to keep after each download (default: 20)",
    )
    _ = parser.add_argument(
        "--connect-timeout",
        type=int,
        default=30,
        help="curl connect timeout in seconds (default: 30)",
    )
    _ = parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="curl retry count for transient failures (default: 5)",
    )
    namespace = cast(ArgsNamespace, cast(object, parser.parse_args()))
    return Config(
        target_dir=namespace.target_dir,
        mode=namespace.mode,
        include_doc_hash=namespace.include_doc_hash,
        dry_run=namespace.dry_run,
        accept_license=namespace.accept_license,
        reserve_gb=namespace.reserve_gb,
        connect_timeout=namespace.connect_timeout,
        retries=namespace.retries,
    )


def ensure_curl_available() -> None:
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required but was not found in PATH")


def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def head_content_length(url: str, timeout: int = 30) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    response_cm = cast(
        UrlOpenResponse,
        cast(object, urllib.request.urlopen(request, timeout=timeout)),
    )
    with response_cm as response:
        headers = response.info()
        header = headers.get("Content-Length")
        if header is None:
            return None
        return int(header)


def resolve_items(config: Config) -> list[DownloadItem]:
    keys = list(COMPONENT_GROUPS[config.mode])
    if config.include_doc_hash:
        keys.append("doc_hash_mapping_tsv")
    return [DOWNLOAD_ITEMS[key] for key in keys]


def disk_free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def reserve_bytes_from_config(config: Config) -> int:
    return int(config.reserve_gb * 1024 * 1024 * 1024)


def ensure_space(path: Path, required_bytes: int | None, reserve_bytes: int) -> None:
    if required_bytes is None:
        return
    free_bytes = disk_free_bytes(path)
    if free_bytes - required_bytes < reserve_bytes:
        raise RuntimeError(
            "Insufficient free space for next file: "
            + "need "
            + format_bytes(required_bytes)
            + " plus reserve "
            + format_bytes(reserve_bytes)
            + ", "
            + "free "
            + format_bytes(free_bytes)
            + " on "
            + str(path)
        )


def sha256sum(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_with_curl(
    item: DownloadItem,
    target_dir: Path,
    expected_size: int | None,
    connect_timeout: int,
    retries: int,
) -> Path:
    final_path = target_dir / item.relative_path
    part_path = Path(f"{final_path}.part")
    final_path.parent.mkdir(parents=True, exist_ok=True)

    if final_path.exists():
        if expected_size is None or final_path.stat().st_size == expected_size:
            print(f"[skip] {item.relative_path} already exists")
            return final_path
        print(
            "[warn] "
            + item.relative_path
            + " exists with size "
            + str(final_path.stat().st_size)
            + ", expected "
            + str(expected_size)
            + "; redownloading"
        )
        _ = final_path.unlink()

    command = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        str(retries),
        "--retry-delay",
        "5",
        "--connect-timeout",
        str(connect_timeout),
        "-C",
        "-",
        "-o",
        str(part_path),
        item.url,
    ]

    print(f"[download] {item.relative_path}")
    _ = subprocess.run(command, check=True)

    if expected_size is not None and part_path.stat().st_size != expected_size:
        raise RuntimeError(
            "Downloaded size mismatch for "
            + item.relative_path
            + ": got "
            + str(part_path.stat().st_size)
            + ", expected "
            + str(expected_size)
        )

    _ = part_path.replace(final_path)
    return final_path


def build_manifest(
    config: Config,
    items: list[DownloadItem],
    sizes: dict[str, int | None],
    downloaded_paths: dict[str, Path],
) -> Manifest:
    files: list[ManifestFile] = []
    for item in items:
        if item.key in downloaded_paths:
            path = downloaded_paths[item.key]
        else:
            path = config.target_dir / item.relative_path

        item_dict = asdict(item)
        entry: ManifestFile = {
            "key": cast(str, item_dict["key"]),
            "relative_path": cast(str, item_dict["relative_path"]),
            "url": cast(str, item_dict["url"]),
            "description": cast(str, item_dict["description"]),
            "required_for": cast(str, item_dict["required_for"]),
            "expected_size": sizes.get(item.key),
            "expected_size_human": format_bytes(sizes.get(item.key)),
            "path": str(path),
            "exists": path.exists(),
        }
        if path.exists():
            entry["size"] = path.stat().st_size
            entry["size_human"] = format_bytes(path.stat().st_size)
            entry["sha256"] = sha256sum(path)
        files.append(entry)

    return {
        "dataset": "MS MARCO Web Search SimANS",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "mode": config.mode,
        "include_doc_hash": config.include_doc_hash,
        "target_dir": str(config.target_dir),
        "official_sources": {
            "landing": OFFICIAL_LANDING,
            "readme": OFFICIAL_README,
        },
        "license_notice": "non-commercial research only",
        "files": files,
    }


def save_manifest(target_dir: Path, manifest: Manifest) -> Path:
    path = target_dir / "download_manifest.json"
    _ = target_dir.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def main() -> int:
    config = parse_args()
    ensure_curl_available()

    print(LICENSE_NOTICE)
    if not config.accept_license:
        print("\nRefusing to download without --accept-license", file=sys.stderr)
        return 2

    _ = config.target_dir.mkdir(parents=True, exist_ok=True)

    items = resolve_items(config)
    print(f"\nTarget directory: {config.target_dir}")
    print(f"Mode: {config.mode}")
    if config.include_doc_hash:
        print("Including doc_hash_mapping.tsv")

    sizes: dict[str, int | None] = {}
    total_known = 0
    for item in items:
        try:
            size = head_content_length(item.url)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] HEAD failed for {item.relative_path}: {exc}")
            size = None
        sizes[item.key] = size
        if size is not None:
            total_known += size
        print(
            " - "
            + item.relative_path
            + ": "
            + format_bytes(size)
            + " ("
            + item.required_for
            + ")"
        )

    free_now = disk_free_bytes(config.target_dir)
    print(f"Known total size: {format_bytes(total_known)}")
    print(f"Current free space: {format_bytes(free_now)}")
    print(f"Reserve requirement: {format_bytes(reserve_bytes_from_config(config))}")

    if config.dry_run:
        manifest = build_manifest(config, items, sizes, downloaded_paths={})
        manifest_path = save_manifest(config.target_dir, manifest)
        print(f"\n[dry-run] Manifest written to {manifest_path}")
        return 0

    reserve_bytes = reserve_bytes_from_config(config)
    downloaded_paths: dict[str, Path] = {}

    for item in items:
        ensure_space(config.target_dir, sizes[item.key], reserve_bytes)
        path = download_with_curl(
            item=item,
            target_dir=config.target_dir,
            expected_size=sizes[item.key],
            connect_timeout=config.connect_timeout,
            retries=config.retries,
        )
        downloaded_paths[item.key] = path
        time.sleep(1)

    manifest = build_manifest(config, items, sizes, downloaded_paths)
    manifest_path = save_manifest(config.target_dir, manifest)
    print(f"\n[done] Manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
