"""Build the immutable Gate 5 DAPT corpus from governed source releases.

The validator can detect input-release drift, artifact mutation, forbidden
sources, cross-lane duplicates, and source leakage across splits. It cannot
establish semantic correctness of raw code beyond the upstream admission
contracts, or detect contamination outside the pinned JAXBench boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from opjax.pallas.corpus import validate_corpus_release
from opjax.pallas.hub_admission import (
    _jaccard,
    _normalise,
    _shingle_hashes,
    validate_hub_dapt_admission,
)
from opjax.pallas.phase2_contamination import assert_project_training_content_clean


SCHEMA_VERSION = 1


class G5CorpusError(RuntimeError):
    """The Gate 5 DAPT release violates its frozen contract."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise G5CorpusError(f"G5_JSON_INVALID: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise G5CorpusError(f"G5_JSON_OBJECT_REQUIRED: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise G5CorpusError(f"G5_ARTIFACT_MISSING: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise G5CorpusError(
                f"G5_JSONL_INVALID: {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise G5CorpusError(f"G5_JSONL_ROW_INVALID: {path}:{line_number}")
        rows.append(row)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _load_config(path: Path) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise G5CorpusError("G5_CONFIG_SCHEMA_INVALID")
    for key in (
        "repository_corpus_release_sha256",
        "hub_corpus_release_sha256",
    ):
        if not _valid_sha256(config.get(key)):
            raise G5CorpusError(f"G5_CONFIG_RELEASE_INVALID: {key}")
    for key in ("validation_source_ids", "forbidden_source_ids"):
        values = config.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise G5CorpusError(f"G5_CONFIG_SOURCE_LIST_INVALID: {key}")
    threshold = config.get("near_duplicate_threshold")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not 0 < threshold <= 1
    ):
        raise G5CorpusError("G5_CONFIG_DEDUP_THRESHOLD_INVALID")
    return config


def _corpus_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "repository_corpus_release_sha256": config[
            "repository_corpus_release_sha256"
        ],
        "hub_corpus_release_sha256": config["hub_corpus_release_sha256"],
        "validation_source_ids": config["validation_source_ids"],
        "forbidden_source_ids": config["forbidden_source_ids"],
        "near_duplicate_threshold": config["near_duplicate_threshold"],
    }


def _repository_row(
    row: Mapping[str, Any],
    *,
    validation_source_ids: set[str],
    forbidden_source_ids: set[str],
) -> dict[str, Any]:
    provenance = row.get("provenance")
    source_id = provenance.get("source_id") if isinstance(provenance, dict) else None
    text = row.get("text")
    if source_id in forbidden_source_ids:
        raise G5CorpusError(f"G5_FORBIDDEN_SOURCE: {source_id}:{row.get('row_id')}")
    if (
        row.get("objective") != "dapt"
        or not isinstance(row.get("row_id"), str)
        or not isinstance(text, str)
        or not text
        or not isinstance(source_id, str)
        or not source_id
    ):
        raise G5CorpusError(f"G5_REPOSITORY_ROW_INVALID: {row.get('row_id')}")
    return {
        "schema_version": SCHEMA_VERSION,
        "row_id": row["row_id"],
        "objective": "dapt",
        "lane": "pallas",
        "split": "validation" if source_id in validation_source_ids else "train",
        "repository": source_id,
        "family_id": row.get("family_id"),
        "text": text,
        "provenance": provenance,
    }


def _hub_row(row: Mapping[str, Any]) -> dict[str, Any]:
    provenance = row.get("provenance")
    repository = provenance.get("repository") if isinstance(provenance, dict) else None
    text = row.get("text")
    if (
        row.get("objective") != "dapt"
        or not isinstance(row.get("row_id"), str)
        or not isinstance(text, str)
        or not text
        or row.get("split") not in {"train", "validation"}
        or not isinstance(repository, str)
        or not repository
    ):
        raise G5CorpusError(f"G5_HUB_ROW_INVALID: {row.get('row_id')}")
    return {
        "schema_version": SCHEMA_VERSION,
        "row_id": row["row_id"],
        "objective": "dapt",
        "lane": "triton",
        "split": row["split"],
        "repository": repository,
        "family_id": row.get("family_id"),
        "text": text,
        "provenance": provenance,
    }


def _assert_cross_lane_disjoint(
    repo_rows: Sequence[Mapping[str, Any]],
    hub_rows: Sequence[Mapping[str, Any]],
    *,
    near_duplicate_threshold: float,
) -> None:
    exact: dict[str, str] = {}
    normalized: dict[str, str] = {}
    shingled: list[tuple[str, frozenset[str]]] = []
    for row in repo_rows:
        text = str(row["text"])
        exact[_text_sha256(text)] = str(row["row_id"])
        normalized[_text_sha256(_normalise(text))] = str(row["row_id"])
        shingled.append((str(row["row_id"]), _shingle_hashes(text)))
    for row in hub_rows:
        text = str(row["text"])
        row_id = str(row["row_id"])
        match = exact.get(_text_sha256(text))
        if match is not None:
            raise G5CorpusError(f"G5_CROSS_LANE_EXACT_DUPLICATE: {row_id}:{match}")
        match = normalized.get(_text_sha256(_normalise(text)))
        if match is not None:
            raise G5CorpusError(
                f"G5_CROSS_LANE_NORMALIZED_DUPLICATE: {row_id}:{match}"
            )
        shingles = _shingle_hashes(text)
        for candidate_id, candidate_shingles in shingled:
            similarity = _jaccard(shingles, candidate_shingles)
            if similarity >= near_duplicate_threshold:
                raise G5CorpusError(
                    "G5_CROSS_LANE_NEAR_DUPLICATE: "
                    f"{row_id}:{candidate_id}:{similarity:.6f}"
                )


def combine_dapt_rows(
    *,
    repo_rows: Sequence[Mapping[str, Any]],
    hub_rows: Sequence[Mapping[str, Any]],
    validation_source_ids: set[str],
    near_duplicate_threshold: float,
    forbidden_source_ids: set[str],
) -> list[dict[str, Any]]:
    repository = [
        _repository_row(
            row,
            validation_source_ids=validation_source_ids,
            forbidden_source_ids=forbidden_source_ids,
        )
        for row in repo_rows
    ]
    hub = [_hub_row(row) for row in hub_rows]
    _assert_cross_lane_disjoint(
        repository,
        hub,
        near_duplicate_threshold=near_duplicate_threshold,
    )
    combined = sorted([*repository, *hub], key=lambda row: row["row_id"])
    row_ids = [row["row_id"] for row in combined]
    if len(row_ids) != len(set(row_ids)):
        raise G5CorpusError("G5_ROW_ID_DUPLICATE")
    content_hashes = [_text_sha256(row["text"]) for row in combined]
    if len(content_hashes) != len(set(content_hashes)):
        raise G5CorpusError("G5_CONTENT_DUPLICATE")
    repositories_by_split = {
        split: {row["repository"] for row in combined if row["split"] == split}
        for split in ("train", "validation")
    }
    overlap = repositories_by_split["train"] & repositories_by_split["validation"]
    if overlap:
        raise G5CorpusError(f"G5_SPLIT_REPOSITORY_LEAKAGE: {sorted(overlap)}")
    return combined


def build_g5_dapt_release(
    *,
    repo_corpus_root: Path,
    hub_corpus_root: Path,
    config_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    config = _load_config(config_path)
    repo_validation = validate_corpus_release(repo_corpus_root)
    hub_validation = validate_hub_dapt_admission(hub_corpus_root)
    if repo_validation["release_sha256"] != config["repository_corpus_release_sha256"]:
        raise G5CorpusError("G5_REPOSITORY_RELEASE_MISMATCH")
    if hub_validation["release_sha256"] != config["hub_corpus_release_sha256"]:
        raise G5CorpusError("G5_HUB_RELEASE_MISMATCH")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise G5CorpusError(f"G5_OUTPUT_NOT_EMPTY: {out_dir}")
    repo_rows = _read_jsonl(repo_corpus_root / "datasets/dapt.jsonl")
    hub_rows = _read_jsonl(hub_corpus_root / "datasets/dapt.jsonl")
    rows = combine_dapt_rows(
        repo_rows=repo_rows,
        hub_rows=hub_rows,
        validation_source_ids=set(config["validation_source_ids"]),
        near_duplicate_threshold=float(config["near_duplicate_threshold"]),
        forbidden_source_ids=set(config["forbidden_source_ids"]),
    )
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    assert_project_training_content_clean(rows)
    _write_jsonl(out_dir / "datasets/dapt.jsonl", rows)
    _write_jsonl(out_dir / "datasets/train.jsonl", train_rows)
    _write_jsonl(out_dir / "datasets/validation.jsonl", validation_rows)
    artifact_paths = (
        "datasets/dapt.jsonl",
        "datasets/train.jsonl",
        "datasets/validation.jsonl",
    )
    artifacts = {relative: _file_sha256(out_dir / relative) for relative in artifact_paths}
    repo_sources = Counter(
        row["provenance"]["source_id"] for row in rows if row["lane"] == "pallas"
    )
    hub_repositories = {row["repository"] for row in rows if row["lane"] == "triton"}
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pallas_g5_dapt_corpus",
        "status": "complete",
        "experiment_id": config["experiment_id"],
        "source_releases": {
            "repository": repo_validation["release_sha256"],
            "hub": hub_validation["release_sha256"],
        },
        "config_sha256": _canonical_sha256(_corpus_contract(config)),
        "counts": {
            "rows": len(rows),
            "train": len(train_rows),
            "validation": len(validation_rows),
            "lanes": dict(sorted(Counter(row["lane"] for row in rows).items())),
            "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
            "sources": dict(sorted(repo_sources.items())),
            "hub_repositories": len(hub_repositories),
            "cross_lane_duplicates": 0,
        },
        "policy": {
            "validation_source_ids": sorted(config["validation_source_ids"]),
            "forbidden_source_ids": sorted(config["forbidden_source_ids"]),
            "near_duplicate_threshold": config["near_duplicate_threshold"],
            "repository_disjoint_splits": True,
            "jaxbench_contamination_checked_upstream": True,
        },
        "artifacts": artifacts,
    }
    manifest["release_sha256"] = _canonical_sha256(manifest)
    _write_json(out_dir / "manifest.json", manifest)
    validate_g5_dapt_release(out_dir)
    return manifest


def validate_g5_dapt_release(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "manifest.json")
    payload = dict(manifest)
    expected_release = payload.pop("release_sha256", None)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != "pallas_g5_dapt_corpus"
        or manifest.get("status") != "complete"
        or _canonical_sha256(payload) != expected_release
    ):
        raise G5CorpusError("G5_MANIFEST_INVALID")
    artifacts = manifest.get("artifacts")
    expected_paths = {
        "datasets/dapt.jsonl",
        "datasets/train.jsonl",
        "datasets/validation.jsonl",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_paths:
        raise G5CorpusError("G5_ARTIFACT_SET_INVALID")
    for relative, expected_hash in artifacts.items():
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or not path.is_file()
            or _file_sha256(path) != expected_hash
        ):
            raise G5CorpusError(f"G5_ARTIFACT_HASH_MISMATCH: {relative}")
    rows = _read_jsonl(root / "datasets/dapt.jsonl")
    train_rows = _read_jsonl(root / "datasets/train.jsonl")
    validation_rows = _read_jsonl(root / "datasets/validation.jsonl")
    if rows != sorted([*train_rows, *validation_rows], key=lambda row: row["row_id"]):
        raise G5CorpusError("G5_SPLIT_MATERIALIZATION_MISMATCH")
    row_ids = [row.get("row_id") for row in rows]
    if len(row_ids) != len(set(row_ids)) or row_ids != sorted(row_ids):
        raise G5CorpusError("G5_ROW_IDENTITY_INVALID")
    forbidden = set(manifest["policy"]["forbidden_source_ids"])
    for row in rows:
        provenance = row.get("provenance")
        source_id = provenance.get("source_id") if isinstance(provenance, dict) else None
        if (
            row.get("objective") != "dapt"
            or row.get("lane") not in {"pallas", "triton"}
            or row.get("split") not in {"train", "validation"}
            or not isinstance(row.get("repository"), str)
            or not isinstance(row.get("text"), str)
            or source_id in forbidden
        ):
            raise G5CorpusError(f"G5_ROW_INVALID: {row.get('row_id')}")
    repositories_by_split = {
        split: {row["repository"] for row in rows if row["split"] == split}
        for split in ("train", "validation")
    }
    if repositories_by_split["train"] & repositories_by_split["validation"]:
        raise G5CorpusError("G5_SPLIT_REPOSITORY_LEAKAGE")
    counts = manifest["counts"]
    observed_counts = {
        "rows": len(rows),
        "train": len(train_rows),
        "validation": len(validation_rows),
        "lanes": dict(sorted(Counter(row["lane"] for row in rows).items())),
        "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
        "sources": dict(
            sorted(
                Counter(
                    row["provenance"]["source_id"]
                    for row in rows
                    if row["lane"] == "pallas"
                ).items()
            )
        ),
        "hub_repositories": len(
            {row["repository"] for row in rows if row["lane"] == "triton"}
        ),
        "cross_lane_duplicates": 0,
    }
    if counts != observed_counts:
        raise G5CorpusError("G5_COUNTS_INVALID")
    return {
        "ok": True,
        "release_sha256": manifest["release_sha256"],
        "counts": counts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-g5-corpus")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--repository-corpus", type=Path, required=True)
    build.add_argument("--hub-corpus", type=Path, required=True)
    build.add_argument("--config", type=Path, default=Path("config/pallas/g5-dapt.json"))
    build.add_argument("--out-dir", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_g5_dapt_release(
                repo_corpus_root=args.repository_corpus,
                hub_corpus_root=args.hub_corpus,
                config_path=args.config,
                out_dir=args.out_dir,
            )
        else:
            result = validate_g5_dapt_release(args.root)
    except (G5CorpusError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"G5_CORPUS_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
