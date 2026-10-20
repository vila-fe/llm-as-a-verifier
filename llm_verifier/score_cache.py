"""Versioned, interruption-safe persistence for verifier score caches."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Union


CACHE_SCHEMA_VERSION = 1
CachePath = Union[str, os.PathLike]


def load_score_cache(cache_file: CachePath) -> Dict[str, Any]:
    """Load a score cache, accepting the unversioned legacy dict format.

    Versioned files hold ``{"schema_version": 1, "scores": {...}}``.  An
    unversioned mapping remains a supported read format so existing benchmark
    caches do not become invalid after upgrading.
    """

    with open(cache_file, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("score cache must contain a JSON object")
    if "schema_version" not in payload:
        return payload
    version = payload["schema_version"]
    if version != CACHE_SCHEMA_VERSION:
        raise ValueError(
            "unsupported score-cache schema version "
            f"{version!r}; expected {CACHE_SCHEMA_VERSION}"
        )
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("score-cache schema v1 requires an object field 'scores'")
    return scores


def _fsync_directory(directory: str) -> None:
    """Durably record the rename where the platform permits directory fsync."""

    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_score_cache(cache_file: CachePath, scores: Mapping[str, Any]) -> None:
    """Atomically replace ``cache_file`` with schema-v1 JSON score data.

    The write is durable before the replacement: JSON is written to a sibling
    temporary file, flushed, `fsync`ed, atomically renamed with `os.replace`,
    and the containing directory is fsynced.  If any stage fails, the old cache
    remains untouched and the temporary file is removed.
    """

    path = os.fspath(cache_file)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(
                {"schema_version": CACHE_SCHEMA_VERSION, "scores": dict(scores)},
                handle,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(directory)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise
