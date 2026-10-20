import json
import os

import pytest

from llm_verifier import score_cache


def test_save_cache_writes_schema_and_loads_scores_atomically(tmp_path, monkeypatch):
    """Catches a non-atomic replacement that can expose a partial cache file."""
    cache_path = tmp_path / "scores.json"
    legacy_scores = {"old|task|0,1|0": {"score_A": 0.25, "score_B": 0.75}}
    cache_path.write_text(json.dumps(legacy_scores), encoding="utf-8")
    scores = {"new|task|0,1|0": {"score_A": 1.0, "score_B": 0.0}}
    fsync_descriptors = []
    replace_observations = []
    real_fsync = os.fsync
    real_replace = os.replace

    def recording_fsync(fd):
        fsync_descriptors.append(fd)
        real_fsync(fd)

    def checked_replace(source, destination):
        replace_observations.append((source, destination, cache_path.read_text(encoding="utf-8")))
        assert os.path.exists(source)
        real_replace(source, destination)

    monkeypatch.setattr(score_cache.os, "fsync", recording_fsync)
    monkeypatch.setattr(score_cache.os, "replace", checked_replace)

    score_cache.save_score_cache(cache_path, scores)

    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    assert raw == {"schema_version": 1, "scores": scores}
    assert score_cache.load_score_cache(cache_path) == scores
    assert fsync_descriptors
    assert replace_observations == [
        (replace_observations[0][0], str(cache_path), json.dumps(legacy_scores))
    ]
    assert not list(tmp_path.glob(".scores.json.*.tmp"))


def test_load_cache_accepts_legacy_plain_score_mapping(tmp_path):
    """Catches a backwards-incompatible reader that discards existing score caches."""
    cache_path = tmp_path / "legacy.json"
    legacy_scores = {"criterion|task|0,1|0": {"score_A": 0.4, "score_B": 0.6}}
    cache_path.write_text(json.dumps(legacy_scores), encoding="utf-8")

    assert score_cache.load_score_cache(cache_path) == legacy_scores


def test_load_cache_rejects_an_unknown_schema_version(tmp_path):
    """Catches silent misinterpretation of a future cache schema."""
    cache_path = tmp_path / "future.json"
    cache_path.write_text(json.dumps({"schema_version": 999, "scores": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported score-cache schema version"):
        score_cache.load_score_cache(cache_path)
